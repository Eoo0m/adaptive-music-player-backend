"""
/smart-search: 의도 파악 후 엔티티 + 무드 결과 동시 반환

흐름:
1. OpenAI 임베딩 1번 (3072d)
2. Intent Classifier → mood 여부 판정 (동기, CPU, ~1ms)
3. CLIP 프로젝션 (동기, CPU, ~1ms)
4. asyncio.gather:
   - 엔티티 검색: trigram 텍스트 쿼리 (항상 실행)
   - 무드 검색: projected_embedding ANN + MMR (mood일 때만)
5. 통합 반환
"""

import asyncio
import json as json_module
import logging
import numpy as np
import torch
import torch.nn.functional as F

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from database import get_db_pool
from models import playlist_clip_model, intent_classifier, INTENT_THRESHOLD, device
from utils import get_openai_embedding_with_retry, mmr_rerank
from telemetry import elapsed_ms, get_request_id, now_ms


async def search_tracks_by_title_raw(query_text: str, match_count: int = 10):
    """제목 기반 트랙 검색 (raw SQL)"""
    rid = get_request_id()
    t0 = now_ms()
    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT
            id,
            track_key::text,
            title::text,
            artist::text,
            album::text,
            release_year,
            playlist_count,
            cover_image_url::text,
            apple_music_id::text,
            similarity(lower(title || ' ' || artist), lower($1))::float AS similarity
        FROM tracks
        WHERE lower(title || ' ' || artist) % lower($1)
           OR lower(title) ILIKE $1 || '%'
           OR lower(artist) ILIKE $1 || '%'
        ORDER BY
            CASE WHEN lower(title) ILIKE $1 || '%' THEN 0 ELSE 1 END,
            similarity(lower(title || ' ' || artist), lower($1)) DESC
        LIMIT $2
        """,
        query_text,
        match_count
    )
    logger.info(f"[{rid}] search_tracks_by_title total={elapsed_ms(t0):.1f}ms rows={len(rows)}")
    return [dict(row) for row in rows]

logger = logging.getLogger(__name__)

router = APIRouter()


class SmartSearchRequest(BaseModel):
    query: str
    entity_limit: Optional[int] = 20
    mood_top_k: Optional[int] = 40
    mood_candidates: Optional[int] = 100
    mmr_lambda: Optional[float] = 0.7
    user_id: Optional[str] = None


async def _mood_search(projected_embedding_np: np.ndarray, candidates: int, top_k: int, mmr_lambda: float):
    """CLIP projected_embedding 기반 무드 검색 + MMR"""
    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT
            te.track_key::text,
            t.title::text,
            t.artist::text,
            t.album::text,
            t.release_year,
            t.playlist_count,
            t.cover_image_url::text,
            t.apple_music_id::text,
            te.projected_embedding::text
        FROM track_vectors_clip te
        JOIN tracks t ON te.track_key = t.track_key
        ORDER BY te.projected_embedding <=> $1::vector
        LIMIT $2
        """,
        str(projected_embedding_np.tolist()),
        candidates,
    )

    if not rows:
        return []

    candidate_ids = []
    candidate_embs = []
    candidate_data = {}

    for row in rows:
        emb_str = row["projected_embedding"]
        if emb_str:
            emb = np.array(json_module.loads(emb_str), dtype=np.float32)
            candidate_ids.append(row["track_key"])
            candidate_embs.append(emb)
            candidate_data[row["track_key"]] = {
                "track_key": row["track_key"],
                "track": row["title"],
                "track_name": row["title"],
                "artist": row["artist"],
                "album": row["album"],
                "release_year": row["release_year"],
                "playlist_count": row["playlist_count"],
                "cover_image_url": row["cover_image_url"],
                "apple_music_id": row["apple_music_id"],
            }

    selected_ids, selected_scores = mmr_rerank(
        projected_embedding_np,
        np.array(candidate_embs),
        candidate_ids,
        top_k=top_k,
        lambda_param=mmr_lambda,
    )

    results = []
    for track_key, score in zip(selected_ids, selected_scores):
        item = dict(candidate_data[track_key])
        item["similarity"] = score
        results.append(item)
    return results


async def _query_suggest_for_search(search_proj: np.ndarray, user_id: Optional[str], pool) -> list:
    """검색어 임베딩 + user medoid 하이브리드 → query_mood_cache에서 top 3 쿼리+트랙 반환"""
    try:
        from routes.query_suggest import _caption_embs, _caption_texts

        if _caption_embs is None:
            return []

        # user medoid 조회
        medoid_proj = None
        if user_id:
            row = await pool.fetchrow(
                "SELECT medoids::text FROM user_interest_clusters WHERE user_id = $1",
                user_id,
            )
            if row and row["medoids"]:
                import json as _json
                medoid_embs = np.array(_json.loads(row["medoids"]), dtype=np.float32)
                with torch.no_grad():
                    t = torch.tensor(medoid_embs).to(device)
                    medoid_proj = playlist_clip_model.playlist_proj(t).cpu().numpy()  # (k, 512)

        # caption 유사도: 검색어 0.5 + medoid 평균 0.5 (medoid 없으면 검색어만)
        cap_norm = _caption_embs / (np.linalg.norm(_caption_embs, axis=1, keepdims=True) + 1e-8)
        q_norm = search_proj / (np.linalg.norm(search_proj) + 1e-8)
        sims = cap_norm @ q_norm  # (N,)

        if medoid_proj is not None:
            med_norm = medoid_proj / (np.linalg.norm(medoid_proj, axis=1, keepdims=True) + 1e-8)
            med_sims = (med_norm @ cap_norm.T).mean(axis=0)
            sims = 0.5 * sims + 0.5 * med_sims

        top_idx = np.argsort(-sims)[:3]
        queries = [str(_caption_texts[i]) for i in top_idx]

        # query_mood_cache에서 트랙 조회
        cache_rows = await pool.fetch(
            "SELECT query, tracks FROM query_mood_cache WHERE query = ANY($1)",
            queries,
        )
        track_map = {r["query"]: json_module.loads(r["tracks"]) for r in cache_rows}

        return [{"query": q, "tracks": track_map.get(q, [])} for q in queries]
    except Exception as e:
        logger.warning(f"[smart-search] query_suggest_for_search error: {e}")
        return []


async def _get_embedding(query: str, pool) -> tuple[np.ndarray, bool]:
    """캐시된 임베딩 조회, 없으면 OpenAI 호출. (embedding_np, cache_hit) 반환"""
    row = await pool.fetchrow(
        "SELECT embedding::text FROM keyword_embeddings WHERE keyword = $1",
        query.lower()
    )
    if row and row["embedding"]:
        emb = np.array(json_module.loads(row["embedding"]), dtype=np.float32)
        return emb, True
    raw_emb = get_openai_embedding_with_retry(query)
    return np.array(raw_emb, dtype=np.float32), False


@router.post("/smart-search")
async def smart_search(request: SmartSearchRequest):
    """의도 분류 → 엔티티/무드 결과 동시 반환"""
    request_id = get_request_id()
    start_ms = now_ms()

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Missing query")

    query = request.query.strip()

    try:
        # 1. 임베딩 (캐시 → 없으면 OpenAI)
        t_emb = now_ms()
        pool = await get_db_pool()
        emb_np, cache_hit = await _get_embedding(query, pool)
        emb_ms = elapsed_ms(t_emb)
        logger.info(f"[{request_id}] /smart-search emb={'cache' if cache_hit else 'openai'}={emb_ms:.1f}ms query='{query}'")

        emb_tensor = torch.tensor(emb_np).unsqueeze(0).to(device)

        # 2. Intent 판정 + CLIP 프로젝션 (동시, CPU ~1ms each)
        with torch.no_grad():
            intent_logits = intent_classifier(emb_tensor)
            mood_prob = F.softmax(intent_logits, dim=1)[0, 1].item()
            is_mood = mood_prob >= INTENT_THRESHOLD

            projected = playlist_clip_model.caption_proj(emb_tensor)
            projected_np = projected.cpu().numpy()[0]

        logger.info(f"[{request_id}] /smart-search intent={'mood' if is_mood else 'non'} mood_prob={mood_prob:.3f}")

        # 3. 엔티티 + 무드 + 쿼리 추천 병렬
        entity_coro = search_tracks_by_title_raw(query, match_count=request.entity_limit)
        mood_coro = (
            _mood_search(projected_np, request.mood_candidates, request.mood_top_k, request.mmr_lambda)
            if is_mood
            else asyncio.sleep(0)
        )
        query_suggest_coro = _query_suggest_for_search(projected_np, request.user_id, pool)

        entity_rows, mood_result, query_suggestions = await asyncio.gather(
            entity_coro, mood_coro, query_suggest_coro
        )

        # 엔티티 포맷
        entity_results = [
            {
                "track_key": r["track_key"],
                "track": r["title"],
                "track_name": r["title"],
                "artist": r["artist"],
                "album": r["album"],
                "release_year": r.get("release_year"),
                "playlist_count": r["playlist_count"],
                "similarity": r.get("similarity", 0),
                "cover_image_url": r.get("cover_image_url"),
                "apple_music_id": r.get("apple_music_id"),
            }
            for r in entity_rows
        ]

        mood_results = mood_result if isinstance(mood_result, list) else []

        total_ms = elapsed_ms(start_ms)
        logger.info(
            f"[{request_id}] /smart-search done total={total_ms:.1f}ms "
            f"entity={len(entity_results)} mood={len(mood_results)} suggestions={len(query_suggestions)}"
        )

        return {
            "query": query,
            "is_mood": is_mood,
            "mood_prob": round(mood_prob, 3),
            "entity_results": entity_results,
            "mood_results": mood_results,
            "query_suggestions": query_suggestions,
        }

    except Exception as e:
        logger.error(f"[{request_id}] /smart-search error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
