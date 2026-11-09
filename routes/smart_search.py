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
from routes.search import search_tracks_by_title_raw

logger = logging.getLogger(__name__)

router = APIRouter()


class SmartSearchRequest(BaseModel):
    query: str
    entity_limit: Optional[int] = 10
    mood_top_k: Optional[int] = 24
    mood_candidates: Optional[int] = 100
    mmr_lambda: Optional[float] = 0.7


async def _mood_search(projected_embedding_np: np.ndarray, candidates: int, top_k: int, mmr_lambda: float):
    """CLIP projected_embedding 기반 무드 검색 + MMR"""
    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT
            te.track_key::text,
            te.title::text,
            te.artist::text,
            te.album::text,
            te.playlist_count,
            te.cover_image_url::text,
            te.apple_music_id::text,
            te.projected_embedding::text
        FROM track_embeddings te
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


@router.post("/smart-search")
async def smart_search(request: SmartSearchRequest):
    """의도 분류 → 엔티티/무드 결과 동시 반환"""
    request_id = get_request_id()
    start_ms = now_ms()

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Missing query")

    query = request.query.strip()

    try:
        # 1. OpenAI 임베딩 (1번만)
        t_emb = now_ms()
        raw_emb = get_openai_embedding_with_retry(query)
        emb_ms = elapsed_ms(t_emb)
        logger.info(f"[{request_id}] /smart-search openai={emb_ms:.1f}ms query='{query}'")

        emb_np = np.array(raw_emb, dtype=np.float32)
        emb_tensor = torch.tensor(emb_np).unsqueeze(0).to(device)

        # 2. Intent 판정 + CLIP 프로젝션 (동시, CPU ~1ms each)
        with torch.no_grad():
            intent_logits = intent_classifier(emb_tensor)
            mood_prob = F.softmax(intent_logits, dim=1)[0, 1].item()
            is_mood = mood_prob >= INTENT_THRESHOLD

            projected = playlist_clip_model.caption_proj(emb_tensor)
            projected_np = projected.cpu().numpy()[0]

        logger.info(f"[{request_id}] /smart-search intent={'mood' if is_mood else 'non'} mood_prob={mood_prob:.3f}")

        # 3. 엔티티 + 무드 병렬
        entity_coro = search_tracks_by_title_raw(query, match_count=request.entity_limit)
        mood_coro = (
            _mood_search(projected_np, request.mood_candidates, request.mood_top_k, request.mmr_lambda)
            if is_mood
            else asyncio.sleep(0)
        )

        entity_rows, mood_result = await asyncio.gather(entity_coro, mood_coro)

        # 엔티티 포맷
        entity_results = [
            {
                "track_key": r["track_key"],
                "track": r["title"],
                "track_name": r["title"],
                "artist": r["artist"],
                "album": r["album"],
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
            f"entity={len(entity_results)} mood={len(mood_results)}"
        )

        return {
            "query": query,
            "is_mood": is_mood,
            "mood_prob": round(mood_prob, 3),
            "entity_results": entity_results,
            "mood_results": mood_results,
        }

    except Exception as e:
        logger.error(f"[{request_id}] /smart-search error: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
