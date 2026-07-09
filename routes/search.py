import logging
import random
import json as json_module
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from database import get_db_pool
from models import playlist_clip_model, device
from utils import get_openai_embedding_with_retry, mmr_rerank
from telemetry import elapsed_ms, get_request_id, now_ms

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic 모델
class SearchRequest(BaseModel):
    query: str


class KeywordSearchRequest(BaseModel):
    keyword: str
    top_k: Optional[int] = 10
    mmr_lambda: Optional[float] = 0.7
    mmr_candidates: Optional[int] = 50


class RecommendRequest(BaseModel):
    track_key: str
    num_recommendations: Optional[int] = 30


# ============== 제목 검색 헬퍼 ==============

async def search_tracks_by_title_raw(query_text: str, match_count: int = 10):
    """제목 기반 트랙 검색 (raw SQL)"""
    request_id = get_request_id()
    start_ms = now_ms()
    logger.info(f"[{request_id}] search_tracks_by_title start query='{query_text}'")

    pool = await get_db_pool()
    db_start_ms = now_ms()
    rows = await pool.fetch(
        """
        SELECT
            id,
            track_key::text,
            title::text,
            artist::text,
            album::text,
            playlist_count,
            cover_image_url::text,
            similarity(lower(title || ' ' || artist), lower($1))::float AS similarity
        FROM track_embeddings
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

    logger.info(
        f"[{request_id}] search_tracks_by_title db={elapsed_ms(db_start_ms):.1f}ms "
        f"total={elapsed_ms(start_ms):.1f}ms rows={len(rows)}"
    )
    return [dict(row) for row in rows]


# ============== Search Endpoints ==============


@router.post("/search-songs")
async def search_songs(request: SearchRequest):
    """제목 기반 검색"""
    if not request.query:
        raise HTTPException(status_code=400, detail="Missing query")

    try:
        rows = await search_tracks_by_title_raw(
            query_text=request.query,
            match_count=10
        )

        # 결과 포맷 변환
        results = []
        for item in rows:
            results.append({
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "playlist_count": item["playlist_count"],
                "similarity": item.get("similarity", 0),
                "cover_image_url": item.get("cover_image_url"),
            })

        logger.info(f"Search: found {len(results)} tracks for '{request.query}'")
        return {"results": results}

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Search service unavailable: {str(e)}"
        )


@router.post("/find-similar-tracks")
async def find_similar_tracks(request: RecommendRequest):
    """track_key 기반 유사 음악 추천 (검색 결과 클릭 시 사용)"""
    request_start_ms = now_ms()
    request_id = get_request_id()

    if not request.track_key:
        raise HTTPException(status_code=400, detail="Missing track_key")

    try:
        pool = await get_db_pool()

        # 50곡 가져오기 (raw SQL)
        db_start_ms = now_ms()
        rows = await pool.fetch(
            """
            WITH query_track AS (
                SELECT embedding
                FROM track_embeddings
                WHERE track_key = $1
                LIMIT 1
            )
            SELECT
                t.id,
                t.track_key::text,
                t.title::text,
                t.artist::text,
                t.album::text,
                t.playlist_count,
                t.cover_image_url::text,
                (1 - (t.embedding <#> q.embedding) / 2)::float AS similarity
            FROM track_embeddings t, query_track q
            WHERE t.track_key != $1
              AND t.embedding IS NOT NULL
            ORDER BY t.embedding <#> q.embedding
            LIMIT $2
            """,
            request.track_key,
            50
        )
        logger.info(
            f"[{request_id}] /find-similar-tracks vector_search="
            f"{elapsed_ms(db_start_ms):.1f}ms rows={len(rows)}"
        )

        if not rows:
            raise HTTPException(status_code=500, detail="Recommendation failed")

        # 결과 포맷 변환
        format_start_ms = now_ms()
        all_recommendations = []
        for item in rows:
            all_recommendations.append({
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "playlist_count": item["playlist_count"],
                "similarity": item.get("similarity", 0),
                "cover_image_url": item.get("cover_image_url"),
            })

        # 랜덤하게 선택
        selected_count = min(request.num_recommendations, len(all_recommendations))
        recommendations = random.sample(all_recommendations, selected_count) if len(all_recommendations) > selected_count else all_recommendations

        # 로그에 선택된 곡 정보 출력
        selected_titles = [f"{r['track']} - {r['artist']}" for r in recommendations]
        logger.info(
            f"[{request_id}] /find-similar-tracks format={elapsed_ms(format_start_ms):.1f}ms "
            f"total={elapsed_ms(request_start_ms):.1f}ms selected={len(recommendations)} "
            f"candidates={len(all_recommendations)} track_key='{request.track_key}'"
        )
        logger.info(f"[{request_id}] selected tracks: {', '.join(selected_titles[:5])}{'...' if len(selected_titles) > 5 else ''}")

        return {
            "recommendations": recommendations,
            "original_song": {"track_key": request.track_key},
        }

    except Exception as e:
        logger.error(f"Recommend error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Recommendation service unavailable: {str(e)}"
        )


# ============== Keyword Search ==============


@router.post("/search-by-keyword")
async def search_by_keyword(request: KeywordSearchRequest):
    """키워드 기반 검색 (CLIP + MMR 다양성 적용)"""
    request_start_ms = now_ms()
    request_id = get_request_id()

    if not request.keyword:
        raise HTTPException(status_code=400, detail="Missing keyword")

    try:
        logger.info(
            f"[{request_id}] /search-by-keyword start keyword='{request.keyword}' "
            f"top_k={request.top_k} mmr_lambda={request.mmr_lambda}"
        )

        # 타이밍 측정
        timings = {}

        # 1. OpenAI 임베딩
        t_openai = now_ms()
        keyword_embedding = get_openai_embedding_with_retry(request.keyword)
        openai_ms = elapsed_ms(t_openai)
        timings["openai"] = openai_ms / 1000
        logger.info(f"[{request_id}] /search-by-keyword openai={openai_ms:.1f}ms")

        # 2. CLIP projection (텍스트 → 512차원 트랙 임베딩 공간)
        t_proj = now_ms()
        keyword_tensor = torch.tensor(keyword_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            projected_query = playlist_clip_model.caption_proj(keyword_tensor)
            projected_embedding_np = projected_query.cpu().numpy()[0]
        projection_ms = elapsed_ms(t_proj)
        timings["projection"] = projection_ms / 1000
        logger.info(f"[{request_id}] /search-by-keyword projection={projection_ms:.1f}ms")

        # 3. MMR 후보 가져오기 (raw SQL)
        t_db = now_ms()
        pool = await get_db_pool()
        rows = await pool.fetch(
            """
            SELECT
                t.track_key::text,
                t.title::text,
                t.artist::text,
                t.album::text,
                t.playlist_count,
                t.cover_image_url::text,
                t.projected_embedding::text
            FROM track_embeddings t
            WHERE t.projected_embedding IS NOT NULL
            ORDER BY t.projected_embedding <#> $1::vector
            LIMIT $2
            """,
            str(projected_embedding_np.tolist()),
            request.mmr_candidates
        )
        db_ms = elapsed_ms(t_db)
        timings["db"] = db_ms / 1000
        logger.info(f"[{request_id}] /search-by-keyword db={db_ms:.1f}ms rows={len(rows)}")

        if not rows:
            logger.warning("No results from projected_embedding search")
            return {"results": []}

        # 4. MMR Reranking
        t_mmr = now_ms()
        candidate_ids = []
        candidate_embs = []
        candidate_data = {}

        for row in rows:
            track_key = row["track_key"]
            emb_str = row["projected_embedding"]
            if emb_str:
                emb = np.array(json_module.loads(emb_str), dtype=np.float32)
                candidate_ids.append(track_key)
                candidate_embs.append(emb)
                candidate_data[track_key] = {
                    "track_key": track_key,
                    "track_name": row["title"],
                    "artist": row["artist"],
                    "album": row["album"],
                    "playlist_count": row["playlist_count"],
                    "cover_image_url": row["cover_image_url"],
                }

        candidate_embs = np.array(candidate_embs)

        # MMR reranking 적용
        selected_ids, selected_scores = mmr_rerank(
            projected_embedding_np,
            candidate_embs,
            candidate_ids,
            top_k=request.top_k,
            lambda_param=request.mmr_lambda
        )
        mmr_ms = elapsed_ms(t_mmr)
        timings["mmr"] = mmr_ms / 1000
        logger.info(f"[{request_id}] /search-by-keyword mmr={mmr_ms:.1f}ms")

        # 5. 결과 포맷 변환
        format_start_ms = now_ms()
        results = []
        for track_key, score in zip(selected_ids, selected_scores):
            data = candidate_data[track_key]
            data["similarity"] = score
            results.append(data)
        format_ms = elapsed_ms(format_start_ms)
        timings["format"] = format_ms / 1000

        logger.info(
            f"[{request_id}] /search-by-keyword format={format_ms:.1f}ms "
            f"total={elapsed_ms(request_start_ms):.1f}ms results={len(results)}"
        )
        return {"results": results, "timings": timings}

    except Exception as e:
        logger.error(f"Keyword search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {str(e)}")
