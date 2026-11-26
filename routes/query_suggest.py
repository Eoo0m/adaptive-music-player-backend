"""
트랙 목록 → 쿼리 추천 (/query-suggest)

알고리즘:
  1. track_keys의 SimGCL 64d 임베딩 조회
  2. Ward Hierarchical Clustering → medoid 선택
  3. medoid → playlist_proj → CLIP shared space (512d)
  4. 플리 캡션 임베딩들(caption_embs.npy)과 cosine 유사도
  5. top-40 풀에서 20개 선택 → user_query_suggestions에 쿼리 텍스트만 저장
  6. 반환 시 랜덤 3개 뽑아 query_mood_cache에서 트랙 조회
"""

import json
import logging
import numpy as np
import torch

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from scipy.cluster.hierarchy import fcluster, linkage

from database import get_db_pool
from models import playlist_clip_model, device
from telemetry import elapsed_ms, get_request_id, now_ms

logger = logging.getLogger(__name__)

router = APIRouter()

ALPHA = 2.0
MAX_QUERIES = 3
CACHE_SIZE = 20

# 서버 시작 시 main.py에서 주입
_caption_embs: Optional[np.ndarray] = None   # (N, 512)
_caption_texts: Optional[np.ndarray] = None  # (N,)


def set_caption_state(embs: np.ndarray, texts: np.ndarray):
    global _caption_embs, _caption_texts
    _caption_embs = embs
    _caption_texts = texts


class QuerySuggestRequest(BaseModel):
    track_keys: List[str]
    user_id: Optional[str] = None
    max_queries: Optional[int] = MAX_QUERIES


def _compute_medoids(embeddings: np.ndarray) -> np.ndarray:
    """Ward clustering → 각 클러스터 medoid 반환."""
    n = len(embeddings)
    if n == 1:
        return embeddings

    Z = linkage(embeddings, method="ward", metric="euclidean")
    labels = fcluster(Z, ALPHA, criterion="distance")

    medoids = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        cluster_embs = embeddings[idx]
        centroid = cluster_embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        sims = (cluster_embs / norms) @ centroid
        medoids.append(cluster_embs[int(np.argmax(sims))])

    return np.array(medoids, dtype=np.float32)


def _select_queries(medoid_embs: np.ndarray, size: int) -> list:
    """medoid 임베딩 → caption 유사도 → top-40 풀에서 size개 랜덤 선택."""
    with torch.no_grad():
        t = torch.tensor(medoid_embs).to(device)
        medoid_proj = playlist_clip_model.playlist_proj(t).cpu().numpy()

    cap_norm = _caption_embs / (np.linalg.norm(_caption_embs, axis=1, keepdims=True) + 1e-8)
    med_norm = medoid_proj / (np.linalg.norm(medoid_proj, axis=1, keepdims=True) + 1e-8)
    sims = (med_norm @ cap_norm.T).mean(axis=0)

    pool_size = min(40, len(sims))
    top_pool_idx = np.argsort(-sims)[:pool_size]
    chosen_idx = np.random.choice(top_pool_idx, size=min(size, pool_size), replace=False)
    return [str(_caption_texts[i]) for i in chosen_idx]


async def compute_and_cache_suggestions(user_id: str, medoid_embs: np.ndarray, pool):
    """medoid 임베딩 → 쿼리 20개 선택 → user_query_suggestions에 텍스트만 저장"""
    try:
        queries = _select_queries(medoid_embs, CACHE_SIZE)
        await pool.execute(
            """
            INSERT INTO user_query_suggestions (user_id, suggestions, updated_at)
            VALUES ($1, $2::jsonb, NOW())
            ON CONFLICT (user_id) DO UPDATE
              SET suggestions = EXCLUDED.suggestions, updated_at = NOW()
            """,
            user_id, json.dumps(queries)
        )
        logger.info(f"[query_suggest] cached for user={user_id} queries={queries}")
    except Exception as e:
        logger.error(f"[query_suggest] compute_and_cache error user={user_id}: {e}")


@router.post("/query-suggest")
async def query_suggest(request: QuerySuggestRequest):
    """트랙 목록 → 추천 검색 쿼리 + 각 쿼리의 mood 검색 결과 반환"""
    t0 = now_ms()
    rid = get_request_id()

    if _caption_embs is None:
        raise HTTPException(status_code=503, detail="Caption embeddings not loaded yet")
    if not request.track_keys:
        raise HTTPException(status_code=400, detail="track_keys is empty")

    logger.info(f"[{rid}] /query-suggest tracks={len(request.track_keys)}")

    pool = await get_db_pool()
    max_q = request.max_queries or MAX_QUERIES

    # 1. user_query_suggestions 캐시에서 쿼리 텍스트 조회
    queries = None
    if request.user_id:
        cache_row = await pool.fetchrow(
            "SELECT suggestions FROM user_query_suggestions WHERE user_id = $1",
            request.user_id,
        )
        if cache_row:
            all_queries = json.loads(cache_row["suggestions"])
            chosen = np.random.choice(len(all_queries), size=min(max_q, len(all_queries)), replace=False)
            queries = [all_queries[i] for i in chosen]

    # 2. 캐시 없으면 직접 계산
    if queries is None:
        medoid_embs = None
        if request.user_id:
            cluster_row = await pool.fetchrow(
                "SELECT medoids::text FROM user_interest_clusters WHERE user_id = $1",
                request.user_id,
            )
            if cluster_row and cluster_row["medoids"]:
                medoid_embs = np.array(json.loads(cluster_row["medoids"]), dtype=np.float32)

        if medoid_embs is None:
            rows = await pool.fetch(
                "SELECT track_key, embedding::text FROM track_vectors_simgcl WHERE track_key = ANY($1)",
                request.track_keys,
            )
            if not rows:
                raise HTTPException(status_code=404, detail="No embeddings found for given track_keys")
            embs = [json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"] for r in rows]
            medoid_embs = _compute_medoids(np.array(embs, dtype=np.float32))

        queries = _select_queries(medoid_embs, max_q)

    # 3. query_mood_cache에서 트랙 조회
    cache_rows = await pool.fetch(
        "SELECT query, tracks FROM query_mood_cache WHERE query = ANY($1)",
        queries,
    )
    track_map = {r["query"]: json.loads(r["tracks"]) for r in cache_rows}

    result_items = [
        {"query": q, "tracks": track_map.get(q, [])}
        for q in queries
    ]

    logger.info(f"[{rid}] total={elapsed_ms(t0):.1f}ms queries={queries}")
    return {"suggestions": result_items, "k": len(result_items)}
