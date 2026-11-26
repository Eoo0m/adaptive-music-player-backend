"""
트랙 목록 → 쿼리 추천 (/query-suggest)

알고리즘:
  1. track_keys의 SimGCL 64d 임베딩 조회
  2. Ward Hierarchical Clustering → medoid 선택
  3. medoid → playlist_proj → CLIP shared space (512d)
  4. 플리 캡션 임베딩들(caption_embs.npy)과 cosine 유사도
  5. top-k 캡션 선택 → 각 캡션으로 mood 검색 (ANN + MMR)
"""

import asyncio
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
from utils import get_openai_embedding_with_retry, mmr_rerank
from telemetry import elapsed_ms, get_request_id, now_ms

logger = logging.getLogger(__name__)

router = APIRouter()

ALPHA = 2.0
MAX_QUERIES = 3

# 서버 시작 시 main.py에서 주입
_caption_embs: Optional[np.ndarray] = None   # (N, 512)
_caption_texts: Optional[np.ndarray] = None  # (N,)


def set_caption_state(embs: np.ndarray, texts: np.ndarray):
    global _caption_embs, _caption_texts
    _caption_embs = embs
    _caption_texts = texts


class QuerySuggestRequest(BaseModel):
    track_keys: List[str]
    user_id: Optional[str] = None   # 있으면 user_interest_clusters 캐시 우선 사용
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


async def _mood_search_for_query(query: str, pool, top_k: int = 9) -> list:
    """쿼리 텍스트 → OpenAI 임베딩 → CLIP proj → ANN + MMR → 트랙 목록"""
    row = await pool.fetchrow(
        "SELECT embedding::text FROM keyword_embeddings WHERE keyword = $1",
        query.lower()
    )
    if row and row["embedding"]:
        raw_emb = np.array(json.loads(row["embedding"]), dtype=np.float32)
    else:
        raw_emb = np.array(get_openai_embedding_with_retry(query), dtype=np.float32)
        try:
            await pool.execute(
                "INSERT INTO keyword_embeddings (keyword, embedding) VALUES ($1, $2::vector) ON CONFLICT DO NOTHING",
                query.lower(), str(raw_emb.tolist())
            )
        except Exception:
            pass

    with torch.no_grad():
        t = torch.tensor(raw_emb).unsqueeze(0).to(device)
        proj = playlist_clip_model.caption_proj(t).cpu().numpy()[0]  # (512,)

    rows = await pool.fetch(
        """
        SELECT
            te.track_key::text,
            t.title::text,
            t.artist::text,
            t.cover_image_url::text,
            t.apple_music_id::text,
            t.playlist_count,
            te.projected_embedding::text
        FROM track_vectors_clip te
        JOIN tracks t ON te.track_key = t.track_key
        ORDER BY te.projected_embedding <=> $1::vector
        LIMIT 80
        """,
        str(proj.tolist()),
    )
    if not rows:
        return []

    cand_ids, cand_embs, cand_data = [], [], {}
    for r in rows:
        emb_str = r["projected_embedding"]
        if emb_str:
            emb = np.array(json.loads(emb_str), dtype=np.float32)
            cand_ids.append(r["track_key"])
            cand_embs.append(emb)
            cand_data[r["track_key"]] = {
                "track_key": r["track_key"],
                "track": r["title"],
                "track_name": r["title"],
                "artist": r["artist"],
                "cover_image_url": r["cover_image_url"],
                "apple_music_id": r["apple_music_id"],
                "playlist_count": r["playlist_count"],
            }

    selected_ids, _ = mmr_rerank(proj, np.array(cand_embs), cand_ids, top_k=top_k, lambda_param=0.7)
    return [cand_data[k] for k in selected_ids if k in cand_data]


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

    # 1. user_interest_clusters 캐시 우선 사용
    medoid_embs = None
    if request.user_id:
        cluster_row = await pool.fetchrow(
            "SELECT medoids::text FROM user_interest_clusters WHERE user_id = $1",
            request.user_id,
        )
        if cluster_row and cluster_row["medoids"]:
            medoid_embs = np.array(json.loads(cluster_row["medoids"]), dtype=np.float32)

    if medoid_embs is None:
        # SimGCL 임베딩 조회 → clustering
        rows = await pool.fetch(
            "SELECT track_key, embedding::text FROM track_vectors_simgcl WHERE track_key = ANY($1)",
            request.track_keys,
        )
        if not rows:
            raise HTTPException(status_code=404, detail="No embeddings found for given track_keys")

        embs = [json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"] for r in rows]
        medoid_embs = _compute_medoids(np.array(embs, dtype=np.float32))

    # 2. playlist_proj → CLIP space
    with torch.no_grad():
        t = torch.tensor(medoid_embs).to(device)
        medoid_proj = playlist_clip_model.playlist_proj(t).cpu().numpy()  # (k, 512)

    # 4. 캡션 임베딩과 cosine 유사도 → top-k 캡션 선택
    max_q = request.max_queries or MAX_QUERIES
    cap_norm = _caption_embs / (np.linalg.norm(_caption_embs, axis=1, keepdims=True) + 1e-8)
    med_norm = medoid_proj / (np.linalg.norm(medoid_proj, axis=1, keepdims=True) + 1e-8)
    sims = (med_norm @ cap_norm.T).mean(axis=0)  # (N,)

    top_idx = np.argsort(-sims)
    queries, seen = [], set()
    for idx in top_idx:
        q = str(_caption_texts[idx])
        if q not in seen:
            seen.add(q)
            queries.append(q)
        if len(queries) >= max_q:
            break

    # 5. 각 쿼리별 mood 검색 병렬
    track_results = await asyncio.gather(
        *[_mood_search_for_query(q, pool) for q in queries]
    )

    result_items = [
        {"query": q, "tracks": tracks}
        for q, tracks in zip(queries, track_results)
    ]

    logger.info(f"[{rid}] total={elapsed_ms(t0):.1f}ms k={len(medoid_embs)} queries={queries}")
    return {"suggestions": result_items, "k": len(medoid_embs)}
