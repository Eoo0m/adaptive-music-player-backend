"""
Dual Attention 기반 추천 (/recommend)

알고리즘:
  1. 클릭한 곡(seed)의 SimGCL 64d 임베딩 조회
  2. seed 기준 ANN 100개 후보 선정 (track_vectors_simgcl, cosine)
  3. user_interest_clusters에서 k개 medoid 조회
     - 없거나 저장곡 <= 5개(클러스터 없음)면 ANN 유사도 순으로 바로 리턴
  4. Dual Attention 계산:
     - SIA = softmax(cos(seed,   medoids) / τ)   # seed의 클러스터 관심 분포
     - CIA = softmax(cos(cand,   medoids) / τ)   # 후보의 클러스터 관심 분포
     - Combined = normalize(SIA × CIA)            # 두 분포가 겹치는 클러스터 강조
     - Final Score = Σ_j (Combined_j × cos(cand, medoid_j))
  5. Final Score 내림차순 정렬 → num_recommendations개 리턴

τ (temperature) = 0.2
"""

import json
import logging
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from database import get_db_pool
from routes.auth import get_current_user
from telemetry import elapsed_ms, get_request_id, now_ms

logger = logging.getLogger(__name__)

router = APIRouter()

TEMPERATURE = 0.2
CANDIDATE_LIMIT = 100


class RecommendRequest(BaseModel):
    track_key: str
    num_recommendations: Optional[int] = 5
    exclude_track_keys: Optional[List[str]] = None


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = x / temperature
    x = x - x.max()          # numerical stability
    e = np.exp(x)
    return e / e.sum()


def _cosine_sim(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    a: (d,)  B: (k, d)
    returns: (k,) cosine similarities
    """
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norms = np.linalg.norm(B, axis=1, keepdims=True)
    B_norm = B / (b_norms + 1e-8)
    return B_norm @ a_norm


def _dual_attention_scores(
    seed_emb: np.ndarray,       # (d,)
    cand_embs: np.ndarray,      # (n, d)
    medoids: np.ndarray,        # (k, d)
) -> np.ndarray:
    """
    Returns final scores (n,) for each candidate.
    """
    k = medoids.shape[0]

    # SIA: seed → medoids cosine sim → softmax
    sia = _softmax(_cosine_sim(seed_emb, medoids), TEMPERATURE)   # (k,)

    # CIA: each candidate → medoids cosine sim → softmax
    # (n, k)
    cia = np.stack([
        _softmax(_cosine_sim(c, medoids), TEMPERATURE)
        for c in cand_embs
    ])

    # Combined = normalize(SIA * CIA)  element-wise per candidate
    combined = sia * cia                          # (n, k)
    combined_sum = combined.sum(axis=1, keepdims=True)
    combined_sum = np.where(combined_sum == 0, 1e-8, combined_sum)
    combined = combined / combined_sum            # (n, k) normalized

    # cos(candidate, medoid_j) for each candidate: (n, k)
    medoid_norms = np.linalg.norm(medoids, axis=1, keepdims=True)
    medoids_normed = medoids / (medoid_norms + 1e-8)
    cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
    cands_normed = cand_embs / (cand_norms + 1e-8)
    cos_cand_medoid = cands_normed @ medoids_normed.T    # (n, k)

    # Final Score = Σ_j combined_j * cos(cand, medoid_j)
    scores = (combined * cos_cand_medoid).sum(axis=1)   # (n,)
    return scores


@router.post("/recommend")
async def recommend(request: RecommendRequest, user: dict = Depends(get_current_user)):
    """Dual Attention 기반 추천"""
    t0 = now_ms()
    rid = get_request_id()
    target = max(1, request.num_recommendations or 5)
    excluded = set(request.exclude_track_keys or [])

    logger.info(f"[{rid}] /recommend seed={request.track_key} target={target}")

    pool = await get_db_pool()

    # 1. seed 임베딩 조회
    seed_row = await pool.fetchrow(
        "SELECT embedding FROM track_embeddings WHERE track_key = $1",
        request.track_key
    )
    if not seed_row or not seed_row["embedding"]:
        raise HTTPException(status_code=404, detail="Seed track embedding not found")

    seed_raw = seed_row["embedding"]
    if isinstance(seed_raw, str):
        seed_raw = json.loads(seed_raw)
    seed_emb = np.array(seed_raw, dtype=np.float32)
    seed_vec_str = str(seed_raw)

    # 2. ANN 100개 후보 선정 (SimGCL, cosine)
    t_ann = now_ms()
    exclude_keys = list(excluded | {request.track_key})
    rows = await pool.fetch(
        """
        SELECT
            t.track_key,
            t.title::text,
            t.artist::text,
            t.album::text,
            t.cover_image_url::text,
            t.apple_music_id::text,
            t.playlist_count,
            te.embedding,
            (1 - (te.embedding <=> $1::vector))::float AS ann_sim
        FROM track_embeddings te
        JOIN tracks t ON te.track_key = t.track_key
        WHERE te.track_key != ALL($2)
        ORDER BY te.embedding <=> $1::vector
        LIMIT $3
        """,
        seed_vec_str, exclude_keys, CANDIDATE_LIMIT
    )
    logger.info(f"[{rid}] ANN={elapsed_ms(t_ann):.1f}ms candidates={len(rows)}")

    if not rows:
        raise HTTPException(status_code=500, detail="No candidates found")

    # 3. 클러스터 조회
    cluster_row = await pool.fetchrow(
        "SELECT k, medoids FROM user_interest_clusters WHERE user_id = $1::uuid",
        user["user_id"]
    )

    # 클러스터 없으면 ANN 유사도 순으로 바로 리턴
    if not cluster_row:
        logger.info(f"[{rid}] no clusters → fallback to ANN order")
        result = _format_rows(rows, excluded, target, score_key="ann_sim")
        logger.info(f"[{rid}] total={elapsed_ms(t0):.1f}ms recs={len(result)}")
        return {"recommendations": result, "mode": "ann_fallback"}

    # 4. Dual Attention 스코어 계산
    t_attn = now_ms()
    medoids_raw = json.loads(cluster_row["medoids"]) if isinstance(cluster_row["medoids"], str) else cluster_row["medoids"]
    medoids = np.array(medoids_raw, dtype=np.float32)   # (k, d)

    cand_embs = []
    cand_rows = []
    for row in rows:
        emb = row["embedding"]
        if isinstance(emb, str):
            emb = json.loads(emb)
        cand_embs.append(emb)
        cand_rows.append(row)

    cand_embs = np.array(cand_embs, dtype=np.float32)   # (n, d)
    scores = _dual_attention_scores(seed_emb, cand_embs, medoids)
    logger.info(f"[{rid}] dual_attention={elapsed_ms(t_attn):.1f}ms k={cluster_row['k']}")

    # 5. 스코어 내림차순 정렬 → 리턴
    order = np.argsort(-scores)
    sorted_rows = [cand_rows[i] for i in order]
    result = _format_rows(sorted_rows, excluded, target, score_key=None, scores=[float(scores[i]) for i in order])

    logger.info(f"[{rid}] total={elapsed_ms(t0):.1f}ms recs={len(result)} mode=dual_attention")
    return {"recommendations": result, "mode": "dual_attention", "k": int(cluster_row["k"])}


def _format_rows(rows, excluded: set, target: int, score_key: Optional[str], scores: list = None):
    fresh, reused = [], []
    score_iter = iter(scores) if scores else None

    for i, row in enumerate(rows):
        score = next(score_iter) if score_iter else (row[score_key] if score_key else 0.0)
        item = {
            "track_key": row["track_key"],
            "track": row["title"],
            "artist": row["artist"],
            "album": row["album"],
            "playlist_count": row["playlist_count"],
            "cover_image_url": row["cover_image_url"],
            "apple_music_id": row["apple_music_id"],
            "similarity": score,
        }
        if row["track_key"] not in excluded:
            fresh.append(item)
        else:
            reused.append(item)

    return (fresh + reused)[:target]
