"""
유저 취향 클러스터 계산 및 저장

알고리즘:
  1. 유저 최근 저장 50곡의 SimGCL 64d 임베딩 조회
  2. 저장곡 <= 5개면 스킵
  3. Global Pooling 벡터 계산 (전체 임베딩 L2-정규화 평균)
  4. Ward Hierarchical Clustering 수행
  5. 거리 임계값 ALPHA로 클러스터 수 자동 결정
  6. 각 클러스터의 Medoid 선택 → user_interest_clusters upsert
"""

import json
import logging
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from scipy.cluster.hierarchy import linkage, fcluster

from database import get_db_pool
from routes.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

MIN_TRACKS_FOR_CLUSTERING = 6   # 5개 이하면 스킵
MAX_FAVORITES = 50
ALPHA = 2.0                     # Ward 거리 임계값: 이 거리 이상이면 별도 클러스터


def _global_pooling(embeddings: np.ndarray) -> list:
    """
    전체 임베딩의 L2-정규화 평균 (Global Pooling 벡터).
    각 벡터를 먼저 L2 정규화한 뒤 평균 → 다시 L2 정규화.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    normed = embeddings / norms
    pooled = normed.mean(axis=0)
    p_norm = np.linalg.norm(pooled)
    if p_norm > 0:
        pooled = pooled / p_norm
    return pooled.tolist()


def _compute_clusters(embeddings: np.ndarray, track_keys: list[str]):
    """
    Ward clustering + 거리 임계값 ALPHA로 K 자동 결정 + Medoid + Global Pooling.
    Returns: (k, medoid_vectors, medoid_track_keys, global_vector)
    """
    n = len(embeddings)
    global_vec = _global_pooling(embeddings)

    if n == 1:
        return 1, embeddings.tolist(), track_keys, global_vec

    Z = linkage(embeddings, method="ward", metric="euclidean")
    labels = fcluster(Z, ALPHA, criterion="distance")
    k = len(np.unique(labels))

    medoid_vectors = []
    medoid_keys = []

    for c in np.unique(labels):
        indices = np.where(labels == c)[0]
        cluster_embs = embeddings[indices]

        # Medoid: 클러스터 내 평균에 cosine similarity 가장 가까운 실제 벡터
        centroid = cluster_embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        sims = (cluster_embs / norms) @ centroid
        medoid_idx = int(np.argmax(sims))

        medoid_vectors.append(cluster_embs[medoid_idx].tolist())
        medoid_keys.append(track_keys[indices[medoid_idx]])

    return k, medoid_vectors, medoid_keys, global_vec


async def recompute_clusters(user_id: str):
    """
    favorites POST 후 백그라운드로 호출.
    clustering 수행 후 user_interest_clusters upsert.
    """
    try:
        pool = await get_db_pool()

        # 최근 저장 50곡 임베딩 조회
        rows = await pool.fetch(
            """
            SELECT f.track_key, te.embedding
            FROM user_favorites f
            JOIN track_embeddings te ON f.track_key = te.track_key
            WHERE f.user_id = $1::uuid
              AND te.embedding IS NOT NULL
            ORDER BY f.created_at DESC
            LIMIT $2
            """,
            user_id, MAX_FAVORITES
        )

        if len(rows) <= MIN_TRACKS_FOR_CLUSTERING - 1:
            logger.info(f"[cluster] user={user_id} tracks={len(rows)} <= {MIN_TRACKS_FOR_CLUSTERING - 1}, skip")
            return

        track_keys = []
        emb_list = []
        for row in rows:
            emb = row["embedding"]
            if isinstance(emb, str):
                emb = json.loads(emb)
            track_keys.append(row["track_key"])
            emb_list.append(emb)

        embeddings = np.array(emb_list, dtype=np.float32)

        k, medoid_vectors, medoid_keys, global_vec = _compute_clusters(embeddings, track_keys)

        await pool.execute(
            """
            INSERT INTO user_interest_clusters (user_id, k, medoids, track_keys, global_embedding, updated_at)
            VALUES ($1::uuid, $2, $3::jsonb, $4::jsonb, $5::jsonb, NOW())
            ON CONFLICT (user_id) DO UPDATE
              SET k = EXCLUDED.k,
                  medoids = EXCLUDED.medoids,
                  track_keys = EXCLUDED.track_keys,
                  global_embedding = EXCLUDED.global_embedding,
                  updated_at = NOW()
            """,
            user_id,
            k,
            json.dumps(medoid_vectors),
            json.dumps(medoid_keys),
            json.dumps(global_vec),
        )

        logger.info(f"[interest_profile] user={user_id} k={k} medoids={medoid_keys}")

    except Exception as e:
        logger.error(f"[interest_profile] recompute_clusters error user={user_id}: {e}")


@router.post("/interest-profile")
async def interest_profile(user: dict = Depends(get_current_user)):
    """
    수동으로 interest profile 재계산 트리거 (테스트/디버그용).
    favorites POST에서 자동 호출되므로 보통 직접 호출 불필요.
    """
    await recompute_clusters(user["user_id"])
    pool = await get_db_pool()
    row = await pool.fetchrow(
        "SELECT k, track_keys FROM user_interest_clusters WHERE user_id = $1::uuid",
        user["user_id"]
    )
    if not row:
        return {"k": 0, "medoid_track_keys": [], "skipped": True}
    return {"k": row["k"], "medoid_track_keys": json.loads(row["track_keys"]), "skipped": False}
