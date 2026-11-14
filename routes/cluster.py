"""
유저 취향 클러스터 계산 및 저장

알고리즘:
  1. 유저 최근 저장 50곡의 SimGCL 64d 임베딩 조회
  2. 저장곡 <= 5개면 스킵
  3. Ward Hierarchical Clustering 수행
  4. 병합 거리 변화량(elbow) 기준으로 K 자동 결정 (1~5)
  5. 각 클러스터의 Medoid (실제 트랙 벡터 중 클러스터 내 평균과 가장 가까운 것) 선택
  6. user_interest_clusters 테이블에 upsert
"""

import json
import logging
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from database import get_db_pool
from routes.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

MIN_TRACKS_FOR_CLUSTERING = 6   # 5개 이하면 스킵
MAX_FAVORITES = 50
K_MIN, K_MAX = 1, 5


def _auto_k(Z: np.ndarray) -> int:
    """
    Ward linkage matrix Z에서 elbow 방식으로 K 결정.
    병합 거리(Z[:,2])의 차분이 최대인 지점을 기준으로 K 결정.
    K_MIN ~ K_MAX 범위 내로 클램프.
    """
    n = Z.shape[0] + 1          # 원본 데이터 포인트 수
    max_k = min(K_MAX, n - 1)
    if max_k <= K_MIN:
        return K_MIN

    # 마지막 max_k개 병합 거리 — 큰 값일수록 나중에 합쳐진 것
    distances = Z[-max_k:, 2]          # shape: (max_k,)

    if len(distances) < 2:
        return K_MIN

    # distances[i] 는 K=(max_k - i) 클러스터에서의 마지막 병합 거리
    # diffs[i] 가 크면 K=(max_k - i) → K=(max_k - i - 1) 병합이 큰 도약
    # → K=(max_k - i) 에서 멈추는 것이 적절
    diffs = np.diff(distances)                      # shape: (max_k - 1,)

    # 전체 거리 범위 대비 15% 이상인 점프를 "유의미한 분리"로 간주
    dist_range = distances[-1] - distances[0] + 1e-8
    threshold = dist_range * 0.15
    jump_indices = np.where(diffs >= threshold)[0]  # diffs[i] >= threshold

    if len(jump_indices) == 0:
        return K_MIN

    # 유의미한 점프 중 가장 첫 번째(인덱스 작은 것) → 더 세분화된 K 선택
    # (마지막 jump는 K=2→K=1로 합치는 가장 큰 병합이므로 제외)
    elbow_idx = int(jump_indices[0])                # diffs[elbow_idx] 가 첫 유의미한 점프
    k = max_k - elbow_idx                           # 이 점프 직전 K
    return max(K_MIN, min(K_MAX, k))


def _compute_clusters(embeddings: np.ndarray, track_keys: list[str]):
    """
    Ward clustering + Medoid 계산.
    Returns: (k, medoid_vectors, medoid_track_keys)
    """
    n = len(embeddings)

    # 단일 곡이면 그 자체가 클러스터
    if n == 1:
        return 1, embeddings.tolist(), track_keys

    # Ward linkage (condensed distance matrix 기반)
    Z = linkage(embeddings, method="ward", metric="euclidean")
    k = _auto_k(Z)

    # K개 클러스터로 분할 (1-based label)
    labels = fcluster(Z, k, criterion="maxclust")  # shape: (n,)

    medoid_vectors = []
    medoid_keys = []

    for c in range(1, k + 1):
        indices = np.where(labels == c)[0]
        cluster_embs = embeddings[indices]

        # Medoid: 클러스터 내 평균과 cosine distance 가장 가까운 실제 벡터
        centroid = cluster_embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # 각 멤버와 centroid 간 cosine similarity
        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        normed = cluster_embs / norms
        sims = normed @ centroid
        medoid_idx = int(np.argmax(sims))

        medoid_vectors.append(cluster_embs[medoid_idx].tolist())
        medoid_keys.append(track_keys[indices[medoid_idx]])

    return k, medoid_vectors, medoid_keys


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

        k, medoid_vectors, medoid_keys = _compute_clusters(embeddings, track_keys)

        await pool.execute(
            """
            INSERT INTO user_interest_clusters (user_id, k, medoids, track_keys, updated_at)
            VALUES ($1::uuid, $2, $3::jsonb, $4::jsonb, NOW())
            ON CONFLICT (user_id) DO UPDATE
              SET k = EXCLUDED.k,
                  medoids = EXCLUDED.medoids,
                  track_keys = EXCLUDED.track_keys,
                  updated_at = NOW()
            """,
            user_id,
            k,
            json.dumps(medoid_vectors),
            json.dumps(medoid_keys),
        )

        logger.info(f"[cluster] user={user_id} k={k} medoids={medoid_keys}")

    except Exception as e:
        logger.error(f"[cluster] recompute_clusters error user={user_id}: {e}")


@router.post("/cluster")
async def cluster(user: dict = Depends(get_current_user)):
    """
    수동으로 클러스터 재계산 트리거 (테스트/디버그용).
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
