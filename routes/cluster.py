"""
유저 취향 클러스터 계산 및 저장

알고리즘:
  1. 유저 최근 저장 50곡의 SimGCL 64d 임베딩 조회
  2. 저장곡 <= 5개면 스킵
  3. Ward Hierarchical Clustering 수행
  4. K=2~5 각각에 대해 Silhouette 점수 평가 → 최고 K 선택
  5. 최소 클러스터 크기(MIN_CLUSTER_SIZE) 미달 클러스터는 가장 가까운 클러스터에 병합
  6. 각 클러스터의 Medoid 선택 → user_interest_clusters upsert
"""

import json
import logging
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score

from database import get_db_pool
from routes.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

MIN_TRACKS_FOR_CLUSTERING = 6   # 5개 이하면 스킵
MAX_FAVORITES = 50
K_MIN, K_MAX = 2, 5
MIN_CLUSTER_SIZE = 3            # 이보다 작은 클러스터는 병합


def _best_k(embeddings: np.ndarray, Z: np.ndarray) -> int:
    """
    K=2~5에 대해 Silhouette 점수 계산 → 최고 K 반환.
    n이 작아서 K를 만들 수 없으면 가능한 최대 K로 클램프.
    """
    n = len(embeddings)
    best_k, best_score = K_MIN, -1.0

    for k in range(K_MIN, min(K_MAX, n - 1) + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        # 클러스터가 실제로 k개 미만으로 나뉘면 스킵
        if len(np.unique(labels)) < k:
            continue
        score = silhouette_score(embeddings, labels, metric="euclidean")
        if score > best_score:
            best_score, best_k = score, k

    return best_k


def _merge_small_clusters(
    embeddings: np.ndarray,
    track_keys: list[str],
    labels: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """
    min_size 미달 클러스터의 각 포인트를 가장 가까운 대형 클러스터 centroid로 재배정.
    반복하여 모든 클러스터가 min_size 이상이 될 때까지 수행.
    """
    labels = labels.copy()
    for _ in range(10):  # 최대 10회 반복 (무한루프 방지)
        unique, counts = np.unique(labels, return_counts=True)
        small = unique[counts < min_size]
        large = unique[counts >= min_size]
        if len(small) == 0 or len(large) == 0:
            break

        # 대형 클러스터 centroid
        centroids = np.array([
            embeddings[labels == c].mean(axis=0) for c in large
        ])

        for c in small:
            for idx in np.where(labels == c)[0]:
                dists = np.linalg.norm(centroids - embeddings[idx], axis=1)
                labels[idx] = large[int(np.argmin(dists))]

    # label을 1부터 연속으로 재번호
    mapping = {old: new for new, old in enumerate(np.unique(labels), start=1)}
    return np.array([mapping[l] for l in labels])


def _compute_clusters(embeddings: np.ndarray, track_keys: list[str]):
    """
    Ward clustering + Silhouette K 선택 + 최소 크기 필터 + Medoid.
    Returns: (k, medoid_vectors, medoid_track_keys)
    """
    n = len(embeddings)

    if n == 1:
        return 1, embeddings.tolist(), track_keys

    Z = linkage(embeddings, method="ward", metric="euclidean")
    k = _best_k(embeddings, Z)
    labels = fcluster(Z, k, criterion="maxclust")

    # 최소 크기 필터
    labels = _merge_small_clusters(embeddings, track_keys, labels, MIN_CLUSTER_SIZE)
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
