import json
import logging
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from scipy.cluster.hierarchy import linkage, fcluster

from database import get_db_pool
from config import T, TV_SIMGCL
from routes.auth import get_current_user
from routes.interest_profile import recompute_clusters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/favorites")


class FavoriteRequest(BaseModel):
    track_key: str


@router.get("")
async def get_favorites(user: dict = Depends(get_current_user)):
    """내 관심 리스트 조회"""
    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT
            f.track_key,
            f.created_at,
            t.title::text,
            t.artist::text,
            t.album::text,
            t.cover_image_url::text,
            t.apple_music_id::text,
            t.playlist_count
        FROM user_favorites f
        JOIN {T} t ON f.track_key = t.track_key
        WHERE f.user_id = $1::uuid
        ORDER BY f.created_at DESC
        """,
        user["user_id"]
    )

    return {
        "favorites": [
            {
                "track_key": row["track_key"],
                "title": row["title"],
                "artist": row["artist"],
                "album": row["album"],
                "cover_image_url": row["cover_image_url"],
                "apple_music_id": row["apple_music_id"],
                "playlist_count": row["playlist_count"],
                "added_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
    }


@router.post("")
async def add_favorite(
    request: FavoriteRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """곡 찜하기"""
    pool = await get_db_pool()
    try:
        await pool.execute(
            """
            INSERT INTO user_favorites (user_id, track_key)
            VALUES ($1::uuid, $2)
            ON CONFLICT (user_id, track_key) DO NOTHING
            """,
            user["user_id"], request.track_key
        )
        logger.info(f"Favorite added: user={user['user_id']}, track={request.track_key}")
        # 클러스터 재계산 (백그라운드)
        background_tasks.add_task(recompute_clusters, user["user_id"])
        return {"success": True}
    except Exception as e:
        logger.error(f"Add favorite error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add favorite")


@router.get("/clustered")
async def get_favorites_clustered(user: dict = Depends(get_current_user)):
    """찜 목록을 Ward 클러스터링으로 그룹핑해서 반환"""
    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT
            f.track_key, f.created_at,
            t.title::text, t.artist::text, t.album::text,
            t.cover_image_url::text, t.apple_music_id::text, t.playlist_count,
            te.embedding
        FROM user_favorites f
        JOIN {T} t ON f.track_key = t.track_key
        LEFT JOIN {TV_SIMGCL} te ON te.track_key = f.track_key
        WHERE f.user_id = $1::uuid
        ORDER BY f.created_at DESC
        """,
        user["user_id"]
    )

    if not rows:
        return {"clusters": []}

    # 임베딩 없는 곡 분리
    valid_rows = [r for r in rows if r["embedding"] is not None]
    no_emb_rows = [r for r in rows if r["embedding"] is None]

    def row_to_track(r):
        return {
            "track_key": r["track_key"],
            "title": r["title"],
            "artist": r["artist"],
            "album": r["album"],
            "cover_image_url": r["cover_image_url"],
            "apple_music_id": r["apple_music_id"],
            "playlist_count": r["playlist_count"],
            "added_at": r["created_at"].isoformat(),
        }

    if len(valid_rows) < 3:
        # 곡이 너무 적으면 클러스터 1개로 반환
        return {"clusters": [{"cluster_id": 1, "tracks": [row_to_track(r) for r in rows]}]}

    embs = np.array([
        json.loads(r["embedding"]) if isinstance(r["embedding"], str) else list(r["embedding"])
        for r in valid_rows
    ], dtype=np.float32)

    Z = linkage(embs, method="ward", metric="euclidean")
    labels = fcluster(Z, 1.5, criterion="distance")

    # label → (track 목록, centroid 임베딩) 매핑
    cluster_tracks = {}
    cluster_embs = {}
    for i, lbl in enumerate(labels):
        lbl = int(lbl)
        cluster_tracks.setdefault(lbl, []).append((i, row_to_track(valid_rows[i])))
        cluster_embs.setdefault(lbl, []).append(embs[i])

    # 각 클러스터의 centroid 계산
    centroids = {lbl: np.mean(np.stack(e), axis=0) for lbl, e in cluster_embs.items()}

    # 작은 클러스터(1~3곡)를 가장 가까운 큰 클러스터에 병합
    MIN_SIZE = 4
    large_ids = {lbl for lbl, t in cluster_tracks.items() if len(t) >= MIN_SIZE}
    small_ids = {lbl for lbl, t in cluster_tracks.items() if len(t) < MIN_SIZE}

    # 큰 클러스터가 없으면 그냥 모두 반환 (전체 곡 수가 적은 경우)
    if large_ids:
        large_centroids = np.stack([centroids[lbl] for lbl in sorted(large_ids)])
        large_ids_sorted = sorted(large_ids)

        for small_lbl in small_ids:
            c = centroids[small_lbl]
            # cosine similarity로 가장 가까운 큰 클러스터 찾기
            c_norm = c / (np.linalg.norm(c) + 1e-8)
            lc_norms = large_centroids / (np.linalg.norm(large_centroids, axis=1, keepdims=True) + 1e-8)
            sims = lc_norms @ c_norm
            nearest = large_ids_sorted[int(np.argmax(sims))]
            cluster_tracks[nearest].extend(cluster_tracks[small_lbl])
            del cluster_tracks[small_lbl]

    # 결과 조립 (cluster_id 재번호 부여)
    final_clusters = {}
    for new_id, (lbl, items) in enumerate(sorted(cluster_tracks.items()), start=1):
        final_clusters[new_id] = [t for _, t in items]

    # 임베딩 없는 곡은 마지막 클러스터에 추가
    if no_emb_rows:
        last_id = max(final_clusters.keys()) + 1
        final_clusters[last_id] = [row_to_track(r) for r in no_emb_rows]

    result = [
        {"cluster_id": cid, "tracks": tracks}
        for cid, tracks in sorted(final_clusters.items())
    ]
    return {"clusters": result}


@router.delete("/{track_key}")
async def remove_favorite(track_key: str, user: dict = Depends(get_current_user)):
    """찜 해제"""
    pool = await get_db_pool()
    result = await pool.execute(
        """
        DELETE FROM user_favorites
        WHERE user_id = $1::uuid AND track_key = $2
        """,
        user["user_id"], track_key
    )

    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Favorite not found")

    logger.info(f"Favorite removed: user={user['user_id']}, track={track_key}")
    return {"success": True}
