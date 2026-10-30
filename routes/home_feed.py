import random
import logging
from fastapi import APIRouter, HTTPException, Depends

from database import get_db_pool
from routes.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/home-feed")
async def home_feed(user: dict = Depends(get_current_user)):
    """홈 피드: 찜한 곡 중 3개를 골라 각각 유사곡 10개 반환"""
    pool = await get_db_pool()

    # 1. 유저의 찜 목록 가져오기
    fav_rows = await pool.fetch(
        """
        SELECT f.track_key, t.title::text, t.artist::text, t.album::text,
               t.cover_image_url::text, t.apple_music_id::text, t.playlist_count
        FROM user_favorites f
        JOIN tracks t ON f.track_key = t.track_key
        WHERE f.user_id = $1::uuid
        """,
        user["user_id"]
    )

    if not fav_rows:
        return {"feeds": []}

    # 2. 랜덤 3곡 선택 (3곡 미만이면 전부)
    seed_count = min(3, len(fav_rows))
    seeds = random.sample(list(fav_rows), seed_count)

    feeds = []
    for seed in seeds:
        # 3. 각 seed 곡의 유사곡 10개 조회 (2-step: HNSW 인덱스 보장)
        query_row = await pool.fetchrow(
            "SELECT embedding::text FROM track_vectors_simgcl WHERE track_key = $1",
            seed["track_key"]
        )
        if not query_row:
            continue
        similar_rows = await pool.fetch(
            """
            SELECT
                t.track_key::text,
                t.title::text,
                t.artist::text,
                t.album::text,
                t.cover_image_url::text,
                t.apple_music_id::text,
                t.playlist_count,
                (1 - (v.embedding <=> $2::vector))::float AS similarity
            FROM track_vectors_simgcl v
            JOIN tracks t ON v.track_key = t.track_key
            WHERE v.track_key != $1
            ORDER BY v.embedding <=> $2::vector
            LIMIT 12
            """,
            seed["track_key"], query_row["embedding"]
        )

        feeds.append({
            "seed_track": {
                "track_key": seed["track_key"],
                "title": seed["title"],
                "artist": seed["artist"],
                "album": seed["album"],
                "cover_image_url": seed["cover_image_url"],
                "apple_music_id": seed["apple_music_id"],
                "playlist_count": seed["playlist_count"],
            },
            "similar": [
                {
                    "track_key": r["track_key"],
                    "title": r["title"],
                    "artist": r["artist"],
                    "album": r["album"],
                    "cover_image_url": r["cover_image_url"],
                    "apple_music_id": r["apple_music_id"],
                    "playlist_count": r["playlist_count"],
                    "similarity": r["similarity"],
                }
                for r in similar_rows
            ]
        })

    logger.info(f"Home feed: {len(feeds)} sections for user={user['user_id']}")
    return {"feeds": feeds}
