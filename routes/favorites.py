import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from database import get_db_pool
from routes.auth import get_current_user

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
            t.playlist_count
        FROM user_favorites f
        JOIN track_embeddings t ON f.track_key = t.track_key
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
                "playlist_count": row["playlist_count"],
                "added_at": row["created_at"].isoformat(),
            }
            for row in rows
        ]
    }


@router.post("")
async def add_favorite(request: FavoriteRequest, user: dict = Depends(get_current_user)):
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
        return {"success": True}
    except Exception as e:
        logger.error(f"Add favorite error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add favorite")


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
