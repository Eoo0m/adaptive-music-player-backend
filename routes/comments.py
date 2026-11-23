import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from database import get_db_pool
from routes.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comments")


class CommentRequest(BaseModel):
    track_key: str
    text: str


@router.post("")
async def add_comment(request: CommentRequest, user: dict = Depends(get_current_user)):
    """트랙에 코멘트 추가"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="코멘트 내용이 필요합니다.")
    pool = await get_db_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO track_comments (user_id, track_key, text, created_at)
        VALUES ($1::uuid, $2, $3, NOW())
        RETURNING id, created_at
        """,
        user["user_id"], request.track_key, request.text.strip()
    )
    logger.info(f"Comment added: user={user['user_id']} track={request.track_key}")
    return {"success": True, "comment_id": row["id"], "created_at": row["created_at"].isoformat()}


@router.get("/my")
async def get_my_comments(user: dict = Depends(get_current_user)):
    """내 코멘트 목록 (최신순)"""
    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT
            c.id, c.track_key, c.text, c.created_at,
            t.title::text, t.artist::text, t.cover_image_url::text, t.apple_music_id::text
        FROM track_comments c
        JOIN tracks t ON t.track_key = c.track_key
        WHERE c.user_id = $1::uuid
        ORDER BY c.created_at DESC
        """,
        user["user_id"]
    )
    return {
        "comments": [
            {
                "id": r["id"],
                "track_key": r["track_key"],
                "text": r["text"],
                "created_at": r["created_at"].isoformat(),
                "title": r["title"],
                "artist": r["artist"],
                "cover_image_url": r["cover_image_url"],
                "apple_music_id": r["apple_music_id"],
            }
            for r in rows
        ]
    }


@router.delete("/{comment_id}")
async def delete_comment(comment_id: int, user: dict = Depends(get_current_user)):
    """코멘트 삭제"""
    pool = await get_db_pool()
    result = await pool.execute(
        "DELETE FROM track_comments WHERE id = $1 AND user_id = $2::uuid",
        comment_id, user["user_id"]
    )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="코멘트를 찾을 수 없습니다.")
    return {"success": True}
