import logging
from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import Optional, List

from database import get_db_pool
from routes.auth import get_optional_user

logger = logging.getLogger(__name__)

router = APIRouter()


class ActionLogRequest(BaseModel):
    session_id: str
    action_type: str  # search | select_from_search | select_from_recommend | favorite | play
    search_query: Optional[str] = None
    search_mode: Optional[str] = None
    selected_track_key: Optional[str] = None
    candidate_track_keys: Optional[List[str]] = None
    extra: Optional[str] = None


@router.post("/log-action")
async def log_action(request: ActionLogRequest, req: Request):
    """유저 행동 로그 저장"""
    valid_types = {"search", "select_from_search", "select_from_recommend", "favorite", "play"}
    if request.action_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid action_type: {request.action_type}")

    # JWT에서 user_id 추출 (비로그인이면 None)
    user_id = None
    auth_header = req.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            from routes.auth import JWT_SECRET, JWT_ALGORITHM
            from jose import jwt, JWTError
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("sub")
        except Exception:
            pass  # 토큰 무효해도 로그는 저장

    try:
        pool = await get_db_pool()
        await pool.execute(
            """
            INSERT INTO user_action_logs (session_id, user_id, action_type, search_query, search_mode, selected_track_key, candidate_track_keys, extra)
            VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8)
            """,
            request.session_id,
            user_id,
            request.action_type,
            request.search_query,
            request.search_mode,
            request.selected_track_key,
            request.candidate_track_keys,
            request.extra
        )

        logger.info(f"Action logged: {request.action_type} | session={request.session_id} | user={user_id}")
        return {"success": True}

    except Exception as e:
        logger.error(f"Action log error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to log action")


@router.get("/recent-plays")
async def recent_plays(limit: int = 20, user: dict | None = Depends(get_optional_user), req: Request = None):
    """최근 재생 트랙 목록 (유저 기반, 중복 제거)"""
    user_id = user["user_id"] if user else None

    # 비로그인이면 session_id로 대체
    session_id = None
    if not user_id and req:
        session_id = req.headers.get("X-Session-ID")

    if not user_id and not session_id:
        return {"tracks": []}

    try:
        pool = await get_db_pool()

        if user_id:
            rows = await pool.fetch(
                """
                SELECT DISTINCT ON (l.selected_track_key)
                    l.selected_track_key AS track_key,
                    l.created_at,
                    t.title, t.artist, t.cover_image_url, t.apple_music_id
                FROM user_action_logs l
                JOIN tracks t ON t.track_key = l.selected_track_key
                WHERE l.action_type = 'play'
                  AND l.user_id = $1::uuid
                  AND l.selected_track_key IS NOT NULL
                ORDER BY l.selected_track_key, l.created_at DESC
                LIMIT $2
                """,
                user_id, limit * 3
            )
        else:
            rows = await pool.fetch(
                """
                SELECT DISTINCT ON (l.selected_track_key)
                    l.selected_track_key AS track_key,
                    l.created_at,
                    t.title, t.artist, t.cover_image_url, t.apple_music_id
                FROM user_action_logs l
                JOIN tracks t ON t.track_key = l.selected_track_key
                WHERE l.action_type = 'play'
                  AND l.session_id = $1
                  AND l.selected_track_key IS NOT NULL
                ORDER BY l.selected_track_key, l.created_at DESC
                LIMIT $2
                """,
                session_id, limit * 3
            )

        # created_at 기준 최신순 정렬 후 limit 적용
        sorted_rows = sorted(rows, key=lambda r: r["created_at"], reverse=True)[:limit]

        tracks = [
            {
                "track_key": r["track_key"],
                "title": r["title"],
                "artist": r["artist"],
                "cover_image_url": r["cover_image_url"],
                "apple_music_id": r["apple_music_id"],
                "played_at": r["created_at"].isoformat(),
            }
            for r in sorted_rows
        ]
        return {"tracks": tracks}

    except Exception as e:
        logger.error(f"Recent plays error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch recent plays")
