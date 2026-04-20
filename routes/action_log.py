import logging
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List

from database import get_db_pool

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
