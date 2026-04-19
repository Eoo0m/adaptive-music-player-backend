import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from supabase import create_client, Client

logger = logging.getLogger(__name__)

router = APIRouter()

# Supabase 클라이언트
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


class ListeningLogRequest(BaseModel):
    track_name: str
    artist_name: str
    album_name: Optional[str] = None
    spotify_uri: Optional[str] = None
    spotify_track_id: Optional[str] = None
    duration_ms: Optional[int] = None
    played_duration_ms: Optional[int] = None
    completion_percentage: Optional[float] = None
    recommendation_mode: Optional[str] = None
    similarity_score: Optional[float] = None
    session_id: Optional[str] = None


@router.post("/log-listening")
async def log_listening(request: ListeningLogRequest):
    """듣는 기록 저장"""
    if not request.track_name or not request.artist_name:
        raise HTTPException(
            status_code=400, detail="Missing required fields: track_name, artist_name"
        )

    try:
        log_data = {
            "track_name": request.track_name,
            "artist_name": request.artist_name,
            "album_name": request.album_name,
            "spotify_uri": request.spotify_uri,
            "spotify_track_id": request.spotify_track_id,
            "session_id": request.session_id,
        }

        # Optional 필드는 값이 있을 때만 추가
        if request.duration_ms is not None:
            log_data["duration_ms"] = request.duration_ms
        if request.played_duration_ms is not None:
            log_data["played_duration_ms"] = request.played_duration_ms
        if request.completion_percentage is not None:
            log_data["completion_percentage"] = request.completion_percentage
        if request.recommendation_mode is not None:
            log_data["recommendation_mode"] = request.recommendation_mode
        if request.similarity_score is not None:
            log_data["similarity_score"] = request.similarity_score

        logger.info(f"Logging track: {request.track_name} by {request.artist_name}")

        response = supabase.table("listening_logs").insert(log_data).execute()

        if response.data is None:
            raise HTTPException(status_code=500, detail="Failed to log listening data")

        logger.info(f"Successfully logged track: {request.track_name}")
        return {"success": True, "data": response.data}

    except Exception as e:
        logger.error(f"Log listening error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
