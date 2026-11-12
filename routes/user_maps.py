"""
/user-maps: 사용자 취향지도 저장/조회

- GET  /user-maps        → 내 맵 목록 (id, name, updated_at, track 개수)
- GET  /user-maps/latest → 가장 최근 맵 전체 데이터 (앱 시작 시 빠른 로드용)
- POST /user-maps        → 맵 저장/업데이트 (name 기준 upsert)
- DELETE /user-maps/{id} → 맵 삭제
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import json

from database import get_db_pool
from routes.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


class MapTrackData(BaseModel):
    track_key: str
    track_name: str
    artist: str
    cover_image_url: Optional[str] = None
    apple_music_id: Optional[str] = None
    x: float
    y: float
    is_favorite: Optional[bool] = False


class SaveMapRequest(BaseModel):
    name: str = "내 취향지도"
    map_data: List[MapTrackData]


@router.get("/user-maps/latest")
async def get_latest_map(user: dict = Depends(get_current_user)):
    """가장 최근 저장된 맵 전체 반환 — 앱 시작 시 캐시 복원용"""
    pool = await get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT id, name, map_data, updated_at
        FROM user_maps
        WHERE user_id = $1
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        user["user_id"],
    )
    if not row:
        return {"map": None}
    return {
        "map": {
            "id": row["id"],
            "name": row["name"],
            "map_data": json.loads(row["map_data"]),
            "updated_at": row["updated_at"].isoformat(),
        }
    }


@router.get("/user-maps")
async def list_maps(user: dict = Depends(get_current_user)):
    """내 맵 목록 반환"""
    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT id, name, jsonb_array_length(map_data) AS track_count, updated_at
        FROM user_maps
        WHERE user_id = $1
        ORDER BY updated_at DESC
        """,
        user["user_id"],
    )
    return {
        "maps": [
            {
                "id": r["id"],
                "name": r["name"],
                "track_count": r["track_count"],
                "updated_at": r["updated_at"].isoformat(),
            }
            for r in rows
        ]
    }


@router.post("/user-maps")
async def save_map(request: SaveMapRequest, user: dict = Depends(get_current_user)):
    """맵 저장 — 같은 name이면 업데이트, 없으면 새로 생성"""
    pool = await get_db_pool()
    map_json = json.dumps([t.dict() for t in request.map_data])

    row = await pool.fetchrow(
        """
        INSERT INTO user_maps (user_id, name, map_data)
        VALUES ($1, $2, $3::jsonb)
        ON CONFLICT DO NOTHING
        RETURNING id
        """,
        user["user_id"], request.name, map_json,
    )

    if not row:
        # 같은 name 존재 → 업데이트
        row = await pool.fetchrow(
            """
            UPDATE user_maps SET map_data = $3::jsonb
            WHERE user_id = $1 AND name = $2
            RETURNING id
            """,
            user["user_id"], request.name, map_json,
        )

    return {"id": row["id"], "name": request.name, "track_count": len(request.map_data)}


@router.delete("/user-maps/{map_id}")
async def delete_map(map_id: int, user: dict = Depends(get_current_user)):
    pool = await get_db_pool()
    result = await pool.execute(
        "DELETE FROM user_maps WHERE id = $1 AND user_id = $2",
        map_id, user["user_id"],
    )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Map not found")
    return {"deleted": map_id}
