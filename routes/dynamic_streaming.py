"""
Dynamic Streaming API

/dynamic-next:
  - listened_half=True  → 지도에서 인접 트랙으로 이동 (cosine 유사도 최근접)
  - listened_half=False → 거리가 먼 시드 트랙으로 이동 (cosine 거리 가장 큰 트랙)
"""

import json as json_module
import logging

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from database import get_db_pool

logger = logging.getLogger(__name__)
router = APIRouter()

TRACK_FIELDS = """
    t.track_key::text, t.title::text, t.artist::text, t.album::text,
    t.cover_image_url::text, t.apple_music_id::text, t.playlist_count
"""


class DynamicNextRequest(BaseModel):
    track_key: str
    listened_half: bool = False  # True: 50% 이상 들음 → adjacent, False: → distant seed


@router.post("/dynamic-next")
async def dynamic_next(request: DynamicNextRequest):
    pool = await get_db_pool()

    # 현재 트랙 임베딩 조회
    row = await pool.fetchrow(
        """
        SELECT v.embedding::text, t.track_key::text
        FROM track_vectors_simgcl v
        JOIN tracks t ON v.track_key = t.track_key
        WHERE t.track_key = $1
        """,
        request.track_key,
    )
    if not row:
        raise HTTPException(status_code=404, detail="트랙을 찾을 수 없습니다")

    emb = np.array(json_module.loads(row["embedding"]), dtype=np.float32)
    emb_str = str(emb.tolist())

    if request.listened_half:
        # 50% 이상 들음 → cosine 가장 가까운 인접 트랙 (자기 자신 제외)
        next_row = await pool.fetchrow(
            f"""
            SELECT {TRACK_FIELDS}
            FROM track_vectors_simgcl v
            JOIN tracks t ON v.track_key = t.track_key
            WHERE t.track_key != $1
            ORDER BY v.embedding <=> $2::vector
            LIMIT 1
            """,
            request.track_key,
            emb_str,
        )
    else:
        # 50% 미만 → cosine 가장 먼 시드 트랙 (playlist_count 높은 상위 200개 중)
        next_row = await pool.fetchrow(
            f"""
            SELECT {TRACK_FIELDS}
            FROM (
                SELECT v.embedding, t.track_key, t.title, t.artist, t.album,
                       t.cover_image_url, t.apple_music_id, t.playlist_count
                FROM track_vectors_simgcl v
                JOIN tracks t ON v.track_key = t.track_key
                WHERE t.track_key != $1
                ORDER BY t.playlist_count DESC
                LIMIT 200
            ) sub
            ORDER BY sub.embedding <=> $2::vector DESC
            LIMIT 1
            """,
            request.track_key,
            emb_str,
        )

    if not next_row:
        raise HTTPException(status_code=404, detail="다음 트랙을 찾을 수 없습니다")

    return {"track": dict(next_row), "mode": "adjacent" if request.listened_half else "distant_seed"}
