"""
Dynamic Streaming API

/next-track    → 현재 곡과 가장 가까운 유사곡 4개 반환
/dynamic-next  → 현재 곡과 거리 먼 상위 30% 중 랜덤 4개 반환 (다른 곡)
"""

import logging
import random

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

RETRY_COUNT = 4


class TrackRequest(BaseModel):
    track_key: str
    candidate_track_keys: Optional[list] = None


@router.post("/next-track")
async def next_track(request: TrackRequest):
    """유사곡 — pgvector ANN으로 현재 곡과 가까운 상위 10% 중 랜덤 4개 반환"""
    pool = await get_db_pool()

    # 현재 곡 임베딩 조회
    row = await pool.fetchrow(
        "SELECT embedding::text FROM track_vectors_simgcl WHERE track_key = $1",
        request.track_key,
    )
    if not row:
        raise HTTPException(status_code=404, detail="트랙을 찾을 수 없습니다")

    emb_str = row["embedding"]
    candidate_keys = request.candidate_track_keys

    if candidate_keys:
        n = len(candidate_keys)
        top_k = max(RETRY_COUNT * 3, int(n * 0.1))
        rows = await pool.fetch(
            f"""
            SELECT {TRACK_FIELDS}
            FROM track_vectors_simgcl v
            JOIN tracks t ON v.track_key = t.track_key
            WHERE t.track_key != $1
              AND t.track_key = ANY($3::text[])
            ORDER BY v.embedding <=> $2::vector
            LIMIT $4
            """,
            request.track_key, emb_str, candidate_keys, top_k,
        )
    else:
        rows = await pool.fetch(
            f"""
            SELECT {TRACK_FIELDS}
            FROM track_vectors_simgcl v
            JOIN tracks t ON v.track_key = t.track_key
            WHERE t.track_key != $1
            ORDER BY v.embedding <=> $2::vector
            LIMIT $3
            """,
            request.track_key, emb_str, max(RETRY_COUNT * 3, 20),
        )

    if not rows:
        raise HTTPException(status_code=404, detail="다음 트랙을 찾을 수 없습니다")

    pool_list = [dict(r) for r in rows]
    random.shuffle(pool_list)
    return {"tracks": pool_list[:RETRY_COUNT]}


@router.post("/dynamic-next")
async def dynamic_next(request: TrackRequest):
    """다른 곡 — pgvector ANN으로 먼 순 정렬 후 상위 30% 중 랜덤 4개 반환"""
    pool = await get_db_pool()

    row = await pool.fetchrow(
        "SELECT embedding::text FROM track_vectors_simgcl WHERE track_key = $1",
        request.track_key,
    )
    if not row:
        raise HTTPException(status_code=404, detail="트랙을 찾을 수 없습니다")

    emb_str = row["embedding"]
    candidate_keys = request.candidate_track_keys

    if candidate_keys:
        n = len(candidate_keys)
        top_k = max(RETRY_COUNT * 3, int(n * 0.3))
        rows = await pool.fetch(
            f"""
            SELECT {TRACK_FIELDS}
            FROM track_vectors_simgcl v
            JOIN tracks t ON v.track_key = t.track_key
            WHERE t.track_key != $1
              AND t.track_key = ANY($3::text[])
            ORDER BY v.embedding <=> $2::vector DESC
            LIMIT $4
            """,
            request.track_key, emb_str, candidate_keys, top_k,
        )
    else:
        rows = await pool.fetch(
            f"""
            SELECT {TRACK_FIELDS}
            FROM track_vectors_simgcl v
            JOIN tracks t ON v.track_key = t.track_key
            WHERE t.track_key != $1
            ORDER BY v.embedding <=> $2::vector DESC
            LIMIT $3
            """,
            request.track_key, emb_str, max(RETRY_COUNT * 3, 20),
        )

    if not rows:
        raise HTTPException(status_code=404, detail="다음 트랙을 찾을 수 없습니다")

    pool_list = [dict(r) for r in rows]
    random.shuffle(pool_list)
    return {"tracks": pool_list[:RETRY_COUNT]}
