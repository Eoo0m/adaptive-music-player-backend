"""
Dynamic Streaming API

/next-track    → 현재 곡과 가장 가까운 유사곡 4개 반환
/dynamic-next  → 현재 곡과 거리 먼 상위 30% 중 랜덤 4개 반환 (다른 곡)
"""

import logging
import random

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

RETRY_COUNT = 4


class TrackRequest(BaseModel):
    track_key: str
    candidate_track_keys: Optional[list] = None


async def _fetch_candidates(pool, track_key: str, candidate_track_keys):
    """현재 곡 임베딩 + 후보 곡 임베딩 조회"""
    row = await pool.fetchrow(
        """
        SELECT v.embedding::text
        FROM track_vectors_simgcl v
        WHERE v.track_key = $1
        """,
        track_key,
    )
    if not row:
        raise HTTPException(status_code=404, detail="트랙을 찾을 수 없습니다")

    emb = np.array(eval(row["embedding"]), dtype=np.float32)
    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

    if candidate_track_keys:
        rows = await pool.fetch(
            f"""
            SELECT {TRACK_FIELDS}, v.embedding::text AS emb
            FROM track_vectors_simgcl v
            JOIN tracks t ON v.track_key = t.track_key
            WHERE t.track_key != $1
            AND t.track_key = ANY($2::text[])
            """,
            track_key, candidate_track_keys,
        )
    else:
        rows = await pool.fetch(
            f"""
            SELECT {TRACK_FIELDS}, v.embedding::text AS emb
            FROM track_vectors_simgcl v
            JOIN tracks t ON v.track_key = t.track_key
            WHERE t.track_key != $1
            """,
            track_key,
        )

    if not rows:
        raise HTTPException(status_code=404, detail="다음 트랙을 찾을 수 없습니다")

    scored = []
    for r in rows:
        e = np.array(eval(r["emb"]), dtype=np.float32)
        e_norm = e / (np.linalg.norm(e) + 1e-8)
        sim = float(np.dot(emb_norm, e_norm))
        scored.append((sim, r))

    return scored


@router.post("/next-track")
async def next_track(request: TrackRequest):
    """유사곡 — 현재 곡과 가까운 상위 20% 중 랜덤 4개 반환"""
    pool = await get_db_pool()
    scored = await _fetch_candidates(pool, request.track_key, request.candidate_track_keys)
    scored.sort(key=lambda x: x[0], reverse=True)  # 가까운 순
    top10_count = max(RETRY_COUNT, int(len(scored) * 0.1))
    pool10 = [r for _, r in scored[:top10_count]]
    random.shuffle(pool10)
    tracks = [{k: v for k, v in dict(r).items() if k != "emb"} for r in pool10[:RETRY_COUNT]]
    return {"tracks": tracks}


@router.post("/dynamic-next")
async def dynamic_next(request: TrackRequest):
    """다른 곡 — 현재 곡과 거리 먼 상위 30% 중 랜덤 4개 반환"""
    pool = await get_db_pool()
    scored = await _fetch_candidates(pool, request.track_key, request.candidate_track_keys)
    scored.sort(key=lambda x: x[0])  # 먼 순
    top30_count = max(RETRY_COUNT, int(len(scored) * 0.3))
    pool30 = [r for _, r in scored[:top30_count]]
    random.shuffle(pool30)
    tracks = [{k: v for k, v in dict(r).items() if k != "emb"} for r in pool30[:RETRY_COUNT]]
    return {"tracks": tracks}
