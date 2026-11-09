"""
Dynamic Streaming API

/dynamic-next:
  - listened_half=True  → 가장 가까운 트랙 상위 4개 반환
  - listened_half=False → 전체 후보 중 cosine 거리 먼 상위 30% → 랜덤 4개 반환
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


class DynamicNextRequest(BaseModel):
    track_key: str
    listened_half: bool = False
    candidate_track_keys: Optional[list] = None


@router.post("/dynamic-next")
async def dynamic_next(request: DynamicNextRequest):
    pool = await get_db_pool()

    # 현재 트랙 임베딩 조회
    row = await pool.fetchrow(
        """
        SELECT v.embedding::text
        FROM track_embeddings v
        WHERE v.track_key = $1
        """,
        request.track_key,
    )
    if not row:
        raise HTTPException(status_code=404, detail="트랙을 찾을 수 없습니다")

    emb = np.array(eval(row["embedding"]), dtype=np.float32)
    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
    emb_str = str(emb.tolist())

    if request.listened_half:
        # 가장 가까운 상위 RETRY_COUNT개
        if request.candidate_track_keys:
            rows = await pool.fetch(
                f"""
                SELECT {TRACK_FIELDS}
                FROM track_embeddings v
                JOIN tracks t ON v.track_key = t.track_key
                WHERE t.track_key != $1
                AND t.track_key = ANY($3::text[])
                ORDER BY v.embedding <=> $2::vector
                LIMIT {RETRY_COUNT}
                """,
                request.track_key, emb_str, request.candidate_track_keys,
            )
        else:
            rows = await pool.fetch(
                f"""
                SELECT {TRACK_FIELDS}
                FROM track_embeddings v
                JOIN tracks t ON v.track_key = t.track_key
                WHERE t.track_key != $1
                ORDER BY v.embedding <=> $2::vector
                LIMIT {RETRY_COUNT}
                """,
                request.track_key, emb_str,
            )
        tracks = [dict(r) for r in rows]
    else:
        # 전체 후보 조회 후 상위 30% 랜덤 (Python에서 cosine 계산)
        if request.candidate_track_keys:
            rows = await pool.fetch(
                f"""
                SELECT {TRACK_FIELDS}, v.embedding::text AS emb
                FROM track_embeddings v
                JOIN tracks t ON v.track_key = t.track_key
                WHERE t.track_key != $1
                AND t.track_key = ANY($2::text[])
                """,
                request.track_key, request.candidate_track_keys,
            )
        else:
            rows = await pool.fetch(
                f"""
                SELECT {TRACK_FIELDS}, v.embedding::text AS emb
                FROM track_embeddings v
                JOIN tracks t ON v.track_key = t.track_key
                WHERE t.track_key != $1
                """,
                request.track_key,
            )
        if not rows:
            raise HTTPException(status_code=404, detail="다음 트랙을 찾을 수 없습니다")

        # cosine 거리 계산 후 먼 상위 30% 추출
        scored = []
        for r in rows:
            e = np.array(eval(r["emb"]), dtype=np.float32)
            e_norm = e / (np.linalg.norm(e) + 1e-8)
            sim = float(np.dot(emb_norm, e_norm))
            scored.append((sim, r))

        scored.sort(key=lambda x: x[0])  # 먼 순(sim 낮은 순)
        top30_count = max(RETRY_COUNT, int(len(scored) * 0.3))
        pool30 = [r for _, r in scored[:top30_count]]
        random.shuffle(pool30)
        tracks = [{k: v for k, v in dict(r).items() if k != "emb"} for r in pool30[:RETRY_COUNT]]

    if not tracks:
        raise HTTPException(status_code=404, detail="다음 트랙을 찾을 수 없습니다")

    return {
        "tracks": tracks,
        "mode": "adjacent" if request.listened_half else "distant_seed"
    }
