import asyncio
import logging
import json
import numpy as np
import torch
from datetime import datetime
from fastapi import APIRouter

from database import get_db_pool
from models import playlist_clip_model, device
from utils import get_openai_embedding_with_retry, mmr_rerank

logger = logging.getLogger(__name__)

router = APIRouter()

# 계절 키워드 캐시 (keyword → (timestamp, results))
_season_cache: dict = {}
_SEASON_CACHE_TTL = 86400  # 24시간


def _get_season_keyword() -> tuple[str, str]:
    """현재 월 기준 계절 키워드 반환 (keyword, title)"""
    month = datetime.now().month
    if month in (3, 4, 5):
        return "봄 감성 설레임", "봄 감성"
    elif month in (6, 7, 8):
        return "여름 청량 바다", "여름 감성"
    elif month in (9, 10, 11):
        return "가을 쓸쓸 감성", "가을 감성"
    else:
        return "겨울 따뜻 포근", "겨울 감성"


async def _fetch_popular(pool, limit: int = 7) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT track_key::text, title::text, artist::text,
               cover_image_url::text, apple_music_id::text, playlist_count
        FROM tracks
        WHERE cover_image_url IS NOT NULL AND cover_image_url != ''
        ORDER BY playlist_count DESC
        LIMIT $1
        """,
        limit,
    )
    return [
        {
            "track_key": r["track_key"],
            "title": r["title"],
            "artist": r["artist"],
            "cover_image_url": r["cover_image_url"],
            "apple_music_id": r["apple_music_id"],
        }
        for r in rows
    ]


async def _fetch_similar(pool, seed_track_key: str, exclude_keys: set, limit: int = 7) -> list[dict]:
    query_row = await pool.fetchrow(
        "SELECT embedding::text FROM track_vectors_simgcl WHERE track_key = $1 LIMIT 1",
        seed_track_key,
    )
    if not query_row:
        return []

    rows = await pool.fetch(
        """
        SELECT t.track_key::text, t.title::text, t.artist::text,
               t.cover_image_url::text, t.apple_music_id::text
        FROM track_vectors_simgcl v
        JOIN tracks t ON v.track_key = t.track_key
        WHERE v.track_key != $1
          AND t.cover_image_url IS NOT NULL AND t.cover_image_url != ''
        ORDER BY v.embedding <=> $2::vector
        LIMIT $3
        """,
        seed_track_key,
        query_row["embedding"],
        limit * 3,
    )
    result = []
    for r in rows:
        if r["track_key"] not in exclude_keys:
            result.append({
                "track_key": r["track_key"],
                "title": r["title"],
                "artist": r["artist"],
                "cover_image_url": r["cover_image_url"],
                "apple_music_id": r["apple_music_id"],
            })
        if len(result) >= limit:
            break
    return result


async def _fetch_season(keyword: str, limit: int = 6) -> list[dict]:
    global _season_cache
    now = datetime.now().timestamp()

    # 캐시 히트
    if keyword in _season_cache:
        ts, cached = _season_cache[keyword]
        if now - ts < _SEASON_CACHE_TTL:
            return cached[:limit]

    try:
        kw_emb = get_openai_embedding_with_retry(keyword)
        kw_tensor = torch.tensor(kw_emb, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            projected = playlist_clip_model.caption_proj(kw_tensor)
            projected_np = projected.cpu().numpy()[0]

        pool = await get_db_pool()
        rows = await pool.fetch(
            """
            SELECT te.track_key::text, te.title::text, te.artist::text,
                   te.cover_image_url::text, te.apple_music_id::text,
                   te.projected_embedding::text
            FROM track_embeddings te
            WHERE te.cover_image_url IS NOT NULL AND te.cover_image_url != ''
            ORDER BY te.projected_embedding <=> $1::vector
            LIMIT $2
            """,
            str(projected_np.tolist()),
            50,
        )

        if not rows:
            return []

        candidate_ids, candidate_embs, candidate_data = [], [], {}
        for row in rows:
            emb_str = row["projected_embedding"]
            if emb_str:
                emb = np.array(json.loads(emb_str), dtype=np.float32)
                candidate_ids.append(row["track_key"])
                candidate_embs.append(emb)
                candidate_data[row["track_key"]] = {
                    "track_key": row["track_key"],
                    "title": row["title"],
                    "artist": row["artist"],
                    "cover_image_url": row["cover_image_url"],
                    "apple_music_id": row["apple_music_id"],
                }

        selected_ids, _ = mmr_rerank(
            projected_np, np.array(candidate_embs), candidate_ids,
            top_k=20, lambda_param=0.7,
        )
        results = [candidate_data[k] for k in selected_ids if k in candidate_data]

        _season_cache[keyword] = (now, results)
        return results[:limit]

    except Exception as e:
        logger.warning(f"Season fetch failed: {e}")
        return []


@router.get("/home-feed-v2")
async def home_feed_v2():
    """홈 피드 v2: 인기곡 7 + 유사곡 7 + 계절 감성 6 = 20곡"""
    pool = await get_db_pool()

    season_keyword, season_title = _get_season_keyword()

    # 인기곡 먼저 (유사곡 seed로 필요)
    popular = await _fetch_popular(pool, limit=7)

    # 유사곡 + 계절 병렬
    popular_keys = {t["track_key"] for t in popular}
    seed_key = popular[0]["track_key"] if popular else None

    similar_task = _fetch_similar(pool, seed_key, popular_keys, limit=7) if seed_key else asyncio.sleep(0)
    season_task = _fetch_season(season_keyword, limit=6)

    similar, season = await asyncio.gather(similar_task, season_task)
    if not isinstance(similar, list):
        similar = []

    # 중복 제거
    seen = set(popular_keys)
    similar_clean = []
    for t in similar:
        if t["track_key"] not in seen:
            similar_clean.append(t)
            seen.add(t["track_key"])

    season_clean = []
    for t in season:
        if t["track_key"] not in seen:
            season_clean.append(t)
            seen.add(t["track_key"])

    tracks = popular + similar_clean + season_clean

    logger.info(f"/home-feed-v2: popular={len(popular)} similar={len(similar_clean)} season={len(season_clean)} total={len(tracks)}")
    return {"tracks": tracks, "season_title": season_title}
