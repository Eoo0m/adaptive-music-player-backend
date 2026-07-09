import json as json_module
import logging
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from database import get_db_pool

logger = logging.getLogger(__name__)
router = APIRouter()


class MusicMapRequest(BaseModel):
    track_keys: List[str]
    favorite_keys: Optional[List[str]] = []
    n_neighbors: Optional[int] = 15
    min_dist: Optional[float] = 0.1
    fill_per_seed: Optional[int] = 30   # seed 주변 fill 수
    bridge_per_pair: Optional[int] = 15  # seed 쌍 사이 보간 fill 수


@router.post("/music-map")
async def music_map(request: MusicMapRequest):
    """
    1. 각 seed 주변에서 fill 뽑기
    2. seed 쌍 간 거리에 비례해 중간 보간 벡터로 bridge fill 뽑기
       → 힙합↔발라드처럼 멀수록 사이에 중간 장르가 채워짐
    """
    if len(request.track_keys) < 1:
        raise HTTPException(status_code=400, detail="트랙을 1개 이상 입력해주세요")
    if len(request.track_keys) > 30:
        raise HTTPException(status_code=400, detail="트랙은 최대 30개까지 입력 가능합니다")

    try:
        import umap
    except ImportError:
        raise HTTPException(status_code=500, detail="umap-learn 패키지가 필요합니다")

    pool = await get_db_pool()

    # 1. seed 트랙 조회
    seed_rows = await pool.fetch(
        """
        SELECT track_key::text, title::text, artist::text, album::text,
               cover_image_url::text, playlist_count, embedding::text
        FROM track_embeddings
        WHERE track_key = ANY($1::text[])
          AND embedding IS NOT NULL
        """,
        request.track_keys,
    )

    if not seed_rows:
        raise HTTPException(status_code=404, detail="트랙을 찾을 수 없습니다")

    seed_keys_found = [r["track_key"] for r in seed_rows]
    seed_set = set(seed_keys_found)
    favorite_set = set(request.favorite_keys or [])

    # seed embedding 파싱
    seed_embs = {}
    for row in seed_rows:
        seed_embs[row["track_key"]] = np.array(
            json_module.loads(row["embedding"]), dtype=np.float32
        )

    all_fill_keys_seen = set(seed_keys_found)
    fill_rows_all = []

    async def fetch_near(emb: np.ndarray, limit: int):
        """주어진 벡터 근처 트랙을 DB에서 뽑기 (이미 본 것 제외)"""
        exclude = list(all_fill_keys_seen)
        rows = await pool.fetch(
            """
            SELECT track_key::text, title::text, artist::text, album::text,
                   cover_image_url::text, playlist_count, embedding::text
            FROM track_embeddings
            WHERE track_key != ALL($1::text[])
              AND embedding IS NOT NULL
            ORDER BY embedding <=> $2::vector
            LIMIT $3
            """,
            exclude,
            str(emb.tolist()),
            limit,
        )
        new = []
        for r in rows:
            if r["track_key"] not in all_fill_keys_seen:
                new.append(r)
                all_fill_keys_seen.add(r["track_key"])
        return new

    # 2. 각 seed 주변 fill
    fill_per_seed = max(request.fill_per_seed, 10)
    for key in seed_keys_found:
        rows = await fetch_near(seed_embs[key], fill_per_seed)
        fill_rows_all.extend(rows)

    # 3. seed 쌍 사이 bridge fill (거리 비례)
    keys = list(seed_embs.keys())
    if len(keys) >= 2:
        # 쌍별 거리 계산
        pairs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ea, eb = seed_embs[keys[i]], seed_embs[keys[j]]
                dist = float(np.linalg.norm(ea - eb))
                pairs.append((dist, keys[i], keys[j], ea, eb))

        # 전체 거리 범위 정규화
        max_dist = max(p[0] for p in pairs) or 1.0

        bridge_base = max(request.bridge_per_pair, 5)
        for dist, ka, kb, ea, eb in pairs:
            # 거리에 비례해 보간 횟수 결정 (가까우면 1개, 멀면 최대 5개 보간점)
            n_interp = max(1, round((dist / max_dist) * 5))
            n_per_point = max(3, round(bridge_base * (dist / max_dist)))

            for k in range(1, n_interp + 1):
                t = k / (n_interp + 1)
                mid = (1 - t) * ea + t * eb
                mid = mid / (np.linalg.norm(mid) + 1e-8)
                rows = await fetch_near(mid, n_per_point)
                fill_rows_all.extend(rows)

    # 4. 전체 트랙 목록 구성
    all_rows = list(seed_rows) + fill_rows_all
    all_embs = []
    all_meta = []

    for row in all_rows:
        emb_str = row["embedding"]
        if not emb_str:
            continue
        emb = np.array(json_module.loads(emb_str), dtype=np.float32)
        all_embs.append(emb)
        all_meta.append({
            "track_key": row["track_key"],
            "title": row["title"],
            "artist": row["artist"],
            "album": row["album"],
            "cover_image_url": row["cover_image_url"],
            "playlist_count": row["playlist_count"],
            "is_seed": row["track_key"] in seed_set,
            "is_favorite": row["track_key"] in favorite_set,
        })

    if len(all_embs) < 2:
        raise HTTPException(status_code=500, detail="임베딩 데이터가 부족합니다")

    logger.info(f"music-map: {len(seed_rows)} seeds, {len(fill_rows_all)} fills, total={len(all_embs)}")

    # 5. UMAP 2D 변환
    X = np.array(all_embs, dtype=np.float32)
    n_neighbors = min(request.n_neighbors, len(X) - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=request.min_dist,
        metric="euclidean",
        random_state=42,
        low_memory=True,
        n_epochs=100,
    )
    coords_2d = reducer.fit_transform(X)

    # 6. 결과 구성
    tracks = []
    for i, meta in enumerate(all_meta):
        tracks.append({
            **meta,
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
        })

    return {
        "tracks": tracks,
        "seed_count": len(seed_rows),
        "fill_count": len(fill_rows_all),
        "total": len(tracks),
    }
