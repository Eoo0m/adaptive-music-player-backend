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
    fill_per_seed: Optional[int] = 60  # seed 1개당 뽑을 fill 수


@router.post("/music-map")
async def music_map(request: MusicMapRequest):
    """
    각 seed별 주변 fill을 따로 뽑아 합침 → 골고루 채워진 UMAP 지도 반환.
    모든 트랙은 동일 크기로 반환 (크기 구분은 프론트에서 호버로만).
    """
    if len(request.track_keys) < 1:
        raise HTTPException(status_code=400, detail="트랙을 1개 이상 입력해주세요")
    if len(request.track_keys) > 30:
        raise HTTPException(status_code=400, detail="트랙은 최대 30개까지 입력 가능합니다")

    try:
        import umap
    except ImportError:
        raise HTTPException(status_code=500, detail="umap-learn 패키지가 필요합니다: pip install umap-learn")

    pool = await get_db_pool()

    # 1. seed 트랙 조회
    seed_rows = await pool.fetch(
        """
        SELECT
            track_key::text, title::text, artist::text, album::text,
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

    # 2. 각 seed별로 주변 fill 트랙을 따로 뽑기
    fill_per_seed = max(request.fill_per_seed, 30)
    all_fill_keys_seen = set(seed_keys_found)
    fill_rows_all = []

    for seed_row in seed_rows:
        seed_emb = np.array(json_module.loads(seed_row["embedding"]), dtype=np.float32)
        # 이미 뽑힌 트랙들 제외하면서 이 seed 주변 fill 뽑기
        exclude_keys = list(all_fill_keys_seen)
        rows = await pool.fetch(
            """
            SELECT
                track_key::text, title::text, artist::text, album::text,
                cover_image_url::text, playlist_count, embedding::text
            FROM track_embeddings
            WHERE track_key != ALL($1::text[])
              AND embedding IS NOT NULL
            ORDER BY embedding <=> $2::vector
            LIMIT $3
            """,
            exclude_keys,
            str(seed_emb.tolist()),
            fill_per_seed,
        )
        for r in rows:
            if r["track_key"] not in all_fill_keys_seen:
                fill_rows_all.append(r)
                all_fill_keys_seen.add(r["track_key"])

    # 3. 전체 트랙 목록 구성
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

    # 4. UMAP 2D 변환
    X = np.array(all_embs, dtype=np.float32)
    n_neighbors = min(request.n_neighbors, len(X) - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=request.min_dist,
        metric="euclidean",
        random_state=42,
        low_memory=False,
    )
    coords_2d = reducer.fit_transform(X)

    # 5. 결과 구성
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
