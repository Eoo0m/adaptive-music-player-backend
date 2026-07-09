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
    track_keys: List[str]           # 사용자가 입력한 seed 트랙들
    favorite_keys: Optional[List[str]] = []  # 찜한 곡 키 목록
    n_neighbors: Optional[int] = 15
    min_dist: Optional[float] = 0.1
    fill_count: Optional[int] = 150  # seed 사이 채울 후보 수


@router.post("/music-map")
async def music_map(request: MusicMapRequest):
    """
    주어진 트랙들의 embedding으로 UMAP 2D 지도를 생성.
    seed 트랙 사이사이에 유사 트랙들을 채워 밀도 있는 지도 반환.
    찜한 곡(favorite_keys)은 별도 플래그로 표시.
    """
    if len(request.track_keys) < 2:
        raise HTTPException(status_code=400, detail="트랙을 2개 이상 입력해주세요")
    if len(request.track_keys) > 30:
        raise HTTPException(status_code=400, detail="트랙은 최대 30개까지 입력 가능합니다")

    try:
        import umap
    except ImportError:
        raise HTTPException(status_code=500, detail="umap-learn 패키지가 필요합니다: pip install umap-learn")

    pool = await get_db_pool()

    # 1. seed 트랙 embedding + 메타데이터 조회
    seed_rows = await pool.fetch(
        """
        SELECT
            track_key::text,
            title::text,
            artist::text,
            album::text,
            cover_image_url::text,
            playlist_count,
            embedding::text
        FROM track_embeddings
        WHERE track_key = ANY($1::text[])
          AND embedding IS NOT NULL
        """,
        request.track_keys,
    )

    if not seed_rows:
        raise HTTPException(status_code=404, detail="트랙을 찾을 수 없습니다")

    seed_keys_found = [r["track_key"] for r in seed_rows]

    # seed 평균 embedding으로 주변 fill 트랙 검색
    seed_embs = []
    for row in seed_rows:
        emb = np.array(json_module.loads(row["embedding"]), dtype=np.float32)
        seed_embs.append(emb)
    mean_emb = np.mean(seed_embs, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)

    # 2. seed 주변 fill 트랙 조회 (seed 제외)
    fill_count = max(request.fill_count, 80)
    fill_rows = await pool.fetch(
        """
        SELECT
            track_key::text,
            title::text,
            artist::text,
            album::text,
            cover_image_url::text,
            playlist_count,
            embedding::text
        FROM track_embeddings
        WHERE track_key != ALL($1::text[])
          AND embedding IS NOT NULL
        ORDER BY embedding <=> $2::vector
        LIMIT $3
        """,
        seed_keys_found,
        str(mean_emb.tolist()),
        fill_count,
    )

    # 3. 전체 트랙 목록 구성 (seed + fill)
    all_rows = list(seed_rows) + list(fill_rows)
    all_keys = []
    all_embs = []
    all_meta = []
    seed_set = set(seed_keys_found)
    favorite_set = set(request.favorite_keys or [])

    for row in all_rows:
        emb_str = row["embedding"]
        if not emb_str:
            continue
        emb = np.array(json_module.loads(emb_str), dtype=np.float32)
        all_keys.append(row["track_key"])
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
        "fill_count": len(fill_rows),
        "total": len(tracks),
    }
