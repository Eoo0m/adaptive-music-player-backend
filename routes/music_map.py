import asyncio
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
    n_neighbors: Optional[int] = 8
    min_dist: Optional[float] = 0.03
    fill_per_seed: Optional[int] = 15   # seed 주변 fill 수
    bridge_per_pair: Optional[int] = 15  # seed 쌍 사이 보간 fill 수


@router.get("/top-tracks")
async def top_tracks(limit: int = 10):
    """저장수 높은 트랙 반환 (찜이 없을 때 취향 지도 seed용)"""
    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT track_key::text, title::text, artist::text, album::text,
               cover_image_url::text, playlist_count
        FROM track_embeddings
        WHERE embedding IS NOT NULL
        ORDER BY playlist_count DESC
        LIMIT $1
        """,
        limit,
    )
    return {"tracks": [dict(r) for r in rows]}


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

    seed_key_set = set(seed_keys_found)

    async def fetch_near_raw(emb: np.ndarray, limit: int, exclude: list):
        """주어진 벡터 근처 트랙을 DB에서 뽑기"""
        return await pool.fetch(
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

    # 2. seed별 fill 병렬 조회
    fill_per_seed = max(request.fill_per_seed, 10)
    exclude_base = list(seed_key_set)

    seed_fill_results = await asyncio.gather(*[
        fetch_near_raw(seed_embs[key], fill_per_seed, exclude_base)
        for key in seed_keys_found
    ])

    seen = set(seed_key_set)
    fill_rows_all = []
    fill_source_seed = {}  # track_key → seed_key (어느 시드에서 뽑혔는지)
    for seed_key, rows in zip(seed_keys_found, seed_fill_results):
        for r in rows:
            tk = r["track_key"]
            if tk not in seen:
                seen.add(tk)
                fill_rows_all.append(r)
                fill_source_seed[tk] = seed_key

    # 3. bridge 없음
    bridge_keys = set()

    # 4. 전체 트랙 목록 구성
    seed_meta_map = {r["track_key"]: {"title": r["title"], "artist": r["artist"]} for r in seed_rows}

    all_rows = list(seed_rows) + fill_rows_all
    all_embs = []
    all_meta = []

    for row in all_rows:
        emb_str = row["embedding"]
        if not emb_str:
            continue
        emb = np.array(json_module.loads(emb_str), dtype=np.float32)
        all_embs.append(emb)
        tk = row["track_key"]
        is_seed = tk in seed_set
        is_bridge = tk in bridge_keys

        source_key = fill_source_seed.get(tk)
        source_meta = seed_meta_map.get(source_key) if source_key else None

        all_meta.append({
            "track_key": tk,
            "title": row["title"],
            "artist": row["artist"],
            "album": row["album"],
            "cover_image_url": row["cover_image_url"],
            "playlist_count": row["playlist_count"],
            "is_seed": is_seed,
            "is_favorite": tk in favorite_set,
            "is_bridge": False,
            "source_seed_key": source_key,
            "source_seed_title": source_meta["title"] if source_meta else None,
            "source_seed_artist": source_meta["artist"] if source_meta else None,
        })

    if len(all_embs) < 2:
        raise HTTPException(status_code=500, detail="임베딩 데이터가 부족합니다")

    # track_key → embedding 맵 (규칙 기반 배치용)
    all_embs_map = {all_meta[i]["track_key"]: all_embs[i] for i in range(len(all_meta))}

    bridge_count = len(bridge_keys)
    fill_only_count = len(fill_rows_all) - bridge_count
    logger.info(
        f"music-map: {len(seed_rows)} seeds, {fill_only_count} fills, "
        f"{bridge_count} bridges → total={len(all_embs)}"
    )

    # 5. 배치: 시드는 UMAP, fill/bridge는 규칙 기반
    try:
        import umap as umap_module
    except ImportError:
        raise HTTPException(status_code=500, detail="umap-learn 패키지가 필요합니다")

    seed_list = [m for m in all_meta if m["is_seed"]]
    seed_keys_ordered = [m["track_key"] for m in seed_list]
    seed_embs_arr = np.array([seed_embs[k] for k in seed_keys_ordered], dtype=np.float32)

    # 5-1. 시드 위치: UMAP (시드 수가 많을 때 유사도 구조 반영)
    n_seeds = len(seed_keys_ordered)
    if n_seeds == 1:
        seed_pos = {seed_keys_ordered[0]: np.array([0.0, 0.0])}
    elif n_seeds <= 3:
        # 시드가 적으면 MDS (UMAP은 최소 n_neighbors+1개 필요)
        from sklearn.manifold import MDS
        seed_n = seed_embs_arr / (np.linalg.norm(seed_embs_arr, axis=1, keepdims=True) + 1e-8)
        cos_dist = np.clip(1.0 - seed_n @ seed_n.T, 0, 2).astype(np.float64)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
        mds_coords = mds.fit_transform(cos_dist)
        rng = mds_coords.max(axis=0) - mds_coords.min(axis=0) + 1e-8
        mds_coords = (mds_coords - mds_coords.mean(axis=0)) / rng.max() * 10.0
        seed_pos = {k: mds_coords[i] for i, k in enumerate(seed_keys_ordered)}
    else:
        n_neighbors = min(10, n_seeds - 1)
        reducer = umap_module.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
            metric="cosine", random_state=42, low_memory=True, n_epochs=200,
        )
        umap_coords = reducer.fit_transform(seed_embs_arr)  # (n_seeds, 2)
        # x, y 각각 독립적으로 [-5, 5] 정규화
        for dim in range(2):
            lo, hi = umap_coords[:, dim].min(), umap_coords[:, dim].max()
            rng_d = hi - lo + 1e-8
            umap_coords[:, dim] = (umap_coords[:, dim] - lo) / rng_d * 10.0 - 5.0
        seed_pos = {k: umap_coords[i] for i, k in enumerate(seed_keys_ordered)}

    # 시드 임베딩 정규화 (fill 유사도 계산용)
    seed_embs_n = {k: seed_embs[k] / (np.linalg.norm(seed_embs[k]) + 1e-8) for k in seed_keys_ordered}

    # 5-2. fill 위치: 소속 시드 중심 + 유사도 기반 반지름 + 각도 분산
    # 같은 시드 소속 fill끼리 각도를 균등 분산
    from collections import defaultdict
    seed_fill_list = defaultdict(list)  # seed_key → [track_key, ...]
    for m in all_meta:
        if not m["is_seed"] and not m["is_bridge"] and m["source_seed_key"]:
            seed_fill_list[m["source_seed_key"]].append(m["track_key"])

    fill_pos = {}

    # 시드 간 최소 거리의 절반을 fill 반지름 상한으로 설정 (다른 시드 영역 침범 방지)
    seed_positions = list(seed_pos.values())
    if len(seed_positions) >= 2:
        min_seed_dist = min(
            np.linalg.norm(seed_positions[i] - seed_positions[j])
            for i in range(len(seed_positions))
            for j in range(i + 1, len(seed_positions))
        )
        FILL_RADIUS_MAX = min_seed_dist * 0.45
    else:
        FILL_RADIUS_MAX = 2.0

    for sk, fill_keys in seed_fill_list.items():
        center = seed_pos[sk]
        sn = seed_embs_n[sk]
        rng_gen = np.random.default_rng(abs(hash(sk)) % 2**32)
        for tk in fill_keys:
            emb = all_embs_map[tk]
            emb_n = emb / (np.linalg.norm(emb) + 1e-8)
            sim = float(np.dot(emb_n, sn))
            # 유사도로 0~FILL_RADIUS_MAX 사이 반지름 결정
            radius = FILL_RADIUS_MAX * max(0.0, 1.0 - sim)
            radius = min(radius, FILL_RADIUS_MAX)
            angle = rng_gen.uniform(0, 2 * np.pi)
            fill_pos[tk] = center + radius * np.array([np.cos(angle), np.sin(angle)])

    # 6. 결과 구성
    tracks = []
    for meta in all_meta:
        tk = meta["track_key"]
        if meta["is_seed"]:
            xy = seed_pos[tk]
        else:
            xy = fill_pos.get(tk, seed_pos.get(meta.get("source_seed_key"), np.zeros(2)))
        tracks.append({
            **meta,
            "x": float(xy[0]),
            "y": float(xy[1]),
        })

    for t in tracks:
        if t["is_seed"]:
            logger.info(f"  seed [{t['title'][:20]}] x={t['x']:.2f} y={t['y']:.2f}")

    return {
        "tracks": tracks,
        "seed_count": len(seed_rows),
        "fill_count": len(fill_rows_all),
        "total": len(tracks),
    }
