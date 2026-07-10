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

    # 2. seed별 fill 병렬 조회 (속도 유지)
    fill_per_seed = max(request.fill_per_seed, 10)
    exclude_base = list(seed_key_set)

    seed_fill_results = await asyncio.gather(*[
        fetch_near_raw(seed_embs[key], fill_per_seed * 3, exclude_base)
        for key in seed_keys_found
    ])

    # 후보 풀 수집 (중복 포함, embedding도 파싱)
    candidate_pool = {}  # track_key → (row, emb)
    for rows in seed_fill_results:
        for r in rows:
            if r["track_key"] not in seed_key_set and r["track_key"] not in candidate_pool:
                emb = np.array(json_module.loads(r["embedding"]), dtype=np.float32)
                candidate_pool[r["track_key"]] = (r, emb)

    # 각 후보를 "가장 가까운 seed"에 배분 — 코사인 거리 기준
    seed_keys_list = list(seed_embs.keys())
    seed_embs_arr = np.array([seed_embs[k] for k in seed_keys_list], dtype=np.float32)  # (S, D)

    buckets = {k: [] for k in seed_keys_list}
    for tk, (row, emb) in candidate_pool.items():
        dists = 1.0 - (seed_embs_arr @ emb) / (
            np.linalg.norm(seed_embs_arr, axis=1) * np.linalg.norm(emb) + 1e-8
        )
        nearest = seed_keys_list[int(np.argmin(dists))]
        buckets[nearest].append((float(dists.min()), row))

    # 각 bucket에서 가까운 순으로 fill_per_seed개 선택
    seen = set(seed_key_set)
    fill_rows_all = []
    for key in seed_keys_list:
        buckets[key].sort(key=lambda x: x[0])
        for _, row in buckets[key][:fill_per_seed]:
            if row["track_key"] not in seen:
                seen.add(row["track_key"])
                fill_rows_all.append(row)

    # 3. bridge fill: 거리 먼 seed 쌍 사이 중간 벡터로 보간
    keys = list(seed_embs.keys())
    if len(keys) >= 2:
        pairs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ea, eb = seed_embs[keys[i]], seed_embs[keys[j]]
                dist = float(np.linalg.norm(ea - eb))
                pairs.append((dist, ea, eb))

        max_dist = max(p[0] for p in pairs) or 1.0
        bridge_base = max(request.bridge_per_pair, 5)

        pairs.sort(reverse=True)
        MAX_BRIDGE_PAIRS = 15
        bridge_queries = []
        for dist, ea, eb in pairs[:MAX_BRIDGE_PAIRS]:
            n_interp = max(1, min(3, round((dist / max_dist) * 5)))
            n_per_point = max(3, round(bridge_base * (dist / max_dist)))
            for k in range(1, n_interp + 1):
                t = k / (n_interp + 1)
                mid = (1 - t) * ea + t * eb
                mid = mid / (np.linalg.norm(mid) + 1e-8)
                bridge_queries.append((mid, n_per_point))

        # bridge는 병렬 조회 (현재 seen 기준 제외)
        exclude_now = list(seen)
        bridge_results = await asyncio.gather(*[
            fetch_near_raw(emb, limit, exclude_now) for emb, limit in bridge_queries
        ])
        for rows in bridge_results:
            for r in rows:
                if r["track_key"] not in seen:
                    seen.add(r["track_key"])
                    fill_rows_all.append(r)

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

    # 5. 2D 레이아웃 — UMAP 없이 직접 계산
    #
    # [seed 위치] MDS 방식: seed끼리 cosine 거리 행렬 → 2D 좌표
    # [fill 위치] 각 seed와의 cosine 유사도를 가중치로 weighted average
    #             → 가장 가까운 seed 쪽으로 강하게 당겨짐

    seed_meta_idx = [i for i, m in enumerate(all_meta) if m["is_seed"]]
    fill_meta_idx = [i for i, m in enumerate(all_meta) if not m["is_seed"]]

    X = np.array(all_embs, dtype=np.float32)
    # L2 정규화 (코사인 = 내적)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    X_norm = X / norms

    S_embs = X_norm[seed_meta_idx]  # (n_seeds, D)
    n_seeds = len(S_embs)

    # seed 간 거리 행렬 (cosine distance = 1 - cosine similarity)
    sim_ss = S_embs @ S_embs.T  # (n_seeds, n_seeds)
    dist_ss = np.clip(1.0 - sim_ss, 0, 2)

    # Classical MDS로 seed 2D 좌표 계산
    if n_seeds == 1:
        seed_coords = np.zeros((1, 2), dtype=np.float32)
    else:
        D2 = dist_ss ** 2
        n = n_seeds
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D2 @ J
        eigvals, eigvecs = np.linalg.eigh(B)
        # 상위 2개 고유값
        idx = np.argsort(eigvals)[::-1][:2]
        vals = np.maximum(eigvals[idx], 0)
        seed_coords = eigvecs[:, idx] * np.sqrt(vals)  # (n_seeds, 2)

    # fill 위치: 각 seed와의 유사도^k 를 가중치로 weighted average
    # k가 클수록 가장 가까운 seed 쪽으로 더 강하게 당겨짐
    SHARPNESS = 8.0
    coords_2d = np.zeros((len(all_meta), 2), dtype=np.float32)

    for ci, si in enumerate(seed_meta_idx):
        coords_2d[si] = seed_coords[ci]

    if fill_meta_idx:
        F_embs = X_norm[fill_meta_idx]          # (n_fills, D)
        sim_fs = F_embs @ S_embs.T              # (n_fills, n_seeds)  유사도
        # 음수 방지 후 sharpness 적용
        w = np.clip(sim_fs, 0, None) ** SHARPNESS
        w_sum = w.sum(axis=1, keepdims=True) + 1e-8
        w_norm = w / w_sum
        fill_coords = w_norm @ seed_coords      # (n_fills, 2)
        # 각 fill에 약간의 noise — 같은 위치에 겹치지 않도록
        rng = np.random.default_rng(42)
        spread = float(np.std(seed_coords) + 1e-4) * 0.15
        fill_coords += rng.normal(0, spread, fill_coords.shape)
        for ci, fi in enumerate(fill_meta_idx):
            coords_2d[fi] = fill_coords[ci]

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
