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

    # 3. bridge fill: 거리 먼 seed 쌍 사이 중간 벡터로 보간
    keys = list(seed_embs.keys())
    if len(keys) >= 2:
        pairs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ka, kb = keys[i], keys[j]
                ea, eb = seed_embs[ka], seed_embs[kb]
                dist = float(np.linalg.norm(ea - eb))
                pairs.append((dist, ea, eb, ka, kb))

        max_dist = max(p[0] for p in pairs) or 1.0
        bridge_base = max(request.bridge_per_pair, 5)

        pairs.sort(reverse=True)
        MAX_BRIDGE_PAIRS = 15
        # (mid_vec, n_per_point, ea, eb, threshold, key_a, key_b) — ea/eb는 검증용
        bridge_queries = []
        for dist, ea, eb, key_a, key_b in pairs[:MAX_BRIDGE_PAIRS]:
            n_interp = max(1, min(3, round((dist / max_dist) * 5)))
            n_per_point = max(3, round(bridge_base * (dist / max_dist)))
            sim_ab = float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))
            threshold = max(sim_ab, 0.2)
            for k in range(1, n_interp + 1):
                t = k / (n_interp + 1)
                mid = (1 - t) * ea + t * eb
                mid = mid / (np.linalg.norm(mid) + 1e-8)
                bridge_queries.append((mid, n_per_point, ea, eb, threshold, key_a, key_b))

        # bridge는 seed만 제외하고 조회 (fill과 겹쳐도 is_bridge 우선)
        bridge_results = await asyncio.gather(*[
            fetch_near_raw(emb, limit, exclude_base) for emb, limit, *_ in bridge_queries
        ])
        bridge_keys = set()
        bridge_info = {}  # track_key → {key_a, key_b, sim_a, sim_b}
        for (_, limit, ea, eb, threshold, key_a, key_b), rows in zip(bridge_queries, bridge_results):
            ea_n = ea / (np.linalg.norm(ea) + 1e-8)
            eb_n = eb / (np.linalg.norm(eb) + 1e-8)
            candidates = []
            for r in rows:
                tk = r["track_key"]
                if tk in seed_key_set:
                    continue
                cemb = np.array(json_module.loads(r["embedding"]), dtype=np.float32)
                cemb_n = cemb / (np.linalg.norm(cemb) + 1e-8)
                sim_a = float(np.dot(cemb_n, ea_n))
                sim_b = float(np.dot(cemb_n, eb_n))
                if sim_a >= threshold and sim_b >= threshold:
                    candidates.append((min(sim_a, sim_b), sim_a, sim_b, r))
            candidates.sort(reverse=True)
            for _, sim_a, sim_b, r in candidates[:max(1, len(candidates) // 2)]:
                tk = r["track_key"]
                bridge_keys.add(tk)
                if tk not in seen:
                    seen.add(tk)
                    fill_rows_all.append(r)
                # 이미 다른 bridge pair에서 등록됐어도 info 덮어쓰기 (더 최근 pair)
                bridge_info[tk] = {
                    "bridge_seed_a_key": key_a,
                    "bridge_seed_b_key": key_b,
                    "bridge_sim_a": round(sim_a, 3),
                    "bridge_sim_b": round(sim_b, 3),
                }
    else:
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

        # fill: 어느 시드에서 뽑혔는지
        source_key = fill_source_seed.get(tk)
        source_meta = seed_meta_map.get(source_key) if source_key else None

        # bridge: 두 시드 정보 + 유사도
        bi = bridge_info.get(tk) if is_bridge else None
        if bi:
            ka, kb = bi["bridge_seed_a_key"], bi["bridge_seed_b_key"]
            ma, mb = seed_meta_map.get(ka), seed_meta_map.get(kb)
            bridge_seed_a = {"key": ka, "title": ma["title"] if ma else None, "artist": ma["artist"] if ma else None, "sim": bi["bridge_sim_a"]}
            bridge_seed_b = {"key": kb, "title": mb["title"] if mb else None, "artist": mb["artist"] if mb else None, "sim": bi["bridge_sim_b"]}
        else:
            bridge_seed_a = bridge_seed_b = None

        all_meta.append({
            "track_key": tk,
            "title": row["title"],
            "artist": row["artist"],
            "album": row["album"],
            "cover_image_url": row["cover_image_url"],
            "playlist_count": row["playlist_count"],
            "is_seed": is_seed,
            "is_favorite": tk in favorite_set,
            "is_bridge": is_bridge,
            "source_seed_key": source_key,
            "source_seed_title": source_meta["title"] if source_meta else None,
            "source_seed_artist": source_meta["artist"] if source_meta else None,
            "bridge_seed_a": bridge_seed_a,
            "bridge_seed_b": bridge_seed_b,
        })

    if len(all_embs) < 2:
        raise HTTPException(status_code=500, detail="임베딩 데이터가 부족합니다")

    bridge_count = len(bridge_keys)
    fill_only_count = len(fill_rows_all) - bridge_count
    logger.info(
        f"music-map: {len(seed_rows)} seeds, {fill_only_count} fills, "
        f"{bridge_count} bridges → UMAP total={len(all_embs)}"
    )

    # 5. UMAP 2D 변환 (seed 곡에 가중치 3배)
    try:
        import umap
    except ImportError:
        raise HTTPException(status_code=500, detail="umap-learn 패키지가 필요합니다")

    X_base = np.array(all_embs, dtype=np.float32)
    weights = np.array([3.0 if m["is_seed"] else 1.0 for m in all_meta], dtype=np.float32)
    n_neighbors = min(request.n_neighbors, len(X_base) - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=request.min_dist,
        metric="cosine",
        random_state=42,
        low_memory=True,
        n_epochs=50,
    )
    coords_2d = reducer.fit_transform(X_base, sample_weight=weights)

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
