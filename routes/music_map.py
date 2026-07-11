import asyncio
import json as json_module
import logging
import math
import time
from collections import defaultdict
from itertools import combinations

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
    fill_per_seed: Optional[int] = 15
    bridge_per_pair: Optional[int] = 8


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
    1. seed → UMAP/MDS 2D 임베딩
    2. K-Means로 시드를 클러스터링
    3. 클러스터별 직사각형 블록에 seed+fill 배치
    4. 블록들을 UMAP 상대위치 기준으로 격자에 붙이기
    5. bridge: 먼 클러스터 쌍 경계 셀에 배치
    6. grid_col, grid_row 반환
    """
    if len(request.track_keys) < 1:
        raise HTTPException(status_code=400, detail="트랙을 1개 이상 입력해주세요")
    if len(request.track_keys) > 30:
        raise HTTPException(status_code=400, detail="트랙은 최대 30개까지 입력 가능합니다")

    t0 = time.perf_counter()
    def elapsed(label: str):
        logger.info(f"  [{label}] {(time.perf_counter() - t0) * 1000:.0f}ms")

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
    elapsed("1. seed 조회")

    seed_keys_found = [r["track_key"] for r in seed_rows]
    seed_set = set(seed_keys_found)
    favorite_set = set(request.favorite_keys or [])

    seed_embs = {}
    for row in seed_rows:
        seed_embs[row["track_key"]] = np.array(
            json_module.loads(row["embedding"]), dtype=np.float32
        )

    seed_key_set = set(seed_keys_found)

    async def fetch_near_raw(emb: np.ndarray, limit: int, exclude: list):
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
    fill_rows_map = defaultdict(list)  # seed_key → [row, ...]
    fill_source_seed = {}

    for seed_key, rows in zip(seed_keys_found, seed_fill_results):
        for r in rows:
            tk = r["track_key"]
            if tk not in seen:
                seen.add(tk)
                fill_rows_map[seed_key].append(r)
                fill_source_seed[tk] = seed_key
    elapsed("2. fill 조회")

    seed_meta_map = {r["track_key"]: {"title": r["title"], "artist": r["artist"]} for r in seed_rows}

    # 3. UMAP/MDS로 seed 2D 좌표 계산
    try:
        import umap as umap_module
    except ImportError:
        raise HTTPException(status_code=500, detail="umap-learn 패키지가 필요합니다")

    seed_keys_ordered = list(seed_keys_found)
    seed_embs_arr = np.array([seed_embs[k] for k in seed_keys_ordered], dtype=np.float32)
    n_seeds = len(seed_keys_ordered)

    if n_seeds == 1:
        seed_umap = {seed_keys_ordered[0]: np.array([0.0, 0.0])}
    elif n_seeds <= 3:
        from sklearn.manifold import MDS
        seed_n = seed_embs_arr / (np.linalg.norm(seed_embs_arr, axis=1, keepdims=True) + 1e-8)
        cos_dist = np.clip(1.0 - seed_n @ seed_n.T, 0, 2).astype(np.float64)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
        coords = mds.fit_transform(cos_dist)
        for dim in range(2):
            lo, hi = coords[:, dim].min(), coords[:, dim].max()
            coords[:, dim] = (coords[:, dim] - lo) / (hi - lo + 1e-8)
        seed_umap = {k: coords[i] for i, k in enumerate(seed_keys_ordered)}
    else:
        n_neighbors = min(10, n_seeds - 1)
        reducer = umap_module.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
            metric="cosine", random_state=42, low_memory=True, n_epochs=200,
        )
        coords = reducer.fit_transform(seed_embs_arr)
        for dim in range(2):
            lo, hi = coords[:, dim].min(), coords[:, dim].max()
            coords[:, dim] = (coords[:, dim] - lo) / (hi - lo + 1e-8)
        seed_umap = {k: coords[i] for i, k in enumerate(seed_keys_ordered)}
    elapsed("3. UMAP/MDS")

    # 4. K-Means 클러스터링
    # 시드 수에 따라 k 결정: 시드 3개당 1클러스터, 최대 4
    k = max(1, min(4, n_seeds // 3))
    if n_seeds < 4:
        # 시드 수가 적으면 각 시드가 자체 클러스터
        k = n_seeds
        cluster_labels = list(range(n_seeds))
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(seed_embs_arr).tolist()

    # 클러스터별 시드 목록
    cluster_seeds = defaultdict(list)  # cluster_id → [seed_key, ...]
    for i, sk in enumerate(seed_keys_ordered):
        cluster_seeds[cluster_labels[i]].append(sk)

    # 클러스터 중심 UMAP 좌표 (블록 배치 기준)
    cluster_center_umap = {}
    cluster_center_emb = {}
    for c, cseeds in cluster_seeds.items():
        cluster_center_umap[c] = np.mean([seed_umap[sk] for sk in cseeds], axis=0)
        cluster_center_emb[c] = np.mean([seed_embs[sk] for sk in cseeds], axis=0)

    elapsed("4. K-Means")

    # seed → cluster 맵
    seed_to_cluster = {sk: cluster_labels[i] for i, sk in enumerate(seed_keys_ordered)}

    # 5. 클러스터별 블록 크기 결정 (seed + fill 포함)
    # 각 클러스터의 트랙 수 = seed 수 + fill 수
    cluster_track_counts = {}
    for c in range(k):
        cseeds = cluster_seeds[c]
        n_fills = sum(len(fill_rows_map[sk]) for sk in cseeds)
        cluster_track_counts[c] = len(cseeds) + n_fills

    def block_shape(n):
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        return cols, rows

    cluster_shape = {c: block_shape(cluster_track_counts[c]) for c in range(k)}

    # 6. 클러스터 블록을 2D 격자에 배치 (UMAP 상대위치 기준)
    # UMAP 중심을 기반으로 블록 배치 순서 결정
    # k=1: 단일 블록
    # k=2: 1행 2열
    # k=3: UMAP x기준 정렬 후 1행 3열
    # k=4: UMAP x,y 중앙값으로 2x2 사분면

    cluster_ids = list(range(k))

    if k == 1:
        block_layout = {0: (0, 0)}  # cluster → (block_col, block_row)
    elif k == 2:
        sorted_c = sorted(cluster_ids, key=lambda c: cluster_center_umap[c][0])
        block_layout = {c: (i, 0) for i, c in enumerate(sorted_c)}
    elif k == 3:
        sorted_c = sorted(cluster_ids, key=lambda c: cluster_center_umap[c][0])
        block_layout = {c: (i, 0) for i, c in enumerate(sorted_c)}
    else:
        # k=4: 2x2 사분면
        xs = [cluster_center_umap[c][0] for c in cluster_ids]
        ys = [cluster_center_umap[c][1] for c in cluster_ids]
        mx, my = np.median(xs), np.median(ys)
        block_layout = {}
        for c in cluster_ids:
            cx, cy = cluster_center_umap[c]
            bc = 1 if cx >= mx else 0
            br = 1 if cy >= my else 0
            block_layout[c] = (bc, br)

    # 블록 배치: 각 블록의 전역 격자 시작 좌표 계산
    # 같은 block_row끼리 col 오프셋 누적, 다른 block_row끼리 row 오프셋 누적
    max_block_col = max(pos[0] for pos in block_layout.values())
    max_block_row = max(pos[1] for pos in block_layout.values())

    # 각 block_col별 최대 cols, 각 block_row별 최대 rows 계산
    col_widths = defaultdict(int)   # block_col → max cols
    row_heights = defaultdict(int)  # block_row → max rows

    for c, (bc, br) in block_layout.items():
        cols_c, rows_c = cluster_shape[c]
        col_widths[bc] = max(col_widths[bc], cols_c)
        row_heights[br] = max(row_heights[br], rows_c)

    # 블록 시작 좌표 (전역 격자)
    col_offsets = {}
    acc = 0
    for bc in range(max_block_col + 1):
        col_offsets[bc] = acc
        acc += col_widths[bc]

    row_offsets = {}
    acc = 0
    for br in range(max_block_row + 1):
        row_offsets[br] = acc
        acc += row_heights[br]

    block_start = {}  # cluster → (start_col, start_row)
    for c, (bc, br) in block_layout.items():
        block_start[c] = (col_offsets[bc], row_offsets[br])

    # 7. 각 클러스터 내 트랙에 격자 셀 할당
    cell_map = {}  # track_key → (col, row)
    occupied = set()  # (col, row)

    for c in range(k):
        sc, sr = block_start[c]
        cols_c, rows_c = cluster_shape[c]
        cseeds = cluster_seeds[c]

        # 시드부터 배치, 그 다음 fill (시드와 유사도 내림차순)
        tracks_in_cluster = []
        for sk in cseeds:
            tracks_in_cluster.append((sk, True, None, None))  # (tk, is_seed, sim, source_seed)
        for sk in cseeds:
            sn = seed_embs[sk] / (np.linalg.norm(seed_embs[sk]) + 1e-8)
            for r in fill_rows_map[sk]:
                tk = r["track_key"]
                emb = np.array(json_module.loads(r["embedding"]), dtype=np.float32)
                emb_n = emb / (np.linalg.norm(emb) + 1e-8)
                sim = float(np.dot(emb_n, sn))
                tracks_in_cluster.append((tk, False, sim, sk))

        # fill을 유사도 내림차순 정렬 (시드는 앞에 고정)
        seeds_part = [t for t in tracks_in_cluster if t[1]]
        fills_part = sorted([t for t in tracks_in_cluster if not t[1]], key=lambda x: -(x[2] or 0))
        ordered = seeds_part + fills_part

        for idx, (tk, is_seed, sim, source_sk) in enumerate(ordered):
            local_col = idx % cols_c
            local_row = idx // cols_c
            gcol = sc + local_col
            grow = sr + local_row
            cell_map[tk] = (gcol, grow)
            occupied.add((gcol, grow))

    elapsed("5. 블록 격자 배치")

    # 8. bridge: 먼 클러스터 쌍에 대해 중간 임베딩으로 트랙 조회
    BRIDGE_DIST_THRESHOLD = 0.35  # 클러스터 중심 UMAP 거리 > 0.35인 쌍만 bridge
    bridge_rows_all = []
    bridge_meta = {}  # track_key → {cluster_a, cluster_b, sim_a, sim_b}

    if k >= 2:
        pair_dists = []
        for ca, cb in combinations(range(k), 2):
            umap_dist = float(np.linalg.norm(cluster_center_umap[ca] - cluster_center_umap[cb]))
            pair_dists.append((umap_dist, ca, cb))
        pair_dists.sort(reverse=True)

        # 상위 min(k-1, 3) 쌍만 bridge
        bridge_pairs = [(ca, cb) for d, ca, cb in pair_dists if d > BRIDGE_DIST_THRESHOLD][:min(k - 1, 3)]

        bridge_per_pair = max(request.bridge_per_pair, 4)
        bridge_queries = []
        for ca, cb in bridge_pairs:
            mid_emb = cluster_center_emb[ca] + cluster_center_emb[cb]
            mid_emb = mid_emb / (np.linalg.norm(mid_emb) + 1e-8)
            bridge_queries.append((mid_emb, ca, cb))

        if bridge_queries:
            bridge_results = await asyncio.gather(*[
                fetch_near_raw(emb, bridge_per_pair, list(seen))
                for emb, ca, cb in bridge_queries
            ])

            cea_n = {c: cluster_center_emb[c] / (np.linalg.norm(cluster_center_emb[c]) + 1e-8) for c in range(k)}

            for (mid_emb, ca, cb), rows in zip(bridge_queries, bridge_results):
                for r in rows:
                    tk = r["track_key"]
                    if tk in seen:
                        continue
                    bemb = np.array(json_module.loads(r["embedding"]), dtype=np.float32)
                    bemb_n = bemb / (np.linalg.norm(bemb) + 1e-8)
                    sim_a = float(np.dot(bemb_n, cea_n[ca]))
                    sim_b = float(np.dot(bemb_n, cea_n[cb]))
                    seen.add(tk)
                    bridge_rows_all.append(r)
                    bridge_meta[tk] = {
                        "cluster_a": ca,
                        "cluster_b": cb,
                        "sim_a": round(sim_a, 3),
                        "sim_b": round(sim_b, 3),
                    }

    elapsed("6. bridge 조회")

    # 9. bridge 트랙을 두 클러스터 경계 근처 빈 셀에 배치
    for tk, bm in bridge_meta.items():
        ca, cb = bm["cluster_a"], bm["cluster_b"]
        sca, sra = block_start[ca]
        scb, srb = block_start[cb]
        cols_a, rows_a = cluster_shape[ca]
        cols_b, rows_b = cluster_shape[cb]

        # 두 블록의 경계 후보 셀: 블록 A 끝 col+1 ~ 블록 B 시작 col 사이 행들
        # 혹은 블록 A 끝 row+1 ~ 블록 B 시작 row 사이 col들
        # 두 블록 중심 사이를 잇는 방향으로 경계 찾기
        ca_cx = sca + cols_a / 2
        ca_cy = sra + rows_a / 2
        cb_cx = scb + cols_b / 2
        cb_cy = srb + rows_b / 2

        # 중간 지점 셀 후보 생성 (두 블록 중심 사이를 균등 샘플)
        candidates = []
        for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
            mc = round(ca_cx + t * (cb_cx - ca_cx))
            mr = round(ca_cy + t * (cb_cy - ca_cy))
            for dc in range(-1, 2):
                for dr in range(-1, 2):
                    cand = (mc + dc, mr + dr)
                    if cand not in occupied and cand[0] >= 0 and cand[1] >= 0:
                        candidates.append(cand)

        if candidates:
            # 두 블록 중심에서 가장 가까운 빈 셀 선택
            mid_cx = (ca_cx + cb_cx) / 2
            mid_cy = (ca_cy + cb_cy) / 2
            candidates.sort(key=lambda p: (p[0] - mid_cx) ** 2 + (p[1] - mid_cy) ** 2)
            chosen = candidates[0]
            cell_map[tk] = chosen
            occupied.add(chosen)
        else:
            # fallback: 전체 격자에서 빈 셀 찾기
            max_col_used = max(p[0] for p in occupied) + 1
            max_row_used = max(p[1] for p in occupied)
            cell_map[tk] = (max_col_used, 0)
            occupied.add((max_col_used, 0))

    elapsed("7. bridge 셀 배치")

    # 10. 전체 트랙 메타 구성
    all_fill_rows = []
    for rows in fill_rows_map.values():
        all_fill_rows.extend(rows)
    all_fill_rows.extend(bridge_rows_all)

    all_rows = list(seed_rows) + all_fill_rows
    all_meta = []

    for row in all_rows:
        tk = row["track_key"]
        emb_str = row["embedding"]
        if not emb_str:
            continue

        is_seed = tk in seed_set
        is_bridge = tk in bridge_meta
        source_key = fill_source_seed.get(tk)
        source_meta = seed_meta_map.get(source_key) if source_key else None
        bm = bridge_meta.get(tk)

        cluster_id = None
        if is_seed:
            cluster_id = seed_to_cluster.get(tk)
        elif source_key:
            cluster_id = seed_to_cluster.get(source_key)

        bridge_seed_a = bridge_seed_b = None
        if bm:
            ca, cb = bm["cluster_a"], bm["cluster_b"]
            # 클러스터 대표 시드로 표시
            ca_seed = cluster_seeds[ca][0] if cluster_seeds[ca] else None
            cb_seed = cluster_seeds[cb][0] if cluster_seeds[cb] else None
            ma = seed_meta_map.get(ca_seed) if ca_seed else None
            mb = seed_meta_map.get(cb_seed) if cb_seed else None
            bridge_seed_a = {"key": ca_seed, "title": ma["title"] if ma else None, "artist": ma["artist"] if ma else None, "sim": bm["sim_a"]}
            bridge_seed_b = {"key": cb_seed, "title": mb["title"] if mb else None, "artist": mb["artist"] if mb else None, "sim": bm["sim_b"]}

        gcol, grow = cell_map.get(tk, (0, 0))

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
            "cluster_id": cluster_id,
            "source_seed_key": source_key,
            "source_seed_title": source_meta["title"] if source_meta else None,
            "source_seed_artist": source_meta["artist"] if source_meta else None,
            "bridge_seed_a": bridge_seed_a,
            "bridge_seed_b": bridge_seed_b,
            "grid_col": gcol,
            "grid_row": grow,
            "x": float(gcol),
            "y": float(grow),
        })

    logger.info(
        f"music-map: {n_seeds} seeds (k={k}), "
        f"{sum(len(v) for v in fill_rows_map.values())} fills, "
        f"{len(bridge_meta)} bridges → total={len(all_meta)}"
    )
    for c in range(k):
        logger.info(f"  cluster {c}: seeds={cluster_seeds[c]}, block={cluster_shape[c]}, start={block_start[c]}")

    return {
        "tracks": all_meta,
        "seed_count": n_seeds,
        "fill_count": sum(len(v) for v in fill_rows_map.values()),
        "bridge_count": len(bridge_meta),
        "total": len(all_meta),
    }
