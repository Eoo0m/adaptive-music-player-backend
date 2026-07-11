import asyncio
import json as json_module
import logging
import time
from collections import defaultdict

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
    fill_per_seed: Optional[int] = 10


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
    1. 시드만 UMAP → 격자 셀 확정
    2. 필은 소속 시드 셀 주변 BFS로 빈 칸 배치
    3. 중심으로 당기기로 뭉치기
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

    seed_embs = {
        r["track_key"]: np.array(json_module.loads(r["embedding"]), dtype=np.float32)
        for r in seed_rows
    }
    row_map = {r["track_key"]: r for r in seed_rows}

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
    fill_per_seed = max(request.fill_per_seed, 5)
    exclude_base = list(seed_set)

    seed_fill_results = await asyncio.gather(*[
        fetch_near_raw(seed_embs[key], fill_per_seed, exclude_base)
        for key in seed_keys_found
    ])

    seen = set(seed_set)
    fill_source_seed = {}  # track_key → seed_key
    fill_by_seed = defaultdict(list)  # seed_key → [row, ...]

    for seed_key, rows in zip(seed_keys_found, seed_fill_results):
        for r in rows:
            tk = r["track_key"]
            if tk not in seen:
                seen.add(tk)
                fill_by_seed[seed_key].append(r)
                fill_source_seed[tk] = seed_key
                row_map[tk] = r

    elapsed("2. fill 조회")

    seed_meta_map = {
        r["track_key"]: {"title": r["title"], "artist": r["artist"]}
        for r in seed_rows
    }

    # 3. 시드만 UMAP
    try:
        import umap as umap_module
    except ImportError:
        raise HTTPException(status_code=500, detail="umap-learn 패키지가 필요합니다")

    n_seeds = len(seed_keys_found)
    seed_embs_arr = np.array([seed_embs[k] for k in seed_keys_found], dtype=np.float32)

    if n_seeds == 1:
        seed_coords = {seed_keys_found[0]: (0, 0)}
    elif n_seeds <= 3:
        from sklearn.manifold import MDS
        seed_n = seed_embs_arr / (np.linalg.norm(seed_embs_arr, axis=1, keepdims=True) + 1e-8)
        cos_dist = np.clip(1.0 - seed_n @ seed_n.T, 0, 2).astype(np.float64)
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
        coords = mds.fit_transform(cos_dist)
        for dim in range(2):
            lo, hi = coords[:, dim].min(), coords[:, dim].max()
            coords[:, dim] = (coords[:, dim] - lo) / (hi - lo + 1e-8)
        seed_coords = {k: coords[i] for i, k in enumerate(seed_keys_found)}
    else:
        n_neighbors = min(15, n_seeds - 1)
        reducer = umap_module.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.3,
            metric="cosine", random_state=42, low_memory=True, n_epochs=200,
        )
        coords = reducer.fit_transform(seed_embs_arr)
        for dim in range(2):
            lo, hi = coords[:, dim].min(), coords[:, dim].max()
            coords[:, dim] = (coords[:, dim] - lo) / (hi - lo + 1e-8)
        seed_coords = {k: coords[i] for i, k in enumerate(seed_keys_found)}

    elapsed("3. UMAP (시드만)")

    # 4. 시드 UMAP 좌표 → 격자 스냅
    # 시드 간 간격을 fill 수에 비례해 여유있게 확보
    max_fills = max((len(fill_by_seed[sk]) for sk in seed_keys_found), default=0)
    seed_spacing = max(3, int(np.ceil(np.sqrt(max_fills + 1))) + 1)
    GRID_SIZE = seed_spacing * max(int(np.ceil(np.sqrt(n_seeds) * 2)), 2)

    occupied = {}   # (col, row) → track_key
    final_cells = {}  # track_key → (col, row)

    # 시드 배치 (UMAP 좌표 → 격자, 충돌 시 BFS)
    raw_seed_cells = {}
    for sk in seed_keys_found:
        ux, uy = seed_coords[sk]
        c = int(round(ux * (GRID_SIZE - 1)))
        r = int(round(uy * (GRID_SIZE - 1)))
        raw_seed_cells[sk] = (c, r)

    for sk in seed_keys_found:
        c, r = raw_seed_cells[sk]
        if (c, r) not in occupied:
            occupied[(c, r)] = sk
            final_cells[sk] = (c, r)
        else:
            # BFS로 가장 가까운 빈 셀
            found = None
            visited = {(c, r)}
            queue = [(c, r)]
            while queue and not found:
                next_q = []
                for qc, qr in queue:
                    for dc, dr in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                        nc, nr = qc + dc, qr + dr
                        if (nc, nr) not in visited:
                            visited.add((nc, nr))
                            if (nc, nr) not in occupied:
                                found = (nc, nr)
                                break
                            next_q.append((nc, nr))
                    if found:
                        break
                queue = next_q
            pos = found or (max(p[0] for p in occupied) + 1, 0)
            occupied[pos] = sk
            final_cells[sk] = pos

    elapsed("4. 시드 격자 배치")

    # 5. fill을 소속 시드 주변 BFS로 배치
    # 시드와 유사도 높은 fill부터 배치 (가까운 셀 먼저 차지)
    for sk in seed_keys_found:
        sc, sr = final_cells[sk]
        fills = fill_by_seed[sk]

        # 유사도 내림차순 정렬
        sn = seed_embs[sk] / (np.linalg.norm(seed_embs[sk]) + 1e-8)
        def fill_sim(r):
            emb = np.array(json_module.loads(r["embedding"]), dtype=np.float32)
            emb_n = emb / (np.linalg.norm(emb) + 1e-8)
            return float(np.dot(emb_n, sn))
        fills_sorted = sorted(fills, key=fill_sim, reverse=True)

        # BFS 큐: 시드 셀 인접부터 탐색
        bfs_visited = set(occupied.keys())
        bfs_queue = []
        for dc, dr in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nc, nr = sc + dc, sr + dr
            if (nc, nr) not in bfs_visited:
                bfs_queue.append((nc, nr))
                bfs_visited.add((nc, nr))

        for fill_row in fills_sorted:
            tk = fill_row["track_key"]
            # BFS에서 빈 셀 찾기
            while bfs_queue:
                pos = bfs_queue.pop(0)
                if pos not in occupied:
                    occupied[pos] = tk
                    final_cells[tk] = pos
                    # 이 셀 인접 셀을 큐에 추가
                    for dc, dr in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                        np_ = (pos[0] + dc, pos[1] + dr)
                        if np_ not in bfs_visited:
                            bfs_visited.add(np_)
                            bfs_queue.append(np_)
                    break
            else:
                # fallback
                max_c = max(p[0] for p in occupied) + 1
                occupied[(max_c, 0)] = tk
                final_cells[tk] = (max_c, 0)

    elapsed("5. fill 시드 주변 배치")

    # 6. 빈 행·열 압축
    used_cols = sorted(set(c for c, r in final_cells.values()))
    used_rows = sorted(set(r for c, r in final_cells.values()))
    col_remap = {c: i for i, c in enumerate(used_cols)}
    row_remap = {r: i for i, r in enumerate(used_rows)}
    for tk in final_cells:
        c, r = final_cells[tk]
        final_cells[tk] = (col_remap[c], row_remap[r])

    elapsed("6. 빈 행·열 압축")

    # 7. 중심으로 당기기
    for iteration in range(30):
        moved = False
        occupied_set = set(final_cells.values())
        all_pos = list(final_cells.values())
        cx = sum(p[0] for p in all_pos) / len(all_pos)
        cy = sum(p[1] for p in all_pos) / len(all_pos)

        order = sorted(final_cells.keys(), key=lambda tk: -(
            (final_cells[tk][0] - cx) ** 2 + (final_cells[tk][1] - cy) ** 2
        ))

        for tk in order:
            c, r = final_cells[tk]
            dc = cx - c
            dr = cy - r
            if (dc ** 2 + dr ** 2) ** 0.5 < 0.5:
                continue

            steps = []
            if abs(dc) >= abs(dr):
                steps.append((int(round(dc / abs(dc))), 0))
                if abs(dr) > 0.3:
                    steps.append((0, int(round(dr / abs(dr)))))
            else:
                steps.append((0, int(round(dr / abs(dr)))))
                if abs(dc) > 0.3:
                    steps.append((int(round(dc / abs(dc))), 0))

            for sc, sr in steps:
                nc, nr = c + sc, r + sr
                if (nc, nr) not in occupied_set:
                    occupied_set.discard((c, r))
                    occupied_set.add((nc, nr))
                    final_cells[tk] = (nc, nr)
                    moved = True
                    break

        if not moved:
            logger.info(f"  중심 압축 수렴: {iteration + 1}회")
            break

    elapsed("7. 중심으로 당기기")

    # 8. 결과 구성
    all_keys = list(seed_keys_found) + [tk for tk in fill_source_seed]
    tracks = []
    for tk in all_keys:
        row = row_map[tk]
        is_seed = tk in seed_set
        source_key = fill_source_seed.get(tk)
        source_meta = seed_meta_map.get(source_key) if source_key else None
        gcol, grow = final_cells.get(tk, (0, 0))

        tracks.append({
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
            "grid_col": gcol,
            "grid_row": grow,
            "x": float(gcol),
            "y": float(grow),
        })

    elapsed("8. 결과 구성")
    logger.info(
        f"music-map: {n_seeds} seeds, {len(fill_source_seed)} fills → total={len(tracks)}"
    )

    return {
        "tracks": tracks,
        "seed_count": n_seeds,
        "fill_count": len(fill_source_seed),
        "total": len(tracks),
    }
