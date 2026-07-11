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
    1. 전체 트랙(seed + fill) UMAP 2D
    2. 좌표를 격자 셀에 스냅
    3. 겹치는 셀만 가까운 빈칸으로 이동
    4. 빈 행·열 삭제해서 전체 압축
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
    fill_source_seed = {}
    fill_rows_all = []

    for seed_key, rows in zip(seed_keys_found, seed_fill_results):
        for r in rows:
            tk = r["track_key"]
            if tk not in seen:
                seen.add(tk)
                fill_rows_all.append(r)
                fill_source_seed[tk] = seed_key

    elapsed("2. fill 조회")

    seed_meta_map = {
        r["track_key"]: {"title": r["title"], "artist": r["artist"]}
        for r in seed_rows
    }

    # 3. 전체 트랙 임베딩 수집 (seed + fill)
    all_rows = list(seed_rows) + fill_rows_all
    all_keys = []
    all_embs = []
    row_map = {}

    for row in all_rows:
        tk = row["track_key"]
        emb_str = row["embedding"]
        if not emb_str:
            continue
        all_keys.append(tk)
        all_embs.append(np.array(json_module.loads(emb_str), dtype=np.float32))
        row_map[tk] = row

    if len(all_embs) < 2:
        raise HTTPException(status_code=500, detail="임베딩 데이터가 부족합니다")

    embs_arr = np.array(all_embs, dtype=np.float32)
    n_total = len(all_keys)

    # 4. UMAP 2D (전체 트랙)
    try:
        import umap as umap_module
    except ImportError:
        raise HTTPException(status_code=500, detail="umap-learn 패키지가 필요합니다")

    n_neighbors = min(15, n_total - 1)
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        low_memory=True,
        n_epochs=200,
    )
    coords = reducer.fit_transform(embs_arr)  # (n_total, 2)
    elapsed("3. UMAP")

    # 5. 격자 셀에 스냅
    # 전체를 GRID_SIZE x GRID_SIZE 격자로 스냅
    GRID_SIZE = max(10, int(np.ceil(np.sqrt(n_total) * 1.5)))

    for dim in range(2):
        lo, hi = coords[:, dim].min(), coords[:, dim].max()
        coords[:, dim] = (coords[:, dim] - lo) / (hi - lo + 1e-8) * (GRID_SIZE - 1)

    raw_cells = [(int(round(coords[i, 0])), int(round(coords[i, 1]))) for i in range(n_total)]

    # 6. 겹치는 셀만 가까운 빈칸으로 이동 (BFS)
    occupied = {}  # (col, row) → track_key
    final_cells = {}  # track_key → (col, row)

    # seed 먼저 배치 (우선순위)
    order = (
        [i for i, tk in enumerate(all_keys) if tk in seed_set] +
        [i for i, tk in enumerate(all_keys) if tk not in seed_set]
    )

    for i in order:
        tk = all_keys[i]
        c, r = raw_cells[i]

        if (c, r) not in occupied:
            occupied[(c, r)] = tk
            final_cells[tk] = (c, r)
        else:
            # BFS로 가장 가까운 빈 셀 탐색
            found = None
            visited = set()
            queue = [(c, r)]
            visited.add((c, r))
            while queue and found is None:
                next_queue = []
                for qc, qr in queue:
                    for dc, dr in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                        nc, nr = qc + dc, qr + dr
                        if (nc, nr) not in visited:
                            visited.add((nc, nr))
                            if (nc, nr) not in occupied:
                                found = (nc, nr)
                                break
                            next_queue.append((nc, nr))
                    if found:
                        break
                queue = next_queue

            if found is None:
                # fallback: 새 열에 추가
                max_c = max(p[0] for p in occupied) + 1
                found = (max_c, 0)

            occupied[found] = tk
            final_cells[tk] = found

    elapsed("4. 격자 스냅 + 충돌 해결")

    # 7. 빈 행·열 삭제 (압축)
    used_cols = sorted(set(c for c, r in final_cells.values()))
    used_rows = sorted(set(r for c, r in final_cells.values()))

    col_remap = {c: i for i, c in enumerate(used_cols)}
    row_remap = {r: i for i, r in enumerate(used_rows)}

    for tk in final_cells:
        c, r = final_cells[tk]
        final_cells[tk] = (col_remap[c], row_remap[r])

    elapsed("5. 빈 행·열 압축")

    # 8. 4방향 중력 압축: 상하좌우 반복으로 사각형에 가깝게 뭉침
    def compress_direction(cells: dict, axis: int, reverse: bool) -> dict:
        """한 방향으로 트랙을 밀어 빈틈 제거. axis=0: col방향, axis=1: row방향"""
        from collections import defaultdict
        groups = defaultdict(list)
        for tk, pos in cells.items():
            key = pos[1 - axis]  # 고정 축
            groups[key].append((pos[axis], tk))
        result = {}
        for key, items in groups.items():
            items.sort(reverse=reverse)
            for new_idx, (_, tk) in enumerate(items):
                idx = (len(items) - 1 - new_idx) if reverse else new_idx
                if axis == 0:
                    result[tk] = (idx, key)
                else:
                    result[tk] = (key, idx)
        return result

    for _ in range(5):  # 수렴할 때까지 반복
        prev = dict(final_cells)
        final_cells = compress_direction(final_cells, axis=1, reverse=False)  # 위로
        final_cells = compress_direction(final_cells, axis=0, reverse=False)  # 왼쪽으로
        final_cells = compress_direction(final_cells, axis=1, reverse=True)   # 아래로
        final_cells = compress_direction(final_cells, axis=0, reverse=True)   # 오른쪽으로
        if final_cells == prev:
            break

    elapsed("6. 4방향 중력 압축")

    # 10. 결과 구성
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
        f"music-map: {len(seed_rows)} seeds, {len(fill_rows_all)} fills "
        f"→ total={len(tracks)}, grid={len(used_cols)}x{len(used_rows)}"
    )

    return {
        "tracks": tracks,
        "seed_count": len(seed_rows),
        "fill_count": len(fill_rows_all),
        "total": len(tracks),
    }
