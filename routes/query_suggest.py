"""
트랙 목록 → 쿼리 추천 (/query-suggest)

알고리즘:
  1. track_keys의 SimGCL 64d 임베딩 조회 (track_vectors_simgcl)
  2. Ward Hierarchical Clustering → medoid 선택 (interest_profile.py 동일 로직)
  3. medoid → playlist_proj → CLIP shared space (512d)
  4. taxonomy 태그 projected embeddings와 cosine 유사도
  5. 카테고리별 최고 태그 선택 (top_k_global 내에서)
  6. 서로 다른 카테고리 2개 조합 → 최대 3개 쿼리 반환
     (CAT_ORDER 기준으로 정렬, genre는 항상 뒤)
"""

import asyncio
import json
import logging
import numpy as np
import torch

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from scipy.cluster.hierarchy import fcluster, linkage

from database import get_db_pool
from models import playlist_clip_model, device
from utils import get_openai_embedding_with_retry, mmr_rerank
from telemetry import elapsed_ms, get_request_id, now_ms

logger = logging.getLogger(__name__)

router = APIRouter()

ALPHA = 2.0
TOP_K_GLOBAL = 20
MAX_QUERIES = 3

# genre는 항상 마지막
CAT_ORDER = ["era", "season", "place", "language", "mood", "activity", "other", "genre"]

# 서버 시작 시 main.py에서 주입
_tag_proj: Optional[np.ndarray] = None   # (num_tags, 512)
_tag_names: Optional[list] = None
_tag_cats: Optional[list] = None


def set_taxonomy_state(tag_proj: np.ndarray, tag_names: list, tag_cats: list):
    global _tag_proj, _tag_names, _tag_cats
    _tag_proj = tag_proj
    _tag_names = tag_names
    _tag_cats = tag_cats


class QuerySuggestRequest(BaseModel):
    track_keys: List[str]
    max_queries: Optional[int] = MAX_QUERIES


def _compute_medoids(embeddings: np.ndarray) -> np.ndarray:
    """Ward clustering → 각 클러스터 medoid 반환 (interest_profile.py 동일)."""
    n = len(embeddings)
    if n == 1:
        return embeddings

    Z = linkage(embeddings, method="ward", metric="euclidean")
    labels = fcluster(Z, ALPHA, criterion="distance")

    medoids = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        cluster_embs = embeddings[idx]
        centroid = cluster_embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        sims = (cluster_embs / norms) @ centroid
        medoids.append(cluster_embs[int(np.argmax(sims))])

    return np.array(medoids, dtype=np.float32)


def _build_queries(medoid_proj: np.ndarray, max_queries: int) -> list[str]:
    """medoid CLIP 임베딩 → taxonomy 태그 유사도 → genre + 다른 카테고리 조합 쿼리."""
    tag_norm    = _tag_proj / (np.linalg.norm(_tag_proj,    axis=1, keepdims=True) + 1e-8)
    medoid_norm = medoid_proj / (np.linalg.norm(medoid_proj, axis=1, keepdims=True) + 1e-8)
    sims = (medoid_norm @ tag_norm.T).mean(axis=0)  # (num_tags,)

    # genre 포함 보장을 위해 충분히 넓게 탐색
    top_global = np.argsort(-sims)[:TOP_K_GLOBAL * 3]

    cat_best: dict[str, tuple[str, float]] = {}
    for idx in top_global:
        cat = _tag_cats[idx]
        score = float(sims[idx])
        if cat not in cat_best or score > cat_best[cat][1]:
            cat_best[cat] = (_tag_names[idx], score)

    if "genre" not in cat_best:
        return []

    genre_tag = cat_best["genre"][0]
    # genre 제외한 나머지 카테고리를 유사도 내림차순으로 정렬
    other_cats = [cat for cat, _ in sorted(
        ((c, v) for c, v in cat_best.items() if c != "genre"),
        key=lambda x: -x[1][1]
    )]

    queries, seen = [], set()
    for cat in other_cats:
        other_tag = cat_best[cat][0]
        # CAT_ORDER 기준: genre는 항상 뒤
        pair = (other_tag, genre_tag)
        if pair not in seen:
            seen.add(pair)
            queries.append(f"{other_tag} {genre_tag}")
        if len(queries) >= max_queries:
            break

    return queries


async def _mood_search_for_query(query: str, pool, top_k: int = 9) -> list:
    """쿼리 텍스트 → OpenAI 임베딩 → CLIP proj → ANN + MMR → 트랙 목록"""
    # 캐시 조회
    row = await pool.fetchrow(
        "SELECT embedding::text FROM keyword_embeddings WHERE keyword = $1",
        query.lower()
    )
    if row and row["embedding"]:
        raw_emb = np.array(json.loads(row["embedding"]), dtype=np.float32)
    else:
        raw_emb = get_openai_embedding_with_retry(query)
        raw_emb = np.array(raw_emb, dtype=np.float32)
        try:
            await pool.execute(
                "INSERT INTO keyword_embeddings (keyword, embedding) VALUES ($1, $2::vector) ON CONFLICT DO NOTHING",
                query.lower(), str(raw_emb.tolist())
            )
        except Exception:
            pass

    with torch.no_grad():
        t = torch.tensor(raw_emb).unsqueeze(0).to(device)
        proj = playlist_clip_model.caption_proj(t).cpu().numpy()[0]  # (512,)

    candidates = 80
    rows = await pool.fetch(
        """
        SELECT
            te.track_key::text,
            t.title::text,
            t.artist::text,
            t.cover_image_url::text,
            t.apple_music_id::text,
            t.playlist_count,
            te.projected_embedding::text
        FROM track_vectors_clip te
        JOIN tracks t ON te.track_key = t.track_key
        ORDER BY te.projected_embedding <=> $1::vector
        LIMIT $2
        """,
        str(proj.tolist()), candidates,
    )
    if not rows:
        return []

    cand_ids, cand_embs, cand_data = [], [], {}
    for r in rows:
        emb_str = r["projected_embedding"]
        if emb_str:
            emb = np.array(json.loads(emb_str), dtype=np.float32)
            cand_ids.append(r["track_key"])
            cand_embs.append(emb)
            cand_data[r["track_key"]] = {
                "track_key": r["track_key"],
                "track": r["title"],
                "track_name": r["title"],
                "artist": r["artist"],
                "cover_image_url": r["cover_image_url"],
                "apple_music_id": r["apple_music_id"],
                "playlist_count": r["playlist_count"],
            }

    selected_ids, _ = mmr_rerank(proj, np.array(cand_embs), cand_ids, top_k=top_k, lambda_param=0.7)
    return [cand_data[k] for k in selected_ids if k in cand_data]


@router.post("/query-suggest")
async def query_suggest(request: QuerySuggestRequest):
    """트랙 목록 → 추천 검색 쿼리 + 각 쿼리의 mood 검색 결과 반환"""
    t0 = now_ms()
    rid = get_request_id()

    if _tag_proj is None:
        raise HTTPException(status_code=503, detail="Taxonomy not loaded yet")
    if not request.track_keys:
        raise HTTPException(status_code=400, detail="track_keys is empty")

    logger.info(f"[{rid}] /query-suggest tracks={len(request.track_keys)}")

    pool = await get_db_pool()

    # 1. SimGCL 임베딩 조회
    rows = await pool.fetch(
        "SELECT track_key, embedding::text FROM track_vectors_simgcl WHERE track_key = ANY($1)",
        request.track_keys,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="No embeddings found for given track_keys")

    embs = []
    for row in rows:
        emb = row["embedding"]
        if isinstance(emb, str):
            emb = json.loads(emb)
        embs.append(emb)
    embs_np = np.array(embs, dtype=np.float32)

    # 2. Clustering → medoid
    medoid_embs = _compute_medoids(embs_np)  # (k, 64)

    # 3. playlist_proj → CLIP space
    with torch.no_grad():
        t = torch.tensor(medoid_embs).to(device)
        medoid_proj = playlist_clip_model.playlist_proj(t).cpu().numpy()  # (k, 512)

    # 4. 쿼리 생성
    queries = _build_queries(medoid_proj, request.max_queries or MAX_QUERIES)

    # 5. 각 쿼리별 mood 검색 결과 병렬 조회
    track_results = await asyncio.gather(
        *[_mood_search_for_query(q, pool) for q in queries]
    )

    result_items = [
        {"query": q, "tracks": tracks}
        for q, tracks in zip(queries, track_results)
    ]

    logger.info(f"[{rid}] total={elapsed_ms(t0):.1f}ms k={len(medoid_embs)} queries={queries}")
    return {"suggestions": result_items, "k": len(medoid_embs)}
