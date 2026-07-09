import logging
import random
import json as json_module
import numpy as np
import torch
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List

from database import get_db_pool
from models import playlist_clip_model, device
from utils import get_openai_embedding_with_retry, mmr_rerank
from routes.auth import get_current_user
from telemetry import elapsed_ms, get_request_id, now_ms

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/playlist-builder")


# ============== 시간대 감지 ==============

def get_season() -> str:
    """현재 월 기준 계절 반환"""
    month = datetime.now().month
    if month in (6, 7, 8):
        return "여름"
    elif month in (9, 10, 11):
        return "가을"
    elif month in (12, 1, 2):
        return "겨울"
    else:
        return "봄"


def get_time_of_day() -> str:
    """현재 시간 기준 시간대 반환 (KST 기준)"""
    hour = datetime.now().hour
    if 0 <= hour < 6:
        return "새벽"
    elif 6 <= hour < 12:
        return "아침"
    elif 12 <= hour < 18:
        return "오후"
    else:
        return "밤"


# ============== 키워드 → 임베딩 + 후보 100곡 가져오기 ==============

async def fetch_keyword_candidates(keyword: str, limit: int = 100):
    """키워드로 projected_embedding 기준 상위 limit곡 가져오기"""
    keyword_embedding = get_openai_embedding_with_retry(keyword)

    keyword_tensor = torch.tensor(keyword_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        projected_query = playlist_clip_model.caption_proj(keyword_tensor)
        projected_embedding_np = projected_query.cpu().numpy()[0]

    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT
            t.track_key::text,
            t.title::text,
            t.artist::text,
            t.album::text,
            t.playlist_count,
            t.cover_image_url::text,
            t.projected_embedding::text
        FROM track_embeddings t
        WHERE t.projected_embedding IS NOT NULL
        ORDER BY t.projected_embedding <=> $1::vector
        LIMIT $2
        """,
        str(projected_embedding_np.tolist()),
        limit
    )

    candidates = []
    embeddings = []
    for row in rows:
        emb_str = row["projected_embedding"]
        if emb_str:
            emb = np.array(json_module.loads(emb_str), dtype=np.float32)
            candidates.append({
                "track_key": row["track_key"],
                "track_name": row["title"],
                "artist": row["artist"],
                "album": row["album"],
                "playlist_count": row["playlist_count"],
                "cover_image_url": row["cover_image_url"],
            })
            embeddings.append(emb)

    return candidates, np.array(embeddings), projected_embedding_np


async def fetch_similar_candidates(track_key: str, limit: int = 50):
    """track_key 기반 유사곡 limit개 가져오기"""
    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        WITH query_track AS (
            SELECT embedding
            FROM track_embeddings
            WHERE track_key = $1
            LIMIT 1
        )
        SELECT
            t.track_key::text,
            t.title::text,
            t.artist::text,
            t.album::text,
            t.playlist_count,
            t.cover_image_url::text,
            (1 - (t.embedding <=> q.embedding))::float AS similarity
        FROM track_embeddings t, query_track q
        WHERE t.track_key != $1
          AND t.embedding IS NOT NULL
        ORDER BY t.embedding <=> q.embedding
        LIMIT $2
        """,
        track_key,
        limit
    )

    candidates = []
    for row in rows:
        candidates.append({
            "track_key": row["track_key"],
            "track_name": row["title"],
            "artist": row["artist"],
            "album": row["album"],
            "playlist_count": row["playlist_count"],
            "cover_image_url": row["cover_image_url"],
            "similarity": row["similarity"],
        })
    return candidates


async def fetch_projected_embeddings(track_keys: List[str]):
    """track_key 목록의 projected_embedding 가져오기"""
    if not track_keys:
        return {}
    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT track_key::text, projected_embedding::text
        FROM track_embeddings
        WHERE track_key = ANY($1::text[])
          AND projected_embedding IS NOT NULL
        """,
        track_keys
    )
    result = {}
    for row in rows:
        emb_str = row["projected_embedding"]
        if emb_str:
            result[row["track_key"]] = np.array(json_module.loads(emb_str), dtype=np.float32)
    return result


def apply_mmr_with_seen_penalty(
    query_emb: np.ndarray,
    keyword_candidates: list,
    keyword_embs: np.ndarray,
    seen_track_keys: set,
    top_k: int = 12,
    lambda_param: float = 0.65,
    seen_penalty: float = 0.3,
    popularity_weight: float = 0.25,
):
    """
    MMR 적용.

    relevance = sim(query, track) + popularity_weight * log1p(playlist_count) / log1p(max_count)
    MMR score = lambda * relevance - (1 - lambda) * max_sim(track, selected)

    seen_track_keys에 있는 곡은 relevance에 seen_penalty 배율 적용.
    """
    if len(keyword_candidates) == 0:
        return []

    candidate_ids = [c["track_key"] for c in keyword_candidates]
    candidate_embs = keyword_embs

    # Normalize embeddings
    query_emb_n = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    norms = np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-10
    candidate_embs_norm = candidate_embs / norms

    # 코사인 유사도 (N,)
    query_sims = candidate_embs_norm @ query_emb_n

    # 인기도 정규화: log1p(count) / log1p(max_count) → [0, 1]
    counts = np.array([max(0, c.get("playlist_count") or 0) for c in keyword_candidates], dtype=np.float32)
    max_count = counts.max() if counts.max() > 0 else 1.0
    pop_scores = np.log1p(counts) / np.log1p(max_count)  # (N,)

    # relevance = similarity + popularity_weight * popularity
    relevance = query_sims + popularity_weight * pop_scores  # (N,)

    # seen 곡에 패널티
    for i, cid in enumerate(candidate_ids):
        if cid in seen_track_keys:
            relevance[i] *= seen_penalty

    doc_sims = candidate_embs_norm @ candidate_embs_norm.T  # (N, N)

    selected = []
    selected_mask = np.zeros(len(candidate_ids), dtype=bool)
    selected_scores = []

    for _ in range(min(top_k, len(candidate_ids))):
        if len(selected) == 0:
            best_idx = int(np.argmax(relevance))
        else:
            remaining_mask = ~selected_mask
            remaining_indices = np.where(remaining_mask)[0]
            if len(remaining_indices) == 0:
                break
            selected_indices_local = np.array(selected)
            max_sim_to_selected = np.max(
                doc_sims[remaining_indices][:, selected_indices_local], axis=1
            )
            mmr = lambda_param * relevance[remaining_indices] - (1 - lambda_param) * max_sim_to_selected
            best_local_idx = int(np.argmax(mmr))
            best_idx = remaining_indices[best_local_idx]

        selected.append(best_idx)
        selected_mask[best_idx] = True
        selected_scores.append(float(relevance[best_idx]))

    return [keyword_candidates[i] for i in selected]


# ============== Pydantic Models ==============

class PlaylistBuilderStartRequest(BaseModel):
    seen_track_keys: List[str] = []  # 새로고침 시 이미 본 곡 목록


class PlaylistBuilderNextRequest(BaseModel):
    selected_track_key: str          # 유저가 선택한 곡
    keyword_track_keys: List[str]    # 초기 키워드 곡 track_key 목록 (풀 유지용)
    seen_track_keys: List[str]       # 지금까지 보여준 모든 곡 track_key
    step: int                        # 현재 단계 (1~5)
    keyword: str                     # 원래 키워드 (임베딩 재사용)


class PlaylistBuilderFinishRequest(BaseModel):
    selected_track_keys: List[str]   # 유저가 선택한 6곡


class PlaylistSaveRequest(BaseModel):
    name: str
    track_keys: List[str]




# ============== API Endpoints ==============

@router.post("/start")
async def playlist_builder_start(request: PlaylistBuilderStartRequest = PlaylistBuilderStartRequest(), user: dict = Depends(get_current_user)):
    """
    플레이리스트 빌더 시작.
    - 현재 시간대 감지 (새벽/아침/오후/밤)
    - '여름 {시간대}' 키워드로 100곡 검색
    - MMR로 12곡 선별 (6x2 그리드)
    - 키워드 풀(100곡) track_key도 반환 (이후 next 단계에서 재사용)
    """
    request_id = get_request_id()
    start_ms = now_ms()

    time_of_day = get_time_of_day()
    season = get_season()
    keyword = f"{season} {time_of_day}"
    user_name = user.get("display_name") or user.get("email", "").split("@")[0]

    logger.info(f"[{request_id}] /playlist-builder/start keyword='{keyword}' user={user.get('user_id')}")

    try:
        candidates, embs, query_emb = await fetch_keyword_candidates(keyword, limit=100)

        if len(candidates) == 0:
            raise HTTPException(status_code=500, detail="키워드 검색 결과가 없습니다.")

        # MMR로 12곡 선별 (새로고침 시 seen_track_keys 패널티 적용)
        selected = apply_mmr_with_seen_penalty(
            query_emb, candidates, embs,
            seen_track_keys=set(request.seen_track_keys),
            top_k=12,
            lambda_param=0.65,
        )

        # 키워드 풀 전체 track_key 목록
        keyword_pool_keys = [c["track_key"] for c in candidates]

        logger.info(
            f"[{request_id}] /playlist-builder/start done "
            f"total={elapsed_ms(start_ms):.1f}ms keyword='{keyword}' candidates={len(candidates)} selected={len(selected)}"
        )

        return {
            "time_of_day": time_of_day,
            "season": season,
            "keyword": keyword,
            "user_name": user_name,
            "candidates": selected,           # 12곡 (그리드에 표시)
            "keyword_pool_keys": keyword_pool_keys,  # 100곡 풀 (next 단계 재사용)
            "keyword_pool_data": candidates,   # 100곡 데이터 (next에서 임베딩 재조회 대신 사용)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] /playlist-builder/start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/next")
async def playlist_builder_next(request: PlaylistBuilderNextRequest, user: dict = Depends(get_current_user)):
    """
    다음 후보 12곡 선별.
    - 선택한 곡의 유사곡 50곡 + 키워드 풀 합산
    - 중복 제거 후 projected_embedding 기준 MMR
    - 이미 본 곡은 후순위 패널티
    """
    request_id = get_request_id()
    start_ms = now_ms()

    logger.info(
        f"[{request_id}] /playlist-builder/next "
        f"selected='{request.selected_track_key}' step={request.step}"
    )

    try:
        # 1. 선택한 곡의 유사곡 50곡
        similar = await fetch_similar_candidates(request.selected_track_key, limit=50)

        # 2. 키워드 풀 track_key들의 데이터 조회
        pool = await get_db_pool()
        keyword_rows = await pool.fetch(
            """
            SELECT
                t.track_key::text,
                t.title::text,
                t.artist::text,
                t.album::text,
                t.playlist_count,
                t.cover_image_url::text
            FROM track_embeddings t
            WHERE t.track_key = ANY($1::text[])
            """,
            request.keyword_track_keys
        )
        keyword_data = {
            row["track_key"]: {
                "track_key": row["track_key"],
                "track_name": row["title"],
                "artist": row["artist"],
                "album": row["album"],
                "playlist_count": row["playlist_count"],
                "cover_image_url": row["cover_image_url"],
            }
            for row in keyword_rows
        }

        # 3. 합산 후 중복 제거 (similar 우선, keyword 추가)
        seen_keys = set()
        merged = []
        for s in similar:
            if s["track_key"] not in seen_keys:
                seen_keys.add(s["track_key"])
                merged.append({
                    "track_key": s["track_key"],
                    "track_name": s["track_name"],
                    "artist": s["artist"],
                    "album": s["album"],
                    "playlist_count": s["playlist_count"],
                    "cover_image_url": s["cover_image_url"],
                })
        for key in request.keyword_track_keys:
            if key not in seen_keys and key in keyword_data:
                seen_keys.add(key)
                merged.append(keyword_data[key])

        if len(merged) == 0:
            raise HTTPException(status_code=500, detail="후보 곡이 없습니다.")

        # 4. projected_embedding 가져오기
        merged_keys = [c["track_key"] for c in merged]
        emb_map = await fetch_projected_embeddings(merged_keys)

        # embedding이 있는 것만 필터
        filtered_candidates = []
        filtered_embs = []
        for c in merged:
            emb = emb_map.get(c["track_key"])
            if emb is not None:
                filtered_candidates.append(c)
                filtered_embs.append(emb)

        if len(filtered_candidates) == 0:
            raise HTTPException(status_code=500, detail="임베딩 데이터가 없습니다.")

        filtered_embs_np = np.array(filtered_embs)

        # 5. 두 query 준비
        # 5a. 선택한 곡의 projected_embedding
        selected_emb_map = await fetch_projected_embeddings([request.selected_track_key])
        track_query_emb = selected_emb_map.get(request.selected_track_key)

        # 5b. 키워드 projected_embedding
        keyword_embedding = get_openai_embedding_with_retry(request.keyword)
        keyword_tensor = torch.tensor(keyword_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            projected_query = playlist_clip_model.caption_proj(keyword_tensor)
            keyword_query_emb = projected_query.cpu().numpy()[0]

        if track_query_emb is None:
            track_query_emb = keyword_query_emb

        # 6. 선택곡 기준 MMR 6곡 + 키워드 기준 MMR 6곡 → 합산 (중복 제거)
        seen_set = set(request.seen_track_keys)

        track_based = apply_mmr_with_seen_penalty(
            track_query_emb,
            filtered_candidates,
            filtered_embs_np,
            seen_track_keys=seen_set,
            top_k=6,
            lambda_param=0.65,
            seen_penalty=0.3,
        )
        track_based_keys = {c["track_key"] for c in track_based}

        keyword_based = apply_mmr_with_seen_penalty(
            keyword_query_emb,
            filtered_candidates,
            filtered_embs_np,
            seen_track_keys=seen_set | track_based_keys,  # 이미 뽑힌 건 패널티
            top_k=6,
            lambda_param=0.65,
            seen_penalty=0.3,
        )

        # 선택곡 기준 먼저, 키워드 기준 뒤에 (중복 제거)
        seen_result = set()
        selected = []
        for c in track_based + keyword_based:
            if c["track_key"] not in seen_result:
                seen_result.add(c["track_key"])
                selected.append(c)

        logger.info(
            f"[{request_id}] /playlist-builder/next done "
            f"total={elapsed_ms(start_ms):.1f}ms merged={len(merged)} "
            f"track_based={len(track_based)} keyword_based={len(keyword_based)}"
        )

        return {
            "candidates": selected,  # 최대 12곡
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] /playlist-builder/next error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/finish")
async def playlist_builder_finish(request: PlaylistBuilderFinishRequest, user: dict = Depends(get_current_user)):
    """
    플레이리스트 완성.
    - 선택한 6곡을 기반으로 각 곡의 유사곡 취합
    - 랜덤 6곡 선별 → 선택 6곡 + 유사 6곡 = 총 12곡
    """
    request_id = get_request_id()
    start_ms = now_ms()

    logger.info(
        f"[{request_id}] /playlist-builder/finish "
        f"selected_count={len(request.selected_track_keys)}"
    )

    if len(request.selected_track_keys) == 0:
        raise HTTPException(status_code=400, detail="선택된 곡이 없습니다.")

    try:
        # 선택한 6곡의 정보 가져오기
        pool = await get_db_pool()
        selected_rows = await pool.fetch(
            """
            SELECT
                t.track_key::text,
                t.title::text,
                t.artist::text,
                t.album::text,
                t.playlist_count,
                t.cover_image_url::text
            FROM track_embeddings t
            WHERE t.track_key = ANY($1::text[])
            """,
            request.selected_track_keys
        )

        selected_data = {row["track_key"]: dict(row) for row in selected_rows}

        # 순서 유지
        selected_tracks = []
        for key in request.selected_track_keys:
            if key in selected_data:
                row = selected_data[key]
                selected_tracks.append({
                    "track_key": row["track_key"],
                    "track_name": row["title"],
                    "artist": row["artist"],
                    "album": row["album"],
                    "playlist_count": row["playlist_count"],
                    "cover_image_url": row["cover_image_url"],
                })

        # 각 선택한 곡에서 유사곡 15개씩 가져오기 (총 pool)
        selected_keys_set = set(request.selected_track_keys)
        similar_pool = {}

        for track_key in request.selected_track_keys:
            similar = await fetch_similar_candidates(track_key, limit=15)
            for s in similar:
                key = s["track_key"]
                if key not in selected_keys_set and key not in similar_pool:
                    similar_pool[key] = s

        similar_list = list(similar_pool.values())

        # 가중 랜덤 6곡 선택
        # weight = similarity * 0.6 + log1p(playlist_count) / log1p(max_count) * 0.4
        pick_count = min(6, len(similar_list))
        if len(similar_list) > pick_count:
            sim_arr = np.array([s.get("similarity", 0.5) for s in similar_list], dtype=np.float32)
            cnt_arr = np.array([max(0, s.get("playlist_count") or 0) for s in similar_list], dtype=np.float32)
            max_cnt = cnt_arr.max() if cnt_arr.max() > 0 else 1.0
            pop_arr = np.log1p(cnt_arr) / np.log1p(max_cnt)
            weights = 0.6 * sim_arr + 0.4 * pop_arr
            weights = np.clip(weights, 1e-6, None)
            probs = weights / weights.sum()
            chosen_indices = np.random.choice(len(similar_list), size=pick_count, replace=False, p=probs)
            filler_tracks_raw = [similar_list[i] for i in chosen_indices]
        else:
            filler_tracks_raw = similar_list

        filler_tracks = []
        for s in filler_tracks_raw:
            filler_tracks.append({
                "track_key": s["track_key"],
                "track_name": s["track_name"],
                "artist": s["artist"],
                "album": s["album"],
                "playlist_count": s["playlist_count"],
                "cover_image_url": s["cover_image_url"],
            })

        # 최종 12곡: 선택 6곡 + 유사 6곡
        playlist = selected_tracks + filler_tracks

        logger.info(
            f"[{request_id}] /playlist-builder/finish done "
            f"total={elapsed_ms(start_ms):.1f}ms "
            f"selected={len(selected_tracks)} filler={len(filler_tracks)} total={len(playlist)}"
        )

        return {
            "playlist": playlist,
            "selected_tracks": selected_tracks,
            "filler_tracks": filler_tracks,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] /playlist-builder/finish error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
async def playlist_builder_save(request: PlaylistSaveRequest, user: dict = Depends(get_current_user)):
    """완성된 플레이리스트 DB 저장"""
    request_id = get_request_id()
    if not request.name or not request.track_keys:
        raise HTTPException(status_code=400, detail="이름과 트랙 목록이 필요합니다.")
    try:
        pool = await get_db_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO user_playlists (user_id, name, track_keys, created_at)
            VALUES ($1::uuid, $2, $3, NOW())
            RETURNING id, created_at
            """,
            user["user_id"], request.name, request.track_keys
        )
        logger.info(f"[{request_id}] Playlist saved: id={row['id']} user={user['user_id']}")
        return {"success": True, "playlist_id": row["id"], "created_at": row["created_at"].isoformat()}
    except Exception as e:
        logger.error(f"[{request_id}] /playlist-builder/save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/my-playlists")
async def get_my_playlists(user: dict = Depends(get_current_user)):
    """저장된 플레이리스트 목록 + 트랙 전체 데이터"""
    request_id = get_request_id()
    try:
        pool = await get_db_pool()
        rows = await pool.fetch(
            """
            SELECT id, name, track_keys, created_at
            FROM user_playlists
            WHERE user_id = $1::uuid
            ORDER BY created_at DESC
            """,
            user["user_id"]
        )

        playlists = []
        for row in rows:
            track_keys = row["track_keys"] or []
            track_rows = await pool.fetch(
                """
                SELECT track_key::text, title::text, artist::text, album::text,
                       cover_image_url::text, playlist_count
                FROM track_embeddings
                WHERE track_key = ANY($1::text[])
                """,
                track_keys
            )
            track_map = {t["track_key"]: t for t in track_rows}
            tracks = []
            for key in track_keys:
                t = track_map.get(key)
                if t:
                    tracks.append({
                        "track_key": t["track_key"],
                        "track_name": t["title"],
                        "artist": t["artist"],
                        "album": t["album"],
                        "cover_image_url": t["cover_image_url"],
                        "playlist_count": t["playlist_count"],
                    })
            playlists.append({
                "id": row["id"],
                "name": row["name"],
                "tracks": tracks,
                "created_at": row["created_at"].isoformat(),
            })

        return {"playlists": playlists}
    except Exception as e:
        logger.error(f"[{request_id}] /playlist-builder/my-playlists error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/playlist/{playlist_id}")
async def delete_playlist(playlist_id: int, user: dict = Depends(get_current_user)):
    """플레이리스트 삭제"""
    request_id = get_request_id()
    try:
        pool = await get_db_pool()
        result = await pool.execute(
            "DELETE FROM user_playlists WHERE id = $1 AND user_id = $2::uuid",
            playlist_id, user["user_id"]
        )
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="플레이리스트를 찾을 수 없습니다.")
        logger.info(f"[{request_id}] Playlist deleted: id={playlist_id}")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] /playlist-builder/delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
