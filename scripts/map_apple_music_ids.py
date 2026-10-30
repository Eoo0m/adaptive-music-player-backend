"""
Spotify track_key → ISRC → Apple Music ID 배치 매핑 스크립트

사용법:
    python scripts/map_apple_music_ids.py

환경변수 필요:
    DATABASE_URL, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET,
    APPLE_TEAM_ID, APPLE_MUSICKIT_KEY_ID, APPLE_MUSICKIT_KEY_PATH

Supabase free tier 친화적:
- 배치 UPDATE (트랙마다 개별 쿼리 X)
- 배치 간 sleep 5초
- 진행상황 파일 저장 (중단 후 이어서 가능)
"""

import asyncio
import asyncpg
import httpx
import os
import time
import json
import logging
from dotenv import load_dotenv
from jose import jwt

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID")
APPLE_MUSICKIT_KEY_ID = os.getenv("APPLE_MUSICKIT_KEY_ID")
APPLE_MUSICKIT_KEY_PATH = os.getenv("APPLE_MUSICKIT_KEY_PATH")

BATCH_SIZE = 50        # Spotify API 한 번에 최대 50개
APPLE_BATCH_SIZE = 25  # Apple ISRC 필터 한 번에 최대 25개
SLEEP_BETWEEN_BATCHES = 5.0   # DB Disk IO 부하 완화 (초)
SLEEP_BETWEEN_APPLE = 0.3     # Apple API rate limit 완화
PROGRESS_FILE = "scripts/.map_progress.json"
DB_WRITE_EVERY = 10    # N 배치마다 DB에 한 번 씀 (50 * 10 = 500개마다)


def generate_musickit_token() -> str:
    now = int(time.time())
    with open(APPLE_MUSICKIT_KEY_PATH, "rb") as f:
        private_key = f.read()
    payload = {"iss": APPLE_TEAM_ID, "iat": now, "exp": now + 3600}
    headers = {"alg": "ES256", "kid": APPLE_MUSICKIT_KEY_ID}
    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)


async def get_spotify_token(client: httpx.AsyncClient) -> str:
    resp = await client.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


async def fetch_isrcs_from_spotify(
    client: httpx.AsyncClient, spotify_token: str, track_keys: list[str]
) -> dict[str, str]:
    """track_key → ISRC"""
    ids = ",".join(track_keys)
    resp = await client.get(
        "https://api.spotify.com/v1/tracks",
        params={"ids": ids},
        headers={"Authorization": f"Bearer {spotify_token}"},
        timeout=30,
    )
    if resp.status_code == 401:
        raise RuntimeError("Spotify token expired")
    resp.raise_for_status()
    result = {}
    for track in resp.json().get("tracks", []):
        if track and track.get("external_ids", {}).get("isrc"):
            result[track["id"]] = track["external_ids"]["isrc"]
    return result


async def fetch_apple_ids_by_isrc(
    client: httpx.AsyncClient, musickit_token: str, isrcs: list[str], storefront: str = "kr"
) -> dict[str, str]:
    """ISRC → Apple Music ID"""
    resp = await client.get(
        f"https://api.music.apple.com/v1/catalog/{storefront}/songs",
        params={"filter[isrc]": ",".join(isrcs)},
        headers={"Authorization": f"Bearer {musickit_token}"},
        timeout=30,
    )
    if resp.status_code == 401:
        raise RuntimeError("MusicKit token invalid")
    if resp.status_code == 429:
        raise RuntimeError("Apple rate limit")
    result = {}
    for item in resp.json().get("data", []):
        isrc = item.get("attributes", {}).get("isrc")
        apple_id = item.get("id")
        if isrc and apple_id and isrc not in result:
            result[isrc] = apple_id
    return result


async def batch_update_db(pool, updates: list[tuple[str, str, str]]):
    """[(isrc, apple_id, track_key), ...] 배치 UPDATE"""
    if not updates:
        return
    # executemany 사용 — 한 트랜잭션에서 처리
    async with pool.acquire() as conn:
        await conn.executemany(
            "UPDATE track_embeddings SET isrc = $1, apple_music_id = $2 WHERE track_key = $3",
            updates,
        )


async def batch_update_isrc_only(pool, updates: list[tuple[str, str]]):
    """[(isrc, track_key), ...] — apple_id 없어도 isrc는 저장"""
    if not updates:
        return
    async with pool.acquire() as conn:
        await conn.executemany(
            "UPDATE track_embeddings SET isrc = $1 WHERE track_key = $2 AND isrc IS NULL",
            updates,
        )


def load_progress() -> int:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f).get("offset", 0)
    return 0


def save_progress(offset: int):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"offset": offset}, f)


async def main():
    pool = await asyncpg.create_pool(DATABASE_URL, statement_cache_size=0, min_size=1, max_size=3)
    musickit_token = generate_musickit_token()
    i = 0

    try:
        rows = await pool.fetch(
            "SELECT track_key FROM track_embeddings WHERE apple_music_id IS NULL AND track_key IS NOT NULL ORDER BY track_key"
        )
        track_keys = [r["track_key"] for r in rows]
        total = len(track_keys)

        start_offset = load_progress()
        logger.info(f"매핑 대상: {total}개, 시작 offset: {start_offset}")

        mapped = 0
        failed = 0
        pending_updates: list[tuple[str, str, str]] = []
        pending_isrc: list[tuple[str, str]] = []

        async with httpx.AsyncClient() as client:
            spotify_token = await get_spotify_token(client)
            spotify_token_time = time.time()
            musickit_token_time = time.time()

            for i in range(start_offset, total, BATCH_SIZE):
                batch = track_keys[i:i + BATCH_SIZE]
                batch_num = (i - start_offset) // BATCH_SIZE + 1

                # 토큰 갱신
                if time.time() - spotify_token_time > 3500:
                    spotify_token = await get_spotify_token(client)
                    spotify_token_time = time.time()
                    logger.info("Spotify 토큰 갱신")

                if time.time() - musickit_token_time > 3500:
                    musickit_token = generate_musickit_token()
                    musickit_token_time = time.time()
                    logger.info("MusicKit 토큰 갱신")

                # 1. Spotify → ISRC
                isrc_map = await fetch_isrcs_from_spotify(client, spotify_token, batch)

                if not isrc_map:
                    save_progress(i + len(batch))
                    continue

                # ISRC 누적
                for tk, isrc in isrc_map.items():
                    pending_isrc.append((isrc, tk))

                # 2. ISRC → Apple Music ID
                isrc_list = list(isrc_map.values())
                apple_map: dict[str, str] = {}

                for j in range(0, len(isrc_list), APPLE_BATCH_SIZE):
                    isrc_chunk = isrc_list[j:j + APPLE_BATCH_SIZE]
                    chunk_result = await fetch_apple_ids_by_isrc(client, musickit_token, isrc_chunk, "kr")
                    if not chunk_result:
                        chunk_result = await fetch_apple_ids_by_isrc(client, musickit_token, isrc_chunk, "us")
                    apple_map.update(chunk_result)
                    await asyncio.sleep(SLEEP_BETWEEN_APPLE)

                # 결과 누적
                for track_key, isrc in isrc_map.items():
                    apple_id = apple_map.get(isrc)
                    if apple_id:
                        pending_updates.append((isrc, apple_id, track_key))
                        mapped += 1
                    else:
                        failed += 1

                # DB_WRITE_EVERY 배치마다 또는 마지막 배치에 DB 쓰기
                is_last = (i + BATCH_SIZE >= total)
                if batch_num % DB_WRITE_EVERY == 0 or is_last:
                    await batch_update_isrc_only(pool, pending_isrc)
                    await batch_update_db(pool, pending_updates)
                    pending_isrc.clear()
                    pending_updates.clear()
                    logger.info(f"DB 쓰기 완료: {i + len(batch)}/{total} | 성공: {mapped} | 실패: {failed}")

                save_progress(i + len(batch))
                logger.info(f"진행: {i + len(batch)}/{total} | 성공: {mapped} | 실패: {failed}")

                await asyncio.sleep(SLEEP_BETWEEN_BATCHES)

        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        logger.info(f"완료 — 성공: {mapped}/{total}, Apple Music 없음: {failed}")

    except Exception as e:
        logger.error(f"오류 발생 — offset {i} 저장 후 중단: {e}")
        save_progress(i)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
