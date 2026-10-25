"""
Spotify track_key → ISRC → Apple Music ID 배치 매핑 스크립트

사용법:
    python scripts/map_apple_music_ids.py

환경변수 필요:
    DATABASE_URL, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET,
    APPLE_TEAM_ID, APPLE_MUSICKIT_KEY_ID, APPLE_MUSICKIT_KEY_PATH
"""

import asyncio
import asyncpg
import httpx
import os
import time
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

BATCH_SIZE = 50       # Spotify API 한 번에 최대 50개
APPLE_BATCH_SIZE = 25 # Apple ISRC 필터 한 번에 최대 25개
STOREFRONT = "kr"     # 한국 카탈로그 기준, 없으면 us 시도


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
    return resp.json()["access_token"]


async def fetch_isrcs_from_spotify(
    client: httpx.AsyncClient, spotify_token: str, track_keys: list[str]
) -> dict[str, str]:
    """track_key → ISRC 매핑 반환"""
    ids = ",".join(track_keys)
    resp = await client.get(
        "https://api.spotify.com/v1/tracks",
        params={"ids": ids},
        headers={"Authorization": f"Bearer {spotify_token}"},
    )
    result = {}
    for track in resp.json().get("tracks", []):
        if track and track.get("external_ids", {}).get("isrc"):
            result[track["id"]] = track["external_ids"]["isrc"]
    return result


async def fetch_apple_ids_by_isrc(
    client: httpx.AsyncClient, musickit_token: str, isrcs: list[str], storefront: str = STOREFRONT
) -> dict[str, str]:
    """ISRC → Apple Music ID 매핑 반환"""
    resp = await client.get(
        f"https://api.music.apple.com/v1/catalog/{storefront}/songs",
        params={"filter[isrc]": ",".join(isrcs)},
        headers={"Authorization": f"Bearer {musickit_token}"},
    )
    if resp.status_code == 401:
        raise RuntimeError("MusicKit token invalid")

    result = {}
    for item in resp.json().get("data", []):
        isrc = item.get("attributes", {}).get("isrc")
        apple_id = item.get("id")
        if isrc and apple_id and isrc not in result:
            result[isrc] = apple_id
    return result


async def main():
    pool = await asyncpg.create_pool(DATABASE_URL, statement_cache_size=0)
    musickit_token = generate_musickit_token()

    # 아직 매핑 안 된 트랙만 가져오기
    rows = await pool.fetch(
        "SELECT track_key FROM track_embeddings WHERE apple_music_id IS NULL AND track_key IS NOT NULL"
    )
    track_keys = [r["track_key"] for r in rows]
    total = len(track_keys)
    logger.info(f"매핑 대상 트랙: {total}개")

    mapped = 0
    failed = 0

    async with httpx.AsyncClient(timeout=30) as client:
        spotify_token = await get_spotify_token(client)
        spotify_token_time = time.time()

        for i in range(0, total, BATCH_SIZE):
            batch = track_keys[i:i + BATCH_SIZE]

            # Spotify 토큰 만료 시 갱신 (1시간)
            if time.time() - spotify_token_time > 3500:
                spotify_token = await get_spotify_token(client)
                spotify_token_time = time.time()

            # 1. Spotify → ISRC
            try:
                isrc_map = await fetch_isrcs_from_spotify(client, spotify_token, batch)
            except Exception as e:
                logger.warning(f"Spotify fetch failed for batch {i}: {e}")
                failed += len(batch)
                await asyncio.sleep(2)
                continue

            if not isrc_map:
                continue

            # 2. ISRC → Apple Music ID (25개씩)
            isrc_list = list(isrc_map.values())
            apple_map: dict[str, str] = {}
            for j in range(0, len(isrc_list), APPLE_BATCH_SIZE):
                isrc_chunk = isrc_list[j:j + APPLE_BATCH_SIZE]
                try:
                    chunk_result = await fetch_apple_ids_by_isrc(client, musickit_token, isrc_chunk)
                    apple_map.update(chunk_result)
                except Exception as e:
                    logger.warning(f"Apple Music fetch failed: {e}")
                    # kr 스토어프론트 실패 시 us 재시도
                    try:
                        chunk_result = await fetch_apple_ids_by_isrc(client, musickit_token, isrc_chunk, "us")
                        apple_map.update(chunk_result)
                    except Exception:
                        pass
                await asyncio.sleep(0.1)

            # 3. DB 업데이트
            for track_key, isrc in isrc_map.items():
                apple_id = apple_map.get(isrc)
                if apple_id:
                    await pool.execute(
                        "UPDATE track_embeddings SET isrc = $1, apple_music_id = $2 WHERE track_key = $3",
                        isrc, apple_id, track_key
                    )
                    mapped += 1
                else:
                    # ISRC만 저장 (Apple Music에 없는 트랙)
                    await pool.execute(
                        "UPDATE track_embeddings SET isrc = $1 WHERE track_key = $2",
                        isrc, track_key
                    )
                    failed += 1

            logger.info(f"진행: {i + len(batch)}/{total} | 매핑 성공: {mapped} | 실패: {failed}")
            await asyncio.sleep(0.3)  # rate limit 방지

    await pool.close()
    logger.info(f"완료 — 성공: {mapped}/{total}, Apple Music 없음: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
