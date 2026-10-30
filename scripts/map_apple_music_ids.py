"""
Spotify track_key → ISRC → Apple Music ID 배치 매핑 스크립트

흐름:
  1. tracks 테이블에서 apple_music_id가 없는 track_key 전체 조회
  2. 50개씩 Spotify API로 ISRC 조회
  3. ISRC로 Apple Music ID 조회
  4. tracks 테이블에 배치 UPDATE (가벼운 테이블, HNSW 없음)

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
import sys
import json
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

BATCH_SIZE = 50
APPLE_BATCH_SIZE = 25
SLEEP_BETWEEN_BATCHES = 3.0
SLEEP_BETWEEN_APPLE = 0.3


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
    resp = await client.get(
        "https://api.spotify.com/v1/tracks",
        params={"ids": ",".join(track_keys)},
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


async def main():
    musickit_token = generate_musickit_token()

    pool = await asyncpg.create_pool(DATABASE_URL, statement_cache_size=0, min_size=1, max_size=2)
    rows = await pool.fetch(
        "SELECT track_key FROM tracks WHERE apple_music_id IS NULL AND track_key IS NOT NULL ORDER BY track_key"
    )
    track_keys = [r["track_key"] for r in rows]
    total = len(track_keys)
    logger.info(f"매핑 대상: {total}개")

    mapped = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        spotify_token = await get_spotify_token(client)
        spotify_token_time = time.time()
        musickit_token_time = time.time()

        for i in range(0, total, BATCH_SIZE):
            batch = track_keys[i:i + BATCH_SIZE]

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
                failed += len(batch)
                logger.info(f"진행: {i + len(batch)}/{total} | 성공: {mapped} | 실패: {failed}")
                await asyncio.sleep(SLEEP_BETWEEN_BATCHES)
                continue

            # 2. ISRC → Apple Music ID
            # ISRC 앞 2자리로 우선 스토어 결정, 없으면 순서대로 시도
            ISRC_COUNTRY_MAP = {
                "KR": "kr", "JP": "jp", "US": "us", "GB": "gb",
                "FR": "fr", "DE": "de", "NL": "nl", "SE": "se",
                "AU": "au", "CA": "ca", "IT": "it", "ES": "es",
                "IN": "in", "BR": "br", "MX": "mx", "BE": "be",
                "QZ": "us", "QM": "us",  # 국제 ISRC → us
            }
            FALLBACK_STOREFRONTS = ["kr", "us", "jp", "gb", "fr", "de", "au", "ca", "se", "nl", "it", "es", "br", "mx", "in", "be"]

            isrc_list = list(isrc_map.values())
            apple_map: dict[str, str] = {}

            for j in range(0, len(isrc_list), APPLE_BATCH_SIZE):
                isrc_chunk = isrc_list[j:j + APPLE_BATCH_SIZE]

                # 청크 내 ISRC별 우선 스토어로 그룹화
                store_groups: dict[str, list[str]] = {}
                for isrc in isrc_chunk:
                    prefix = isrc[:2].upper()
                    store = ISRC_COUNTRY_MAP.get(prefix, "us")
                    store_groups.setdefault(store, []).append(isrc)

                # 우선 스토어별로 조회
                found_isrcs = set()
                for store, isrcs in store_groups.items():
                    result = await fetch_apple_ids_by_isrc(client, musickit_token, isrcs, store)
                    apple_map.update(result)
                    found_isrcs.update(result.keys())
                    await asyncio.sleep(SLEEP_BETWEEN_APPLE)

                # 못 찾은 ISRC는 fallback 스토어 순서대로 재시도
                remaining = [i for i in isrc_chunk if i not in found_isrcs]
                for store in FALLBACK_STOREFRONTS:
                    if not remaining:
                        break
                    result = await fetch_apple_ids_by_isrc(client, musickit_token, remaining, store)
                    apple_map.update(result)
                    found_isrcs.update(result.keys())
                    remaining = [i for i in remaining if i not in found_isrcs]
                    await asyncio.sleep(SLEEP_BETWEEN_APPLE)

            # 3. tracks 테이블에 배치 UPDATE
            update_rows = []
            for track_key, isrc in isrc_map.items():
                apple_id = apple_map.get(isrc)
                if apple_id:
                    update_rows.append((isrc, apple_id, track_key))
                    mapped += 1
                else:
                    failed += 1

            if update_rows:
                await pool.executemany(
                    "UPDATE tracks SET isrc = $1, apple_music_id = $2 WHERE track_key = $3",
                    update_rows,
                )

            logger.info(f"진행: {i + len(batch)}/{total} | 성공: {mapped} | 실패: {failed}")
            await asyncio.sleep(SLEEP_BETWEEN_BATCHES)

    await pool.close()
    logger.info(f"완료 — 성공: {mapped}, Apple Music 없음: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
