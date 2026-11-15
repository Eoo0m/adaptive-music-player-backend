"""
테스트용: 트랙 1개씩 조회 → API → DB UPDATE
목적: 배치 없이 1개씩 처리해도 DB가 터지는지 확인
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

SLEEP_BETWEEN_TRACKS = 1.5  # 트랙 사이 쉬는 시간 (초)


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


async def main():
    musickit_token = generate_musickit_token()

    pool = await asyncpg.create_pool(DATABASE_URL, statement_cache_size=0, min_size=1, max_size=2)
    rows = await pool.fetch(
        "SELECT track_key FROM track_embeddings WHERE apple_music_id IS NULL AND track_key IS NOT NULL ORDER BY track_key LIMIT 200"
    )
    track_keys = [r["track_key"] for r in rows]
    total = len(track_keys)
    logger.info(f"테스트 대상: {total}개")

    mapped = 0
    failed = 0

    async with httpx.AsyncClient() as client:
        spotify_token = await get_spotify_token(client)
        spotify_token_time = time.time()
        musickit_token_time = time.time()

        for i, track_key in enumerate(track_keys):
            if time.time() - spotify_token_time > 3500:
                spotify_token = await get_spotify_token(client)
                spotify_token_time = time.time()

            if time.time() - musickit_token_time > 3500:
                musickit_token = generate_musickit_token()
                musickit_token_time = time.time()

            # 1. Spotify → ISRC (1개)
            resp = await client.get(
                "https://api.spotify.com/v1/tracks",
                params={"ids": track_key},
                headers={"Authorization": f"Bearer {spotify_token}"},
                timeout=30,
            )
            if resp.status_code == 401:
                raise RuntimeError("Spotify token expired")
            resp.raise_for_status()

            track_data = resp.json().get("tracks", [None])[0]
            if not track_data:
                failed += 1
                await asyncio.sleep(SLEEP_BETWEEN_TRACKS)
                continue

            isrc = track_data.get("external_ids", {}).get("isrc")
            if not isrc:
                failed += 1
                await asyncio.sleep(SLEEP_BETWEEN_TRACKS)
                continue

            # 2. Apple Music ID (ISRC)
            apple_id = None
            for storefront in ["kr", "us"]:
                resp2 = await client.get(
                    f"https://api.music.apple.com/v1/catalog/{storefront}/songs",
                    params={"filter[isrc]": isrc},
                    headers={"Authorization": f"Bearer {musickit_token}"},
                    timeout=30,
                )
                if resp2.status_code == 401:
                    raise RuntimeError("MusicKit token invalid")
                if resp2.status_code == 429:
                    raise RuntimeError("Apple rate limit")
                data = resp2.json().get("data", [])
                if data:
                    apple_id = data[0].get("id")
                    break

            if not apple_id:
                failed += 1
                await asyncio.sleep(SLEEP_BETWEEN_TRACKS)
                continue

            # 3. DB UPDATE (1개)
            await pool.execute(
                "UPDATE track_embeddings SET isrc = $1, apple_music_id = $2 WHERE track_key = $3",
                isrc, apple_id, track_key,
            )
            mapped += 1
            logger.info(f"[{i+1}/{total}] {track_key} → {apple_id} | 성공: {mapped} 실패: {failed}")

            await asyncio.sleep(SLEEP_BETWEEN_TRACKS)

    await pool.close()
    logger.info(f"완료 — 성공: {mapped}, 실패: {failed}")


if __name__ == "__main__":
    asyncio.run(main())
