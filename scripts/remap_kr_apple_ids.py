"""
US/QM/QZ ISRC 곡들을 한국 스토어에서 재조회해서 apple_music_id 업데이트
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
APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID")
APPLE_MUSICKIT_KEY_ID = os.getenv("APPLE_MUSICKIT_KEY_ID")
APPLE_MUSICKIT_KEY_PATH = os.getenv("APPLE_MUSICKIT_KEY_PATH")

APPLE_BATCH_SIZE = 25
SLEEP_BETWEEN = 0.3


def generate_musickit_token() -> str:
    now = int(time.time())
    with open(APPLE_MUSICKIT_KEY_PATH, "rb") as f:
        private_key = f.read()
    payload = {"iss": APPLE_TEAM_ID, "iat": now, "exp": now + 3600}
    headers = {"alg": "ES256", "kid": APPLE_MUSICKIT_KEY_ID}
    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)


async def fetch_kr_ids_by_isrc(client, token, isrcs):
    resp = await client.get(
        "https://api.music.apple.com/v1/catalog/kr/songs",
        params={"filter[isrc]": ",".join(isrcs)},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if resp.status_code == 429:
        logger.warning("Rate limit, waiting 10s")
        await asyncio.sleep(10)
        return {}
    result = {}
    for item in resp.json().get("data", []):
        isrc = item.get("attributes", {}).get("isrc")
        apple_id = item.get("id")
        if isrc and apple_id and isrc not in result:
            result[isrc] = apple_id
    return result


async def main():
    token = generate_musickit_token()
    token_time = time.time()

    pool = await asyncpg.create_pool(DATABASE_URL, statement_cache_size=0, min_size=1, max_size=2)

    rows = await pool.fetch("""
        SELECT track_key, isrc, apple_music_id
        FROM tracks
        WHERE isrc IS NOT NULL
          AND (isrc LIKE 'US%' OR isrc LIKE 'QM%' OR isrc LIKE 'QZ%')
          AND apple_music_id IS NOT NULL
        ORDER BY track_key
    """)

    total = len(rows)
    logger.info(f"재매핑 대상: {total}개")

    updated = 0
    same = 0
    not_found = 0

    async with httpx.AsyncClient() as client:
        for i in range(0, total, APPLE_BATCH_SIZE):
            batch = rows[i:i + APPLE_BATCH_SIZE]

            if time.time() - token_time > 3500:
                token = generate_musickit_token()
                token_time = time.time()

            isrcs = [r["isrc"] for r in batch]
            kr_map = await fetch_kr_ids_by_isrc(client, token, isrcs)

            update_rows = []
            for r in batch:
                kr_id = kr_map.get(r["isrc"])
                if kr_id and kr_id != r["apple_music_id"]:
                    update_rows.append((kr_id, r["track_key"]))
                    updated += 1
                elif kr_id:
                    same += 1
                else:
                    not_found += 1

            if update_rows:
                await pool.executemany(
                    "UPDATE tracks SET apple_music_id = $1 WHERE track_key = $2",
                    update_rows,
                )

            if i % 500 == 0:
                logger.info(f"진행: {i+len(batch)}/{total} | 업데이트: {updated} | 동일: {same} | 없음: {not_found}")

            await asyncio.sleep(SLEEP_BETWEEN)

    await pool.close()
    logger.info(f"완료 — 업데이트: {updated}, 동일: {same}, 한국에 없음: {not_found}")


if __name__ == "__main__":
    asyncio.run(main())
