import asyncio
import os
import time

import asyncpg
from dotenv import load_dotenv

load_dotenv()

OLD_HOST = "xopigdxsobjbagmazgba.supabase.co"
NEW_HOST = "yxihvsnrszafmluyjqke.supabase.co"
LIMIT = int(os.getenv("COVER_URL_UPDATE_LIMIT", "1000"))


async def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    conn = await asyncpg.connect(database_url, statement_cache_size=0, timeout=600)
    try:
        before = await conn.fetchval(
            """
            SELECT count(*)
            FROM track_embeddings
            WHERE cover_image_url LIKE '%' || $1 || '%'
            """,
            OLD_HOST,
        )
        print(f"remaining before: {before}")

        started_at = time.time()
        result = await conn.execute(
            """
            WITH target AS (
                SELECT track_key
                FROM track_embeddings
                WHERE cover_image_url LIKE '%' || $1 || '%'
                LIMIT $3
            )
            UPDATE track_embeddings t
            SET cover_image_url = replace(t.cover_image_url, $1, $2)
            FROM target
            WHERE t.track_key = target.track_key
            """,
            OLD_HOST,
            NEW_HOST,
            LIMIT,
        )
        elapsed = time.time() - started_at

        after = await conn.fetchval(
            """
            SELECT count(*)
            FROM track_embeddings
            WHERE cover_image_url LIKE '%' || $1 || '%'
            """,
            OLD_HOST,
        )
        print(f"updated: {result}")
        print(f"elapsed: {elapsed:.2f}s")
        print(f"remaining after: {after}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
