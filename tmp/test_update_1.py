"""딱 1건만 UPDATE 테스트"""
import asyncio, os, time, asyncpg
from dotenv import load_dotenv
load_dotenv()

async def main():
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"), statement_cache_size=0)

    # 1건 가져오기
    key = await conn.fetchval(
        "SELECT track_key FROM track_embeddings WHERE cover_image_url LIKE $1 LIMIT 1",
        "%xopigdxsobjbagmazgba%"
    )
    print(f"대상: {key}", flush=True)

    t0 = time.time()
    await conn.execute(
        "UPDATE track_embeddings SET cover_image_url = REPLACE(cover_image_url, $1, $2) WHERE track_key = $3",
        "xopigdxsobjbagmazgba", "yxihvsnrszafmluyjqke", key
    )
    print(f"1건 UPDATE: {time.time()-t0:.2f}s", flush=True)

    await conn.close()

asyncio.run(main())
