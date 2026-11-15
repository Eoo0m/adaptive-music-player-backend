"""인덱스 삭제 → UPDATE → 인덱스 재생성"""

import asyncio
import os
import time
import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def main():
    database_url = os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(database_url, statement_cache_size=0, timeout=600)

    # 0. 좀비 쿼리 정리
    print("좀비 쿼리 정리...", flush=True)
    await conn.execute("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE state = 'active'
        AND query LIKE '%track_embeddings%'
        AND pid != pg_backend_pid()
    """)
    await asyncio.sleep(3)

    # 1. HNSW 인덱스 삭제
    print("1/4 HNSW 인덱스 삭제...", flush=True)
    t0 = time.time()
    await conn.execute("DROP INDEX IF EXISTS idx_track_embedding_hnsw")
    await conn.execute("DROP INDEX IF EXISTS idx_track_projected_embedding_hnsw")
    await conn.execute("DROP INDEX IF EXISTS idx_track_itemtower_embedding_hnsw")
    print(f"  완료 ({time.time()-t0:.1f}s)", flush=True)

    # 2. UPDATE
    print("2/4 cover_image_url 일괄 변경...", flush=True)
    t0 = time.time()
    result = await conn.execute("""
        UPDATE track_embeddings
        SET cover_image_url = REPLACE(cover_image_url,
          'https://xopigdxsobjbagmazgba.supabase.co/storage/v1/',
          'https://yxihvsnrszafmluyjqke.supabase.co/storage/v1/')
        WHERE cover_image_url LIKE '%xopigdxsobjbagmazgba%'
    """)
    print(f"  완료: {result} ({time.time()-t0:.1f}s)", flush=True)

    # 3. HNSW 인덱스 재생성
    print("3/4 HNSW 인덱스 재생성 (embedding 64d)...", flush=True)
    t0 = time.time()
    await conn.execute("""
        CREATE INDEX idx_track_embedding_hnsw ON track_embeddings
        USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)
    print(f"  완료 ({time.time()-t0:.1f}s)", flush=True)

    print("4/4 HNSW 인덱스 재생성 (projected 512d + itemtower 128d)...", flush=True)
    t0 = time.time()
    await conn.execute("""
        CREATE INDEX idx_track_projected_embedding_hnsw ON track_embeddings
        USING hnsw (projected_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)
    await conn.execute("""
        CREATE INDEX idx_track_itemtower_embedding_hnsw ON track_embeddings
        USING hnsw (itemtower_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)
    print(f"  완료 ({time.time()-t0:.1f}s)", flush=True)

    print("전체 완료!", flush=True)
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
