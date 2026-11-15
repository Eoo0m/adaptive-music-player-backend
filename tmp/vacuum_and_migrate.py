"""VACUUM → 배치 UPDATE → 인덱스 재생성"""

import asyncio
import os
import time
import asyncpg
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 1000


async def main():
    # PgBouncer가 아닌 직접 연결 필요 (VACUUM은 트랜잭션 밖에서 실행해야 함)
    # PgBouncer URL의 포트 6543 → 직접 연결 5432로 변경
    database_url = os.getenv("DATABASE_URL")

    conn = await asyncpg.connect(database_url, statement_cache_size=0, timeout=3600)

    # 좀비 정리
    print("좀비 쿼리 정리...", flush=True)
    await conn.execute("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE state = 'active'
        AND query LIKE '%track_embeddings%'
        AND pid != pg_backend_pid()
    """)
    await asyncio.sleep(3)

    # dead tuple 확인
    row = await conn.fetchrow("""
        SELECT n_live_tup, n_dead_tup, last_autovacuum
        FROM pg_stat_user_tables WHERE relname = 'track_embeddings'
    """)
    print(f"live: {row['n_live_tup']}, dead: {row['n_dead_tup']}, last_vacuum: {row['last_autovacuum']}", flush=True)

    # VACUUM
    print("\nVACUUM 실행 중 (수 분 걸릴 수 있음)...", flush=True)
    t0 = time.time()
    await conn.execute("VACUUM track_embeddings")
    print(f"VACUUM 완료 ({time.time()-t0:.1f}s)", flush=True)

    # dead tuple 재확인
    row = await conn.fetchrow("""
        SELECT n_live_tup, n_dead_tup
        FROM pg_stat_user_tables WHERE relname = 'track_embeddings'
    """)
    print(f"VACUUM 후 - live: {row['n_live_tup']}, dead: {row['n_dead_tup']}", flush=True)

    # 대상 조회
    print("\n대상 track_key 조회...", flush=True)
    keys = await conn.fetch(
        "SELECT track_key FROM track_embeddings WHERE cover_image_url LIKE $1",
        "%xopigdxsobjbagmazgba%",
    )
    total = len(keys)
    print(f"변경 대상: {total}건", flush=True)

    if total == 0:
        print("변경할 항목이 없습니다.")
    else:
        updated = 0
        for i in range(0, total, BATCH_SIZE):
            batch_keys = [row["track_key"] for row in keys[i : i + BATCH_SIZE]]
            t0 = time.time()
            result = await conn.execute(
                """
                UPDATE track_embeddings
                SET cover_image_url = REPLACE(cover_image_url,
                    'https://xopigdxsobjbagmazgba.supabase.co/storage/v1/',
                    'https://yxihvsnrszafmluyjqke.supabase.co/storage/v1/')
                WHERE track_key = ANY($1::text[])
                """,
                batch_keys,
            )
            count = int(result.split()[-1])
            updated += count
            print(f"  진행: {updated}/{total} ({updated*100//total}%) - {time.time()-t0:.1f}s", flush=True)

        print(f"\nUPDATE 완료: {updated}건", flush=True)

    # 인덱스 재생성
    print("\nHNSW 인덱스 재생성 (1/3: embedding 64d)...", flush=True)
    t0 = time.time()
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_track_embedding_hnsw ON track_embeddings
        USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)
    print(f"  완료 ({time.time()-t0:.1f}s)", flush=True)

    print("HNSW 인덱스 재생성 (2/3: projected 512d)...", flush=True)
    t0 = time.time()
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_track_projected_embedding_hnsw ON track_embeddings
        USING hnsw (projected_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)
    print(f"  완료 ({time.time()-t0:.1f}s)", flush=True)

    print("HNSW 인덱스 재생성 (3/3: itemtower 128d)...", flush=True)
    t0 = time.time()
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_track_itemtower_embedding_hnsw ON track_embeddings
        USING hnsw (itemtower_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)
    print(f"  완료 ({time.time()-t0:.1f}s)", flush=True)

    print("\n전체 완료!", flush=True)
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
