"""VACUUM FULL로 테이블 재구축"""

import asyncio
import os
import time
import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def main():
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"), statement_cache_size=0, timeout=3600)

    # 좀비 정리
    print("좀비 쿼리 정리...", flush=True)
    await conn.execute("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE state = 'active'
        AND pid != pg_backend_pid()
    """)
    await asyncio.sleep(3)

    # VACUUM FULL (테이블 완전 재구축 - 시간 걸림)
    print("VACUUM FULL 실행 중 (테이블 재구축, 수 분 소요)...", flush=True)
    t0 = time.time()
    await conn.execute("VACUUM FULL track_embeddings")
    print(f"VACUUM FULL 완료 ({time.time()-t0:.1f}s)", flush=True)

    # 상태 확인
    r = await conn.fetchrow("""
        SELECT n_live_tup, n_dead_tup,
               pg_size_pretty(pg_table_size('track_embeddings')) as table_size,
               pg_size_pretty(pg_total_relation_size('track_embeddings')) as total_size
        FROM pg_stat_user_tables WHERE relname = 'track_embeddings'
    """)
    print(f"live: {r['n_live_tup']}, dead: {r['n_dead_tup']}")
    print(f"테이블: {r['table_size']}, 전체: {r['total_size']}")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
