import asyncio, os, asyncpg
from dotenv import load_dotenv
load_dotenv()

async def main():
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"), statement_cache_size=0)

    # 테이블 상태
    r = await conn.fetchrow("""
        SELECT n_live_tup, n_dead_tup,
               pg_size_pretty(pg_total_relation_size('track_embeddings')) as total_size,
               pg_size_pretty(pg_table_size('track_embeddings')) as table_size,
               last_autovacuum
        FROM pg_stat_user_tables WHERE relname = 'track_embeddings'
    """)
    print(f"live: {r['n_live_tup']}, dead: {r['n_dead_tup']}")
    print(f"테이블 크기: {r['table_size']}, 전체(인덱스포함): {r['total_size']}")
    print(f"last autovacuum: {r['last_autovacuum']}")

    # 실행 중인 쿼리
    rows = await conn.fetch("""
        SELECT pid, state, now() - query_start AS duration, left(query, 80) AS query
        FROM pg_stat_activity
        WHERE state != 'idle' AND pid != pg_backend_pid()
    """)
    print(f"\n활성 쿼리: {len(rows)}개")
    for row in rows:
        print(f"  pid={row['pid']} state={row['state']} duration={row['duration']} query={row['query']}")

    await conn.close()

asyncio.run(main())
