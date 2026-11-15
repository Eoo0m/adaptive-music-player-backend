"""DB 좀비 쿼리 강제 종료"""

import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def main():
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"), statement_cache_size=0)

    # 실행 중인 쿼리 확인
    rows = await conn.fetch("""
        SELECT pid, state, now() - query_start AS duration, left(query, 80) AS query
        FROM pg_stat_activity
        WHERE state = 'active' AND pid != pg_backend_pid()
        ORDER BY duration DESC
    """)

    print(f"실행 중인 쿼리: {len(rows)}개\n")
    for r in rows:
        print(f"  pid={r['pid']}  duration={r['duration']}  query={r['query']}")

    # 전부 kill
    if rows:
        result = await conn.execute("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE state = 'active' AND pid != pg_backend_pid()
        """)
        print(f"\n전부 종료 완료")

    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
