import os
import logging
import asyncpg

logger = logging.getLogger(__name__)

# PostgreSQL 직접 연결 (asyncpg) - Supabase Edge 타임아웃 우회
db_pool: asyncpg.Pool = None

async def get_db_pool():
    """asyncpg connection pool 가져오기"""
    global db_pool
    if db_pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL 환경 변수가 설정되지 않았습니다")
        db_pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
            statement_cache_size=0,  # PgBouncer transaction mode 호환
            init=lambda conn: conn.execute("SET work_mem = '256MB'"),
        )
        logger.info("✅ asyncpg connection pool created")
    return db_pool
