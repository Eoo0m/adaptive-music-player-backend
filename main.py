import time
import logging
import sys
import asyncio
from datetime import datetime

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 환경 변수 로드 (다른 모듈보다 먼저)
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# 외부 라이브러리 로그 레벨 조정
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.WARNING)

# 시작 시 강제로 출력
print("=" * 80, flush=True)
print("🚀 DYNPLAYER BACKEND STARTING UP", flush=True)
print("=" * 80, flush=True)
sys.stdout.flush()

# 앱 버전 (배포 확인용)
APP_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
logger.info(f"🚀 Starting Adaptive Music Player Backend - Version: {APP_VERSION}")

# 내부 모듈 import (load_dotenv 이후)
from database import get_db_pool
import database
from utils import openai_client
from models import playlist_clip_model
from routes.search import router as search_router, search_by_keyword
from routes.recommend import router as recommend_router
from routes.log import router as log_router
from routes.auth import router as auth_router
from routes.favorites import router as favorites_router

# FastAPI 앱 생성
app = FastAPI(title="Dynplayer API")


# 백그라운드 Keep-Alive 작업
keep_alive_task = None

async def keep_alive_ping():
    """주기적으로 서버를 깨워있게 하는 백그라운드 작업"""
    await asyncio.sleep(30)  # 워밍업 후 30초 대기

    while True:
        try:
            # 20분마다 DB ping
            await asyncio.sleep(1200)  # 20분 = 1200초

            logger.debug("🏓 DB Keep-alive ping...")

            pool = await get_db_pool()

            # DB 벡터 검색 ping (raw SQL)
            ping_vec = np.random.randn(512).astype(np.float32)
            ping_vec = ping_vec / np.linalg.norm(ping_vec)
            await pool.fetch(
                """
                SELECT track_key, title, artist
                FROM track_embeddings
                WHERE projected_embedding IS NOT NULL
                ORDER BY projected_embedding <=> $1::vector
                LIMIT 10
                """,
                str(ping_vec.tolist())
            )

            # 제목 검색 ping (raw SQL)
            await pool.fetch(
                """
                SELECT id, track_key, title, artist
                FROM track_embeddings
                WHERE lower(title) ILIKE 'test%'
                LIMIT 5
                """
            )

            logger.debug("✅ DB Keep-alive ping successful")

        except asyncio.CancelledError:
            logger.info("🛑 Keep-alive task cancelled")
            break
        except Exception as e:
            logger.warning(f"⚠️  Keep-alive ping failed (non-critical): {str(e)}")


# 서버 시작 시 워밍업
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 콜드 스타트 제거를 위한 워밍업 + Keep-Alive 시작"""
    global keep_alive_task

    logger.info("🔥 Starting warmup sequence...")

    try:
        # 0. asyncpg 연결 풀 초기화
        logger.info("  ⏳ Initializing asyncpg connection pool...")
        pool_start = time.time()
        await get_db_pool()
        logger.info(f"  ✅ asyncpg pool initialized ({time.time() - pool_start:.2f}s)")

        # 1. OpenAI API 워밍업 (small 모델 사용)
        logger.info("  ⏳ Warming up OpenAI API...")
        warmup_start = time.time()
        openai_client.embeddings.create(
            model="text-embedding-3-small",
            input="warmup"
        )
        logger.info(f"  ✅ OpenAI API warmed up ({time.time() - warmup_start:.2f}s)")

        # 2. DB 벡터 검색 워밍업 (raw SQL)
        logger.info("  ⏳ Warming up database (vector search with 5 queries)...")
        warmup_start = time.time()
        pool = await get_db_pool()
        for i in range(5):
            warmup_vec = np.random.randn(512).astype(np.float32)
            warmup_vec = warmup_vec / np.linalg.norm(warmup_vec)
            await pool.fetch(
                """
                SELECT track_key, title, artist, album, playlist_count, cover_image_url,
                       (1 - (projected_embedding <=> $1::vector))::float AS similarity
                FROM track_embeddings
                WHERE projected_embedding IS NOT NULL
                ORDER BY projected_embedding <=> $1::vector
                LIMIT 10
                """,
                str(warmup_vec.tolist())
            )
        logger.info(f"  ✅ Database fully warmed up ({time.time() - warmup_start:.2f}s, 5 queries)")

        # 3. 제목 검색 워밍업 (raw SQL)
        logger.info("  ⏳ Warming up title search (prefix + pg_trgm)...")
        title_warmup_start = time.time()
        warmup_queries = ["love", "sum", "night", "xyz123notfound"]
        for query in warmup_queries:
            await pool.fetch(
                """
                SELECT id, track_key, title, artist, album, playlist_count, cover_image_url,
                       similarity(lower(title || ' ' || artist), lower($1))::float AS similarity
                FROM track_embeddings
                WHERE lower(title || ' ' || artist) % lower($1)
                   OR lower(title) ILIKE $1 || '%'
                   OR lower(artist) ILIKE $1 || '%'
                ORDER BY
                    CASE WHEN lower(title) ILIKE $1 || '%' THEN 0 ELSE 1 END,
                    similarity(lower(title || ' ' || artist), lower($1)) DESC
                LIMIT 10
                """,
                query
            )
        logger.info(f"  ✅ Title search warmed up ({time.time() - title_warmup_start:.2f}s, {len(warmup_queries)} queries)")

        # 4. 키워드 검색 전체 파이프라인 워밍업 (실제 엔드포인트 호출)
        logger.info("  ⏳ Warming up keyword search endpoint...")
        keyword_warmup_start = time.time()
        try:
            # KeywordSearchRequest는 startup보다 나중에 정의되므로,
            # 수동으로 간단한 요청 객체 생성
            class _WarmupReq:
                keyword = "love"

            _ = await search_by_keyword(_WarmupReq())

            logger.info(f"  ✅ Keyword search endpoint warmed up ({time.time() - keyword_warmup_start:.2f}s)")
        except Exception as e:
            logger.warning(f"  ⚠️  Keyword warmup failed (non-critical): {str(e)}")

        # Keep-Alive 백그라운드 작업 시작
        keep_alive_task = asyncio.create_task(keep_alive_ping())

        logger.info("🚀 Full warmup done - server ready!")
        logger.info("🏓 Keep-alive task started (DB ping every 20 minutes)")

    except Exception as e:
        logger.warning(f"⚠️  Startup failed (non-critical): {str(e)}")


# 서버 종료 시 정리
@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 백그라운드 작업 정리"""
    global keep_alive_task

    if keep_alive_task:
        keep_alive_task.cancel()
        try:
            await keep_alive_task
        except asyncio.CancelledError:
            pass
        logger.info("🛑 Keep-alive task stopped")

    if database.db_pool:
        await database.db_pool.close()
        logger.info("🛑 asyncpg pool closed")


# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.debug(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")

    return response

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dynplayer.win",
        "https://www.dynplayer.win",
        "https://api.dynplayer.win",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
    expose_headers=["*"],  # 모든 응답 헤더 노출
)

# Warmup endpoint (유저 세션 시작 시 프론트엔드에서 호출)
@app.post("/warmup")
async def warmup():
    """유저 세션 시작 시 백엔드 워밍업"""
    logger.info("🔥 Warmup requested by frontend")

    # 백그라운드에서 키워드 검색 워밍업 (응답 기다리지 않음)
    async def _run_warmup():
        try:
            class _WarmupReq:
                keyword = "love"
            await search_by_keyword(_WarmupReq())
            logger.info("✅ Warmup completed")
        except Exception as e:
            logger.warning(f"⚠️ Warmup failed: {str(e)}")

    # 백그라운드 태스크로 실행
    asyncio.create_task(_run_warmup())

    # 즉시 응답 반환 (프론트는 기다리지 않음)
    return {"status": "warmup_started"}

# Health check endpoint (배포 확인용)
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Adaptive Music Player API",
        "version": APP_VERSION,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "models_loaded": {
            "playlist_clip": playlist_clip_model is not None
        }
    }


# 라우터 등록
app.include_router(search_router)
app.include_router(recommend_router)
app.include_router(log_router)
app.include_router(auth_router)
app.include_router(favorites_router)


# ============== 서버 실행 ==============
if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8889)), reload=True
    )
