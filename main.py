from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import time
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import sys
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, APIConnectionError, RateLimitError, Timeout
import asyncio
import random
import asyncpg

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

# 환경 변수 로드
load_dotenv()

# 앱 버전 (배포 확인용)
APP_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")
logger.info(f"🚀 Starting Adaptive Music Player Backend - Version: {APP_VERSION}")

# Supabase 클라이언트
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

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
            init=lambda conn: conn.execute("SET work_mem = '256MB'"),
        )
        logger.info("✅ asyncpg connection pool created")
    return db_pool

# OpenAI 클라이언트
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# CaptionPlaylistCLIP 모델 정의 (플레이리스트 검색용)
class PlaylistProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()

        # First projection to hidden dimension
        self.proj_in = nn.Linear(in_dim, hidden_dim)

        # Residual blocks
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

        # Final projection to output dimension
        self.proj_out = nn.Linear(hidden_dim, out_dim)

        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Project to hidden dimension
        h = self.proj_in(x)
        h = self.activation(h)
        h = self.norm(h)

        # Residual block 1
        residual = h
        h = self.block1(h)
        h = h + residual

        # Residual block 2
        residual = h
        h = self.block2(h)
        h = h + residual

        # Project to output dimension
        h = self.proj_out(h)

        # L2 normalize
        return F.normalize(h, dim=-1)


class CaptionPlaylistCLIP(nn.Module):
    def __init__(self, caption_dim, playlist_dim, out_dim=1024, temperature=0.07):
        super().__init__()
        self.caption_proj = PlaylistProjectionMLP(caption_dim, out_dim)
        self.playlist_proj = PlaylistProjectionMLP(playlist_dim, out_dim)
        self.temperature = temperature


# Two-Tower 모델 정의 (세션 기반 추천용)
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class UserTower(nn.Module):
    """User Tower: Encode sequence of item embeddings using a small Transformer."""

    def __init__(self, item_dim, out_dim, hidden_dim=128, n_heads=4, n_layers=2, max_seq_len=16):
        super().__init__()
        self.input_proj = nn.Linear(item_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=mask)
        cls_output = x[:, 0, :]
        out = self.output_proj(cls_output)
        out = F.normalize(out, p=2, dim=-1)
        return out


class ItemTower(nn.Module):
    """Item Tower: Project item embedding using MLP."""

    def __init__(self, item_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        out = self.mlp(x)
        out = F.normalize(out, p=2, dim=-1)
        return out


class TwoTowerModel(nn.Module):
    """Two-Tower Model for session-based recommendation."""

    def __init__(self, item_dim, out_dim=128, hidden_dim=128, n_heads=4, n_layers=2, temperature=0.07):
        super().__init__()
        self.user_tower = UserTower(
            item_dim=item_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.item_tower = ItemTower(
            item_dim=item_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim * 2,
        )
        self.temperature = temperature
        self.out_dim = out_dim

    def encode_user(self, user_seq, user_mask=None):
        """Encode user sequence to embedding."""
        return self.user_tower(user_seq, user_mask)

    def encode_item(self, items):
        """Encode items to embeddings."""
        return self.item_tower(items)


# 모델 로드
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
title_dim = 3072  # OpenAI text-embedding-3-large
playlist_dim = 64  # SimGCL 트랙 임베딩 차원

# CLIP 모델 로드 (텍스트 → 트랙 임베딩 프로젝션용)
playlist_clip_model = CaptionPlaylistCLIP(
    caption_dim=title_dim, playlist_dim=playlist_dim, out_dim=512
).to(device)
playlist_clip_model.load_state_dict(torch.load("clip_simgcl.pt", map_location=device))
playlist_clip_model.eval()

logger.info(f"✅ CLIP model (clip_simgcl.pt) loaded on {device}")

# Two-Tower 모델 로드 (세션 기반 추천용)
two_tower_model = TwoTowerModel(
    item_dim=64,  # SimGCL 트랙 임베딩 차원
    out_dim=128,
    hidden_dim=128,
    n_heads=4,
    n_layers=2,
).to(device)
two_tower_model.load_state_dict(torch.load("two_tower_best.pt", map_location=device))
two_tower_model.eval()

logger.info(f"✅ Two-Tower model (two_tower_best.pt) loaded on {device}")

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
    global keep_alive_task, db_pool

    if keep_alive_task:
        keep_alive_task.cancel()
        try:
            await keep_alive_task
        except asyncio.CancelledError:
            pass
        logger.info("🛑 Keep-alive task stopped")

    if db_pool:
        await db_pool.close()
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


# Pydantic 모델
class SearchRequest(BaseModel):
    query: str


class KeywordSearchRequest(BaseModel):
    keyword: str
    top_k: Optional[int] = 200


class RecommendRequest(BaseModel):
    track_key: str
    num_recommendations: Optional[int] = 30


class RecommendAverageRequest(BaseModel):
    track_keys: List[str]
    num_recommendations: Optional[int] = 5




class ListeningLogRequest(BaseModel):
    track_name: str
    artist_name: str
    album_name: Optional[str] = None
    spotify_uri: Optional[str] = None
    spotify_track_id: Optional[str] = None
    duration_ms: Optional[int] = None
    played_duration_ms: Optional[int] = None
    completion_percentage: Optional[float] = None
    recommendation_mode: Optional[str] = None
    similarity_score: Optional[float] = None
    session_id: Optional[str] = None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((APIError, APIConnectionError, RateLimitError, Timeout)),
    reraise=True
)
def get_openai_embedding_with_retry(keyword: str):
    """
    OpenAI 임베딩 생성 (재시도 로직 포함)

    Args:
        keyword: 검색 키워드

    Returns:
        embedding vector
    """
    try:
        logger.info(f"Calling OpenAI API for keyword: '{keyword}'")
        start_time = time.time()

        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=[keyword],
            timeout=10.0  # 30s → 10s로 줄여서 빠르게 실패하고 재시도
        )

        elapsed = time.time() - start_time
        logger.info(f"OpenAI API success ({elapsed:.2f}s)")
        return embedding_response.data[0].embedding

    except Exception as e:
        logger.warning(f"OpenAI API attempt failed: {type(e).__name__}: {str(e)}")
        raise




async def search_tracks_by_title_raw(query_text: str, match_count: int = 10):
    """제목 기반 트랙 검색 (raw SQL)"""
    logger.info(f"🔍 search_tracks_by_title: '{query_text}'")
    start_time = time.time()

    pool = await get_db_pool()
    rows = await pool.fetch(
        """
        SELECT
            id,
            track_key::text,
            title::text,
            artist::text,
            album::text,
            playlist_count,
            cover_image_url::text,
            similarity(lower(title || ' ' || artist), lower($1))::float AS similarity
        FROM track_embeddings
        WHERE lower(title || ' ' || artist) % lower($1)
           OR lower(title) ILIKE $1 || '%'
           OR lower(artist) ILIKE $1 || '%'
        ORDER BY
            CASE WHEN lower(title) ILIKE $1 || '%' THEN 0 ELSE 1 END,
            similarity(lower(title || ' ' || artist), lower($1)) DESC
        LIMIT $2
        """,
        query_text,
        match_count
    )

    elapsed = time.time() - start_time
    logger.info(f"✅ search_tracks_by_title: {len(rows)} tracks ({elapsed:.2f}s)")
    return [dict(row) for row in rows]


# ============== Search & Recommendation ==============


@app.post("/search-songs")
async def search_songs(request: SearchRequest):
    """제목 기반 검색"""
    if not request.query:
        raise HTTPException(status_code=400, detail="Missing query")

    try:
        rows = await search_tracks_by_title_raw(
            query_text=request.query,
            match_count=10
        )

        # 결과 포맷 변환
        results = []
        for item in rows:
            results.append({
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "playlist_count": item["playlist_count"],
                "similarity": item.get("similarity", 0),
                "cover_image_url": item.get("cover_image_url"),
            })

        logger.info(f"Search: found {len(results)} tracks for '{request.query}'")
        return {"results": results}

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Search service unavailable: {str(e)}"
        )


@app.post("/find-similar-tracks")
async def find_similar_tracks(request: RecommendRequest):
    """track_key 기반 유사 음악 추천 (검색 결과 클릭 시 사용)"""
    if not request.track_key:
        raise HTTPException(status_code=400, detail="Missing track_key")

    try:
        pool = await get_db_pool()

        # 50곡 가져오기 (raw SQL)
        rows = await pool.fetch(
            """
            WITH query_track AS (
                SELECT embedding
                FROM track_embeddings
                WHERE track_key = $1
                LIMIT 1
            )
            SELECT
                t.id,
                t.track_key::text,
                t.title::text,
                t.artist::text,
                t.album::text,
                t.playlist_count,
                t.cover_image_url::text,
                (1 - (t.embedding <=> q.embedding))::float AS similarity
            FROM track_embeddings t, query_track q
            WHERE t.track_key != $1
              AND t.embedding IS NOT NULL
            ORDER BY t.embedding <=> q.embedding
            LIMIT $2
            """,
            request.track_key,
            50
        )

        if not rows:
            raise HTTPException(status_code=500, detail="Recommendation failed")

        # 결과 포맷 변환
        all_recommendations = []
        for item in rows:
            all_recommendations.append({
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "playlist_count": item["playlist_count"],
                "similarity": item.get("similarity", 0),
                "cover_image_url": item.get("cover_image_url"),
            })

        # 랜덤하게 선택
        selected_count = min(request.num_recommendations, len(all_recommendations))
        recommendations = random.sample(all_recommendations, selected_count) if len(all_recommendations) > selected_count else all_recommendations

        # 로그에 선택된 곡 정보 출력
        selected_titles = [f"{r['track']} - {r['artist']}" for r in recommendations]
        logger.info(f"Recommend: {len(recommendations)} tracks selected from {len(all_recommendations)} for '{request.track_key}'")
        logger.info(f"Selected tracks: {', '.join(selected_titles[:5])}{'...' if len(selected_titles) > 5 else ''}")

        return {
            "recommendations": recommendations,
            "original_song": {"track_key": request.track_key},
        }

    except Exception as e:
        logger.error(f"Recommend error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Recommendation service unavailable: {str(e)}"
        )


@app.post("/recommend")
async def recommend(request: RecommendAverageRequest):
    """세션 기반 추천 (Two-Tower 모델 사용)"""
    if not request.track_keys or len(request.track_keys) == 0:
        raise HTTPException(status_code=400, detail="Missing track_keys")

    try:
        logger.info(f"🎯 Two-Tower recommend with {len(request.track_keys)} session tracks")

        pool = await get_db_pool()

        # 1. 모든 track_key의 임베딩을 한 번에 가져오기 (raw SQL)
        rows = await pool.fetch(
            """
            SELECT track_key, embedding
            FROM track_embeddings
            WHERE track_key = ANY($1)
            """,
            request.track_keys
        )

        # track_key 순서대로 임베딩 정렬
        import json
        embedding_map = {}
        for row in rows:
            embedding = row.get("embedding")
            if embedding:
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                embedding_map[row["track_key"]] = np.array(embedding, dtype=np.float32)

        # 요청 순서대로 임베딩 리스트 생성
        embeddings = [embedding_map[tk] for tk in request.track_keys if tk in embedding_map]

        if len(embeddings) == 0:
            raise HTTPException(
                status_code=404, detail="No embeddings found for provided track_keys"
            )

        logger.info(f"📦 Fetched {len(embeddings)} embeddings in single batch query")

        # 2. User Tower로 세션 임베딩 생성 (128차원)
        user_seq = torch.tensor(np.stack(embeddings), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            user_embedding = two_tower_model.encode_user(user_seq)  # (1, 128)

        user_embedding_list = user_embedding.squeeze(0).cpu().numpy().tolist()
        logger.info(f"✅ User embedding computed from {len(embeddings)} tracks")

        # 3. DB에서 itemtower_embedding과 직접 비교하여 추천 (raw SQL)
        t_db = time.time()
        rows = await pool.fetch(
            """
            SELECT
                t.id,
                t.track_key::text,
                t.title::text,
                t.artist::text,
                t.album::text,
                t.playlist_count,
                t.cover_image_url::text,
                (1 - (t.itemtower_embedding <=> $1::vector))::float AS similarity
            FROM track_embeddings t
            WHERE t.itemtower_embedding IS NOT NULL
              AND t.track_key::text != ALL($2)
            ORDER BY t.itemtower_embedding <=> $1::vector
            LIMIT $3
            """,
            str(user_embedding_list),
            request.track_keys,
            request.num_recommendations
        )
        db_time = time.time() - t_db
        logger.info(f"📊 /recommend DB query time: {db_time:.3f}s")

        if not rows:
            raise HTTPException(status_code=500, detail="No recommendations found")

        # 4. 결과 포맷 변환
        recommendations = []
        for item in rows:
            recommendations.append({
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "playlist_count": item["playlist_count"],
                "similarity": item.get("similarity", 0),
                "cover_image_url": item.get("cover_image_url"),
            })

        logger.info(f"Two-Tower recommend: {len(recommendations)} tracks from {len(embeddings)} session tracks")

        return {
            "recommendations": recommendations,
            "num_session_tracks": len(embeddings),
        }

    except Exception as e:
        logger.error(f"Two-Tower recommend error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Session-based recommendation failed: {str(e)}"
        )




# ============== Keyword Search ==============


@app.post("/search-by-keyword")
async def search_by_keyword(request: KeywordSearchRequest):
    """키워드 기반 검색 (트랙 임베딩 직접 검색)"""
    if not request.keyword:
        raise HTTPException(status_code=400, detail="Missing keyword")

    try:
        logger.info(f"Keyword search: '{request.keyword}'")

        # 타이밍 측정
        timings = {}

        # 1. OpenAI 임베딩
        t_openai = time.time()
        keyword_embedding = get_openai_embedding_with_retry(request.keyword)
        timings["openai"] = time.time() - t_openai

        # 2. CLIP projection (텍스트 → 512차원 트랙 임베딩 공간)
        t_proj = time.time()
        keyword_tensor = torch.tensor(keyword_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            projected_query = playlist_clip_model.caption_proj(keyword_tensor)
            projected_embedding = projected_query.cpu().numpy()[0].tolist()
        timings["projection"] = time.time() - t_proj

        # 3. 트랙 임베딩 직접 검색 (raw SQL)
        t_db = time.time()
        pool = await get_db_pool()
        rows = await pool.fetch(
            """
            SELECT
                t.track_key::text,
                t.title::text,
                t.artist::text,
                t.album::text,
                t.playlist_count,
                t.cover_image_url::text,
                (1 - (t.projected_embedding <=> $1::vector))::float AS similarity
            FROM track_embeddings t
            WHERE t.projected_embedding IS NOT NULL
            ORDER BY t.projected_embedding <=> $1::vector
            LIMIT 10
            """,
            str(projected_embedding)
        )
        timings["db"] = time.time() - t_db

        if not rows:
            logger.warning("No results from projected_embedding search")
            return {"results": []}

        # 4. 결과 포맷 변환
        results = [
            {
                "track_key": item["track_key"],
                "track_name": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "playlist_count": item["playlist_count"],
                "cover_image_url": item["cover_image_url"],
                "similarity": item.get("similarity", 0),
            }
            for item in rows
        ]

        logger.info(f"Keyword search completed: {len(results)} tracks in {sum(timings.values()):.2f}s")
        return {"results": results, "timings": timings}

    except Exception as e:
        logger.error(f"Keyword search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {str(e)}")


# ============== Listening Log ==============


@app.post("/log-listening")
async def log_listening(request: ListeningLogRequest):
    """듣는 기록 저장"""
    if not request.track_name or not request.artist_name:
        raise HTTPException(
            status_code=400, detail="Missing required fields: track_name, artist_name"
        )

    try:
        log_data = {
            "track_name": request.track_name,
            "artist_name": request.artist_name,
            "album_name": request.album_name,
            "spotify_uri": request.spotify_uri,
            "spotify_track_id": request.spotify_track_id,
            "session_id": request.session_id,
        }

        # Optional 필드는 값이 있을 때만 추가
        if request.duration_ms is not None:
            log_data["duration_ms"] = request.duration_ms
        if request.played_duration_ms is not None:
            log_data["played_duration_ms"] = request.played_duration_ms
        if request.completion_percentage is not None:
            log_data["completion_percentage"] = request.completion_percentage
        if request.recommendation_mode is not None:
            log_data["recommendation_mode"] = request.recommendation_mode
        if request.similarity_score is not None:
            log_data["similarity_score"] = request.similarity_score

        logger.info(f"Logging track: {request.track_name} by {request.artist_name}")

        response = supabase.table("listening_logs").insert(log_data).execute()

        if response.data is None:
            raise HTTPException(status_code=500, detail="Failed to log listening data")

        logger.info(f"Successfully logged track: {request.track_name}")
        return {"success": True, "data": response.data}

    except Exception as e:
        logger.error(f"Log listening error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============== 서버 실행 ==============
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8889)), reload=True
    )
