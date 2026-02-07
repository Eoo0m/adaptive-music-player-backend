from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import time
import os
import base64
import secrets
import httpx
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
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

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


# 모델 로드
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
title_dim = 3072  # OpenAI text-embedding-3-large
playlist_dim = 256  # 플레이리스트 임베딩 차원

# 플레이리스트 CLIP 모델 로드 (텍스트 임베딩 프로젝션용)
playlist_clip_model = CaptionPlaylistCLIP(
    caption_dim=title_dim, playlist_dim=playlist_dim, out_dim=512
).to(device)
playlist_clip_model.load_state_dict(torch.load("clip_u10_valid_tracks_best.pt", map_location=device))
playlist_clip_model.eval()

logger.info(f"✅ Playlist CLIP model loaded on {device}")

# FastAPI 앱 생성
app = FastAPI(title="Dynplayer API")


# 백그라운드 Keep-Alive 작업
keep_alive_task = None

async def keep_alive_ping():
    """주기적으로 서버를 깨워있게 하는 백그라운드 작업"""
    await asyncio.sleep(30)  # 워밍업 후 30초 대기

    while True:
        try:
            # 5분마다 가벼운 DB 쿼리 실행 (캐시 유지)
            await asyncio.sleep(300)  # 5분 = 300초

            logger.debug("🏓 Keep-alive ping...")
            _ = supabase.table("new_playlists").select("playlist_id").limit(1).execute()
            logger.debug("✅ Keep-alive ping successful")

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
        # 1. OpenAI API 워밍업
        logger.info("  ⏳ Warming up OpenAI API...")
        warmup_start = time.time()
        test_embedding = get_openai_embedding_with_retry("warmup")
        logger.info(f"  ✅ OpenAI API warmed up ({time.time() - warmup_start:.2f}s)")

        # 2. CLIP 모델 워밍업
        logger.info("  ⏳ Warming up CLIP projection...")
        warmup_start = time.time()
        test_tensor = torch.tensor(test_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = playlist_clip_model.caption_proj(test_tensor)
        logger.info(f"  ✅ CLIP projection warmed up ({time.time() - warmup_start:.2f}s)")

        # 3. 데이터베이스 워밍업 (간단한 쿼리)
        logger.info("  ⏳ Warming up database connection...")
        warmup_start = time.time()
        _ = supabase.table("new_playlists").select("playlist_id").limit(1).execute()
        logger.info(f"  ✅ Database warmed up ({time.time() - warmup_start:.2f}s)")

        logger.info("🚀 Warmup complete - server ready!")

        # Keep-Alive 백그라운드 작업 시작
        keep_alive_task = asyncio.create_task(keep_alive_ping())
        logger.info("🏓 Keep-alive task started (ping every 5 minutes)")

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
class RefreshTokenRequest(BaseModel):
    refresh_token: str


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


class FindSpotifyTracksRequest(BaseModel):
    tracks: List[dict]
    access_token: str


class RecommendDiverseRequest(BaseModel):
    spotify_track: dict
    access_token: str


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


# 유틸리티 함수
def generate_random_string(length: int = 16) -> str:
    """랜덤 문자열 생성"""
    return secrets.token_urlsafe(length)[:length]


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True
)
def search_supabase_playlists_with_retry(projected_embedding: list, top_k: int):
    """
    Supabase 벡터 검색 (재시도 로직 포함)

    Args:
        projected_embedding: 프로젝션된 임베딩 벡터
        top_k: 반환할 상위 플레이리스트 개수

    Returns:
        Supabase response data
    """
    logger.info(f"🔄 Calling Supabase RPC for vector search (top_k={top_k})")
    response = supabase.rpc(
        "match_playlist_embeddings",
        {
            "query_embedding": projected_embedding,
            "match_count": top_k,
        },
    ).execute()
    logger.info(f"✅ Supabase RPC response received")
    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def search_tracks_by_keyword_fast_with_retry(query_embedding: list, playlist_count: int = 100, track_count: int = 10):
    """
    키워드 기반 트랙 검색 RPC (재시도 로직 포함)

    Args:
        query_embedding: 프로젝션된 쿼리 임베딩
        playlist_count: 검색할 플레이리스트 개수
        track_count: 반환할 트랙 개수

    Returns:
        Supabase response data
    """
    logger.info(f"🔄 Calling Supabase RPC: search_tracks_by_keyword_fast")
    start_time = time.time()
    response = supabase.rpc(
        "search_tracks_by_keyword_fast",
        {
            "query_embedding": query_embedding,
            "playlist_count": playlist_count,
            "track_count": track_count
        }
    ).execute()
    elapsed = time.time() - start_time
    logger.info(f"✅ search_tracks_by_keyword_fast completed ({elapsed:.2f}s)")
    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def search_tracks_by_title_with_retry(query_text: str, match_count: int = 10):
    """
    제목 기반 트랙 검색 RPC (재시도 로직 포함)

    Args:
        query_text: 검색할 제목 텍스트
        match_count: 반환할 트랙 개수

    Returns:
        Supabase response data
    """
    logger.info(f"🔄 Calling Supabase RPC: search_tracks_by_title for '{query_text}'")
    start_time = time.time()
    response = supabase.rpc(
        "search_tracks_by_title",
        {
            "query_text": query_text,
            "match_count": match_count
        }
    ).execute()
    elapsed = time.time() - start_time
    logger.info(f"✅ search_tracks_by_title completed ({elapsed:.2f}s)")
    return response


def search_playlists_by_keyword(keyword: str, top_k: int = 50):
    """
    키워드로 플레이리스트 검색 (Supabase 벡터 검색 사용)

    Args:
        keyword: 검색 키워드
        top_k: 반환할 상위 플레이리스트 개수

    Returns:
        list of (playlist_id, track_ids, similarity_score) tuples
    """
    try:
        # 1. OpenAI 임베딩 (재시도 로직 포함)
        keyword_embedding = get_openai_embedding_with_retry(keyword)

        # 2. CLIP caption projection (텍스트만 프로젝션)
        keyword_tensor = (
            torch.tensor(keyword_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            projected_query = playlist_clip_model.caption_proj(keyword_tensor)  # (1, 512)
            projected_embedding = projected_query.cpu().numpy()[0].tolist()

        # 3. Supabase에서 유사한 플레이리스트 검색 (재시도 로직 포함)
        response = search_supabase_playlists_with_retry(projected_embedding, top_k)

        if not response.data:
            return []

        # 4. 결과 반환 (playlist_id, track_ids, similarity)
        results = []
        for item in response.data:
            results.append((item["playlist_id"], item["track_ids"], item["similarity"]))

        return results

    except (APIError, APIConnectionError, RateLimitError, Timeout) as e:
        logger.error(f"❌ OpenAI API error after retries: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"검색 서비스가 일시적으로 사용 불가합니다. 잠시 후 다시 시도해주세요."
        )
    except Exception as e:
        logger.error(f"❌ Unexpected error in search_playlists_by_keyword: {str(e)}")
        raise


def recommend_tracks_by_weighted_frequency(playlist_results, top_k: int = 10):
    """
    플레이리스트 유사도로 가중평균한 트랙 추천

    Args:
        playlist_results: list of (playlist_id, track_ids, similarity_score) tuples
        top_k: 반환할 트랙 개수

    Returns:
        list of track_key strings
    """
    from collections import defaultdict
    import json

    # 트랙별 가중 점수 계산
    track_scores = defaultdict(float)

    for playlist_id, track_ids_data, similarity_score in playlist_results:
        if track_ids_data:
            # track_ids는 JSON 배열 형태 (문자열 또는 리스트)
            if isinstance(track_ids_data, str):
                try:
                    track_list = json.loads(track_ids_data)
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 "|"로 구분된 문자열로 시도
                    track_list = [
                        t.strip() for t in track_ids_data.split("|") if t.strip()
                    ]
            else:
                track_list = track_ids_data

            # 각 트랙에 유사도 점수 가중치 부여
            for track_id in track_list:
                track_scores[track_id] += similarity_score

    # 상위 k개 트랙 선택
    sorted_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)[
        :top_k
    ]

    return [track_id for track_id, _ in sorted_tracks]


# ============== Routes ==============


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {"message": "dynplayer API"}


@app.get("/health")
async def health():
    """헬스 체크"""
    return "ok"


# ============== Spotify OAuth ==============


@app.get("/login")
async def login():
    """Spotify 로그인 시작"""
    scopes = [
        "streaming",
        "user-read-email",
        "user-read-private",
        "user-library-read",
        "user-library-modify",
        "user-read-playback-state",
        "user-modify-playback-state",
        "playlist-read-private",
        "playlist-read-collaborative",
    ]

    params = {
        "response_type": "code",
        "client_id": os.getenv("SPOTIFY_CLIENT_ID"),
        "scope": " ".join(scopes),
        "redirect_uri": os.getenv("REDIRECT_URI"),
        "state": generate_random_string(16),
    }

    from urllib.parse import urlencode

    auth_url = f"https://accounts.spotify.com/authorize?{urlencode(params)}"
    return RedirectResponse(url=auth_url)


@app.get("/callback")
async def callback(code: Optional[str] = None):
    """Spotify OAuth 콜백"""
    if not code:
        return RedirectResponse(url="/#error=access_denied")

    try:
        # Spotify 토큰 교환
        auth_str = (
            f"{os.getenv('SPOTIFY_CLIENT_ID')}:{os.getenv('SPOTIFY_CLIENT_SECRET')}"
        )
        auth_b64 = base64.b64encode(auth_str.encode()).decode()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://accounts.spotify.com/api/token",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": f"Basic {auth_b64}",
                },
                data={
                    "code": code,
                    "redirect_uri": os.getenv("REDIRECT_URI"),
                    "grant_type": "authorization_code",
                },
            )
            token_data = response.json()

        if token_data.get("access_token"):
            redirect_url = (
                f"https://dynplayer.win/#access_token={token_data['access_token']}"
            )
            if token_data.get("refresh_token"):
                redirect_url += f"&refresh_token={token_data['refresh_token']}"
            return RedirectResponse(url=redirect_url)
        else:
            return RedirectResponse(url="/#error=invalid_token")

    except Exception as e:
        logger.error(f"OAuth token error: {str(e)}")
        return RedirectResponse(url="/#error=server_error")


@app.post("/refresh_token")
async def refresh_token(request: RefreshTokenRequest):
    """토큰 리프레시"""
    if not request.refresh_token:
        raise HTTPException(status_code=400, detail="Missing refresh token")

    try:
        auth_str = (
            f"{os.getenv('SPOTIFY_CLIENT_ID')}:{os.getenv('SPOTIFY_CLIENT_SECRET')}"
        )
        auth_b64 = base64.b64encode(auth_str.encode()).decode()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://accounts.spotify.com/api/token",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Authorization": f"Basic {auth_b64}",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": request.refresh_token,
                },
            )
            return response.json()

    except Exception as e:
        logger.error(f"Refresh token error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to refresh token")


# ============== Search & Recommendation ==============


@app.post("/search-songs")
async def search_songs(request: SearchRequest):
    """제목 기반 검색"""
    if not request.query:
        raise HTTPException(status_code=400, detail="Missing query")

    try:
        response = search_tracks_by_title_with_retry(
            query_text=request.query,
            match_count=10
        )

        if response.data is None:
            raise HTTPException(status_code=500, detail="Search failed")

        # 결과 포맷 변환
        results = []
        for item in response.data:
            results.append({
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "pos_count": item["pos_count"],
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


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """track_key 기반 유사 음악 추천"""
    if not request.track_key:
        raise HTTPException(status_code=400, detail="Missing track_key")

    try:
        response = supabase.rpc(
            "match_tracks_by_key",
            {
                "input_track_key": request.track_key,
                "match_count": request.num_recommendations,
            },
        ).execute()

        if response.data is None:
            raise HTTPException(status_code=500, detail="Recommendation failed")

        # 결과 포맷 변환
        recommendations = []
        for item in response.data:
            recommendations.append({
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "pos_count": item["pos_count"],
                "similarity": item.get("similarity", 0),
                "cover_image_url": item.get("cover_image_url"),
            })

        logger.info(f"Recommend: {len(recommendations)} tracks for '{request.track_key}'")

        return {
            "recommendations": recommendations,
            "original_song": {"track_key": request.track_key},
        }

    except Exception as e:
        logger.error(f"Recommend error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Recommendation service unavailable: {str(e)}"
        )


@app.post("/recommend-average")
async def recommend_average(request: RecommendAverageRequest):
    """여러 track_key들의 평균 임베딩 기반 유사 음악 추천"""
    if not request.track_keys or len(request.track_keys) == 0:
        raise HTTPException(status_code=400, detail="Missing track_keys")

    try:
        logger.info(f"🎯 Averaging {len(request.track_keys)} track embeddings")

        # 각 track_key의 임베딩 가져오기
        embeddings = []
        for track_key in request.track_keys:
            response = (
                supabase.table("new_track_embeddings")
                .select("embedding")
                .eq("track_key", track_key)
                .limit(1)
                .execute()
            )

            if response.data and len(response.data) > 0:
                embedding = response.data[0].get("embedding")
                if embedding:
                    # 문자열로 저장된 경우 JSON 파싱
                    if isinstance(embedding, str):
                        import json
                        embedding = json.loads(embedding)

                    # 임베딩을 float 배열로 명시적 변환
                    embedding_array = np.array(embedding, dtype=np.float32)
                    embeddings.append(embedding_array)
                    logger.debug(f"Loaded embedding for {track_key}: shape={embedding_array.shape}, dtype={embedding_array.dtype}")

        if len(embeddings) == 0:
            raise HTTPException(
                status_code=404, detail="No embeddings found for provided track_keys"
            )

        # 평균 임베딩 계산
        embeddings_stack = np.stack(embeddings)  # 명시적으로 stack
        avg_embedding = np.mean(embeddings_stack, axis=0)

        # 정규화
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm

        logger.info(f"✅ Computed average embedding from {len(embeddings)} tracks (shape: {avg_embedding.shape}, dtype: {avg_embedding.dtype})")

        # 평균 임베딩으로 유사 곡 검색
        response = supabase.rpc(
            "match_tracks_by_embedding",
            {
                "query_embedding": avg_embedding.tolist(),
                "match_count": request.num_recommendations,
            },
        ).execute()

        if response.data is None:
            raise HTTPException(
                status_code=500, detail="Average-based recommendation failed"
            )

        # 결과 포맷 변환
        recommendations = [
            {
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "pos_count": item["pos_count"],
                "similarity": item.get("similarity", 0),
                "cover_image_url": item.get("cover_image_url"),
            }
            for item in response.data
        ]

        logger.info(f"Average-based recommend: {len(recommendations)} tracks from {len(embeddings)} averaged tracks")

        return {
            "recommendations": recommendations,
            "num_averaged_tracks": len(embeddings),
        }

    except Exception as e:
        logger.error(f"Recommend average error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Average-based recommendation failed: {str(e)}"
        )


@app.post("/find-spotify-tracks")
async def find_spotify_tracks(request: FindSpotifyTracksRequest):
    """추천 결과를 Spotify 트랙으로 매핑"""
    if not request.access_token or not request.tracks:
        raise HTTPException(status_code=400, detail="Missing access token or tracks")

    try:
        out = []
        if len(request.tracks) == 0:
            return {"spotify_tracks": []}

        # 유사도 순서 유지 (상위 10개만)
        top_tracks = request.tracks[:10]

        async with httpx.AsyncClient() as client:
            for track in top_tracks:
                track_name = track.get("track") or track.get("track_name")
                artist_name = track.get("artist") or track.get("artist_name")

                if not track_name or not artist_name:
                    continue

                q = f'track:"{track_name}" artist:"{artist_name}"'
                response = await client.get(
                    f"https://api.spotify.com/v1/search?q={q}&type=track&limit=1",
                    headers={"Authorization": f"Bearer {request.access_token}"},
                )

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("tracks", {}).get("items", [])
                    if items and len(items) > 0:
                        item = items[0]
                        out.append({
                            **track,
                            "spotify_track": item,
                            "uri": item["uri"],
                            "preview_url": item.get("preview_url"),
                        })

        logger.info(f"Find Spotify tracks: {len(out)}/{len(top_tracks)} matched")
        return {"spotify_tracks": out}

    except Exception as e:
        logger.error(f"Find Spotify tracks error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to find Spotify tracks: {str(e)}"
        )


# ============== Keyword Search ==============


@app.post("/search-by-keyword")
async def search_by_keyword(request: KeywordSearchRequest):
    """키워드 기반 검색 (DB에서 모두 처리)"""
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

        # 2. CLIP projection
        t_proj = time.time()
        keyword_tensor = torch.tensor(keyword_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            projected_query = playlist_clip_model.caption_proj(keyword_tensor)
            projected_embedding = projected_query.cpu().numpy()[0].tolist()
        timings["projection"] = time.time() - t_proj

        # 3. DB에서 한 번에 처리 (플레이리스트 검색 + 트랙 가중합 + 메타데이터 조회)
        t_db = time.time()
        try:
            response = search_tracks_by_keyword_fast_with_retry(
                query_embedding=projected_embedding,
                playlist_count=50,  # 100 → 50으로 줄여서 속도 개선
                track_count=10
            )
            timings["db"] = time.time() - t_db
            logger.info(f"DB processing took {timings['db']:.2f}s")

            if not response.data:
                logger.warning("No results from search_tracks_by_keyword_fast")
                return {"results": []}

            # 4. 결과 포맷 변환
            results = [
                {
                    "track_key": item["track_key"],
                    "track_name": item["title"],
                    "artist": item["artist"],
                    "album": item["album"],
                    "pos_count": item["pos_count"],
                    "cover_image_url": item["cover_image_url"],
                }
                for item in response.data
            ]

            logger.info(f"Keyword search completed: {len(results)} tracks")
            return {"results": results, "timings": timings}

        except Exception as rpc_error:
            logger.error(f"RPC function failed: {str(rpc_error)}, falling back to old method")

            # Fallback: 기존 방식 사용
            playlist_results = search_playlists_by_keyword(request.keyword, top_k=100)
            if not playlist_results:
                return {"results": []}

            track_ids = recommend_tracks_by_weighted_frequency(playlist_results, top_k=10)
            if not track_ids:
                return {"results": []}

            response = (
                supabase.table("new_track_embeddings")
                .select("track_key, title, artist, album, pos_count, cover_image_url")
                .in_("track_key", track_ids)
                .execute()
            )

            if not response.data:
                return {"results": []}

            track_data_map = {item["track_key"]: item for item in response.data}
            results = []

            for track_id in track_ids:
                if track_id in track_data_map:
                    item = track_data_map[track_id]
                    results.append({
                        "track_key": item["track_key"],
                        "track_name": item.get("title"),
                        "artist": item.get("artist"),
                        "album": item.get("album"),
                        "pos_count": item.get("pos_count"),
                        "cover_image_url": item.get("cover_image_url"),
                    })

            logger.info(f"Keyword search completed (fallback): {len(results)} tracks")
            return {"results": results}

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
