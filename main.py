from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
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

# 환경 변수 로드
load_dotenv()

# Supabase 클라이언트
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# OpenAI 클라이언트
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# CLIP 모델 정의
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim=512, hidden_dim=1024, heads=4):
        super().__init__()
        self.heads = heads
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, out_dim),
                )
                for _ in range(heads)
            ]
        )

    def forward(self, x):
        outs = []
        for proj in self.projs:
            h = proj(x)
            h = F.normalize(h, dim=-1)
            outs.append(h)
        h_final = torch.stack(outs, dim=0).mean(dim=0)
        return F.normalize(h_final, dim=-1)


class TitleTrackCLIP(nn.Module):
    def __init__(self, title_dim, track_dim, out_dim=512):
        super().__init__()
        self.title_proj = ProjectionMLP(title_dim, out_dim)
        self.track_proj = ProjectionMLP(track_dim, out_dim)


# 모델 로드
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
title_dim = 3072  # OpenAI text-embedding-3-large
track_dim = 256  # 트랙 임베딩 차원

# 모델 생성 및 로드
clip_model = TitleTrackCLIP(title_dim, track_dim, out_dim=512).to(device)
clip_model.load_state_dict(
    torch.load("title_track_clip_twostage.pt", map_location=device)
)
clip_model.eval()

print(f"✅ CLIP model loaded on {device}")

# FastAPI 앱 생성
app = FastAPI(title="Dynplayer API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dynplayer.win",
        "https://www.dynplayer.win",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


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
        print(f"OAuth token error: {e}")
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
        print(f"Refresh token error: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh token")


# ============== Search & Recommendation ==============


@app.post("/search-songs")
async def search_songs(request: SearchRequest):
    """제목 기반 검색"""
    if not request.query:
        raise HTTPException(status_code=400, detail="Missing query")

    try:
        response = supabase.rpc(
            "search_tracks_by_title", {"query_text": request.query, "match_count": 10}
        ).execute()

        if response.data is None:
            raise HTTPException(status_code=500, detail="Search failed")

        # 결과 포맷 변환
        results = [
            {
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "pos_count": item["pos_count"],
                "similarity": item.get("similarity", 0),
            }
            for item in response.data
        ]

        return {"results": results}

    except Exception as e:
        print(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search service unavailable")


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
        recommendations = [
            {
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "pos_count": item["pos_count"],
                "similarity": item.get("similarity", 0),
            }
            for item in response.data
        ]

        return {
            "recommendations": recommendations,
            "original_song": {"track_key": request.track_key},
        }

    except Exception as e:
        print(f"Recommend error: {e}")
        raise HTTPException(
            status_code=500, detail="Recommendation service unavailable"
        )


@app.post("/find-spotify-tracks")
async def find_spotify_tracks(request: FindSpotifyTracksRequest):
    """추천 결과를 Spotify 트랙으로 매핑"""
    if not request.access_token or not request.tracks:
        raise HTTPException(status_code=400, detail="Missing access token or tracks")

    try:
        import random

        out = []
        shuffled = random.sample(request.tracks, min(len(request.tracks), 10))

        async with httpx.AsyncClient() as client:
            for track in shuffled:
                q = f'track:"{track["track"]}" artist:"{track["artist"]}"'
                response = await client.get(
                    f"https://api.spotify.com/v1/search?q={q}&type=track&limit=1",
                    headers={"Authorization": f"Bearer {request.access_token}"},
                )

                if response.status_code == 200:
                    data = response.json()
                    item = data.get("tracks", {}).get("items", [None])[0]
                    if item:
                        out.append(
                            {
                                **track,
                                "spotify_track": item,
                                "uri": item["uri"],
                                "preview_url": item.get("preview_url"),
                            }
                        )

        return {"spotify_tracks": out}

    except Exception as e:
        print(f"find-spotify-tracks error: {e}")
        raise HTTPException(status_code=500, detail="Failed to find Spotify tracks")


# ============== Keyword Search ==============


@app.post("/search-by-keyword")
async def search_by_keyword(request: KeywordSearchRequest):
    """키워드 기반 검색 → 200개 벡터 받아서 클러스터링 후 pos_count 높은 10개 반환"""

    if not request.keyword:
        raise HTTPException(status_code=400, detail="Missing keyword")

    try:
        print(f"🔍 Keyword search: '{request.keyword}'")

        # 1) OpenAI 임베딩
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-large", input=[request.keyword]
        )
        keyword_embedding = embedding_response.data[0].embedding

        # 2) CLIP title projection
        keyword_tensor = (
            torch.tensor(keyword_embedding, dtype=torch.float32).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            projected_embedding = clip_model.title_proj(keyword_tensor).cpu().numpy()[0]

        # 3) Supabase에서 top 200개 받아오기
        response = supabase.rpc(
            "match_keyword_embeddings",
            {
                "query_embedding": projected_embedding.tolist(),
                "match_count": request.top_k,
            },
        ).execute()

        if not response.data:
            print("⚠️ No data from Supabase")
            return {"results": []}

        # =============== 4) 클러스터링 준비 ===============
        import numpy as np
        from sklearn.cluster import KMeans
        import json

        # 임베딩 + 메타데이터 분리
        all_items = response.data

        # Supabase에서 embedding도 반환되도록 함수 수정돼 있어야 함
        # embedding이 문자열로 저장되어 있을 경우를 대비해 파싱
        print(f"📊 Received {len(all_items)} items from Supabase")

        embeddings = []
        for item in all_items:
            emb = item["embedding"]
            if isinstance(emb, str):
                # 문자열인 경우 JSON 파싱
                emb = json.loads(emb)
            embeddings.append(emb)

        embeddings = np.array(embeddings)
        print(f"✅ Parsed {len(embeddings)} embeddings, shape: {embeddings.shape}")

        pos_counts = [item.get("pos_count", 0) for item in all_items]

        # =============== 5) KMeans 클러스터링 ===============
        K = 10
        kmeans = KMeans(n_clusters=K, n_init="auto")
        labels = kmeans.fit_predict(embeddings)

        # =============== 6) 각 클러스터에서 pos_count TOP 1씩 뽑기 ===============
        cluster_selected = []

        for cluster_id in range(K):
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not cluster_indices:
                continue

            # pos_count 높은 순 정렬
            sorted_cluster = sorted(
                cluster_indices, key=lambda idx: pos_counts[idx], reverse=True
            )

            best_idx = sorted_cluster[0]
            cluster_selected.append(all_items[best_idx])

        # 만약 10개보다 적으면 pos_count 상위곡으로 보충
        if len(cluster_selected) < 10:
            remaining = sorted(
                all_items, key=lambda x: x.get("pos_count", 0), reverse=True
            )

            for item in remaining:
                if len(cluster_selected) >= 10:
                    break
                if item not in cluster_selected:
                    cluster_selected.append(item)

        # =============== 7) 반환 포맷 변환 ===============
        results = []
        for item in cluster_selected[:10]:
            results.append(
                {
                    "track_key": item["track_key"],
                    "track_name": item.get("title"),
                    "pos_count": item.get("pos_count"),
                    "similarity": item.get("similarity", 0),
                }
            )

        print(f"✅ Final selected: {len(results)} tracks")
        print(f"📦 Response: {results}")
        return {"results": results}

    except Exception as e:
        print(f"Keyword search error: {e}")
        import traceback

        traceback.print_exc()
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
        response = (
            supabase.table("listening_logs")
            .insert(
                {
                    "track_name": request.track_name,
                    "artist_name": request.artist_name,
                    "album_name": request.album_name,
                    "spotify_uri": request.spotify_uri,
                    "spotify_track_id": request.spotify_track_id,
                    "duration_ms": request.duration_ms,
                    "played_duration_ms": request.played_duration_ms,
                    "completion_percentage": request.completion_percentage,
                    "recommendation_mode": request.recommendation_mode,
                    "similarity_score": request.similarity_score,
                    "session_id": request.session_id,
                }
            )
            .execute()
        )

        if response.data is None:
            raise HTTPException(status_code=500, detail="Failed to log listening data")

        return {"success": True, "data": response.data}

    except Exception as e:
        print(f"log-listening error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============== 서버 실행 ==============
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host="127.0.0.1", port=int(os.getenv("PORT", 8889)), reload=True
    )
