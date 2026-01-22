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

# 환경 변수 로드
load_dotenv()

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

print(f"✅ Playlist CLIP model loaded on {device}")

# FastAPI 앱 생성
app = FastAPI(title="Dynplayer API")

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


def search_playlists_by_keyword(keyword: str, top_k: int = 50):
    """
    키워드로 플레이리스트 검색 (Supabase 벡터 검색 사용)

    Args:
        keyword: 검색 키워드
        top_k: 반환할 상위 플레이리스트 개수

    Returns:
        list of (playlist_id, track_ids, similarity_score) tuples
    """
    # 1. OpenAI 임베딩
    embedding_response = openai_client.embeddings.create(
        model="text-embedding-3-large", input=[keyword]
    )
    keyword_embedding = embedding_response.data[0].embedding

    # 2. CLIP caption projection (텍스트만 프로젝션)
    keyword_tensor = (
        torch.tensor(keyword_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        projected_query = playlist_clip_model.caption_proj(keyword_tensor)  # (1, 512)
        projected_embedding = projected_query.cpu().numpy()[0].tolist()

    # 3. Supabase에서 유사한 플레이리스트 검색
    response = supabase.rpc(
        "match_new_playlist_embeddings",
        {
            "query_embedding": projected_embedding,
            "match_count": top_k,
        },
    ).execute()

    if not response.data:
        return []

    # 4. 결과 반환 (playlist_id, track_ids, similarity)
    results = []
    for item in response.data:
        results.append((item["playlist_id"], item["track_ids"], item["similarity"]))

    return results


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
        print(f"🔍 Search query: {request.query}")

        response = supabase.rpc(
            "search_tracks_by_title", {"query_text": request.query, "match_count": 10}
        ).execute()

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

        print(f"✅ Found {len(results)} tracks")
        return {"results": results}

    except Exception as e:
        print(f"❌ Search error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Search service unavailable: {str(e)}"
        )


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """track_key 기반 유사 음악 추천"""
    if not request.track_key:
        raise HTTPException(status_code=400, detail="Missing track_key")

    try:
        print(f"🎵 Recommend request for track_key: {request.track_key}")

        response = supabase.rpc(
            "match_tracks_by_key",
            {
                "input_track_key": request.track_key,
                "match_count": request.num_recommendations,
            },
        ).execute()

        print(
            f"📊 Supabase response: {response.data is not None}, count: {len(response.data) if response.data else 0}"
        )

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

        print(f"✅ Returning {len(recommendations)} recommendations")

        return {
            "recommendations": recommendations,
            "original_song": {"track_key": request.track_key},
        }

    except Exception as e:
        print(f"❌ Recommend error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Recommendation service unavailable: {str(e)}"
        )


@app.post("/find-spotify-tracks")
async def find_spotify_tracks(request: FindSpotifyTracksRequest):
    """추천 결과를 Spotify 트랙으로 매핑"""
    if not request.access_token or not request.tracks:
        raise HTTPException(status_code=400, detail="Missing access token or tracks")

    try:
        print(f"🔍 Finding Spotify tracks for {len(request.tracks)} recommendations")
        print(
            f"📋 First track sample: {request.tracks[0] if request.tracks else 'empty'}"
        )

        out = []
        # 빈 리스트 체크
        if len(request.tracks) == 0:
            print("⚠️ No tracks to search")
            return {"spotify_tracks": []}

        # 유사도 순서 유지 (상위 10개만)
        top_tracks = request.tracks[:10]

        print("📋 Top 10 tracks to search (in similarity order):")
        for i, track in enumerate(top_tracks):
            track_name = track.get("track") or track.get("track_name")
            artist_name = track.get("artist") or track.get("artist_name")
            similarity = track.get("similarity", "N/A")
            print(f"  {i+1}. {track_name} - {artist_name} (similarity: {similarity})")

        async with httpx.AsyncClient() as client:
            for idx, track in enumerate(top_tracks):
                # track 필드 확인 및 안전하게 접근
                track_name = track.get("track") or track.get("track_name")
                artist_name = track.get("artist") or track.get("artist_name")

                if not track_name or not artist_name:
                    print(f"⚠️ Missing track or artist info: {track}")
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
                        print(f"  ✅ [{idx+1}] Matched: {item['name']} - {item['artists'][0]['name']}")
                        out.append(
                            {
                                **track,
                                "spotify_track": item,
                                "uri": item["uri"],
                                "preview_url": item.get("preview_url"),
                            }
                        )
                else:
                    print(
                        f"⚠️ Spotify search failed for {track_name}: {response.status_code}"
                    )

        print(f"✅ Found {len(out)} Spotify tracks")
        return {"spotify_tracks": out}

    except Exception as e:
        print(f"❌ find-spotify-tracks error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to find Spotify tracks: {str(e)}"
        )


# ============== Keyword Search ==============


@app.post("/search-by-keyword")
async def search_by_keyword(request: KeywordSearchRequest):
    """
    키워드 기반 검색
    1. 키워드와 유사한 플레이리스트를 찾음
    2. 해당 플레이리스트에 포함된 트랙을 가중평균으로 순위 매김
    3. 상위 10개 트랙 반환
    """

    if not request.keyword:
        raise HTTPException(status_code=400, detail="Missing keyword")

    try:
        print(f"🔍 Keyword search: '{request.keyword}'")

        # 1. 플레이리스트 검색 (상위 50개)
        playlist_results = search_playlists_by_keyword(request.keyword, top_k=50)
        print(f"📊 Found {len(playlist_results)} matching playlists")

        if not playlist_results:
            print("⚠️ No matching playlists found")
            return {"results": []}

        # 2. 가중 빈도 기반 트랙 추천
        track_ids = recommend_tracks_by_weighted_frequency(playlist_results, top_k=10)
        print(f"🎵 Recommended {len(track_ids)} tracks")

        if not track_ids:
            print("⚠️ No tracks found in playlists")
            return {"results": []}

        # 3. Supabase에서 트랙 메타데이터 가져오기
        response = (
            supabase.table("new_track_embeddings")
            .select("track_key, title, artist, album, pos_count, cover_image_url")
            .in_("track_key", track_ids)
            .execute()
        )

        if not response.data:
            print("⚠️ No track metadata found")
            return {"results": []}

        # 4. 결과 포맷 변환 (원래 순서 유지)
        track_data_map = {item["track_key"]: item for item in response.data}
        results = []
        for track_id in track_ids:
            if track_id in track_data_map:
                item = track_data_map[track_id]
                results.append(
                    {
                        "track_key": item["track_key"],
                        "track_name": item.get("title"),
                        "artist": item.get("artist"),
                        "album": item.get("album"),
                        "pos_count": item.get("pos_count"),
                        "cover_image_url": item.get("cover_image_url"),
                    }
                )

        print(f"✅ Final selected: {len(results)} tracks")
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
        "main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8889)), reload=True
    )
