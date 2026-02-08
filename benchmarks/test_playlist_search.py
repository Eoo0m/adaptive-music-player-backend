"""
플레이리스트 검색 성능 테스트
HNSW 인덱스 성능 측정
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# 환경 변수 로드
load_dotenv()

# Supabase 클라이언트
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# OpenAI 클라이언트
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# CLIP 모델 정의
class PlaylistProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
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
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.proj_in(x)
        h = self.activation(h)
        h = self.norm(h)
        residual = h
        h = self.block1(h)
        h = h + residual
        residual = h
        h = self.block2(h)
        h = h + residual
        h = self.proj_out(h)
        return F.normalize(h, dim=-1)


class CaptionPlaylistCLIP(nn.Module):
    def __init__(self, caption_dim, playlist_dim, out_dim=1024, temperature=0.07):
        super().__init__()
        self.caption_proj = PlaylistProjectionMLP(caption_dim, out_dim)
        self.playlist_proj = PlaylistProjectionMLP(playlist_dim, out_dim)
        self.temperature = temperature


# 모델 로드
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
title_dim = 3072
playlist_dim = 256

print("🔄 Loading CLIP model...")
playlist_clip_model = CaptionPlaylistCLIP(
    caption_dim=title_dim, playlist_dim=playlist_dim, out_dim=512
).to(device)
playlist_clip_model.load_state_dict(torch.load("clip_u10_valid_tracks_best.pt", map_location=device))
playlist_clip_model.eval()
print(f"✅ Model loaded on {device}\n")


def search_playlists_with_timing(keyword: str, top_k: int = 50):
    """플레이리스트 검색 with 시간 측정"""

    print(f"🔍 Searching for: '{keyword}'")
    print("=" * 80)

    # 1. OpenAI 임베딩
    t0 = time.time()
    embedding_response = openai_client.embeddings.create(
        model="text-embedding-3-large", input=[keyword]
    )
    keyword_embedding = embedding_response.data[0].embedding
    t1 = time.time()
    print(f"⏱️  OpenAI embedding: {(t1-t0)*1000:.2f}ms")

    # 2. CLIP projection
    t2 = time.time()
    keyword_tensor = (
        torch.tensor(keyword_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        projected_query = playlist_clip_model.caption_proj(keyword_tensor)
        projected_embedding = projected_query.cpu().numpy()[0].tolist()
    t3 = time.time()
    print(f"⏱️  CLIP projection: {(t3-t2)*1000:.2f}ms")

    # 3. Supabase 벡터 검색 (HNSW)
    t4 = time.time()
    response = supabase.rpc(
        "match_playlist_embeddings",
        {
            "query_embedding": projected_embedding,
            "match_count": top_k,
        },
    ).execute()
    t5 = time.time()
    print(f"⏱️  Supabase search (HNSW): {(t5-t4)*1000:.2f}ms")

    # 전체 시간
    total_time = t5 - t0
    print(f"⏱️  Total time: {total_time*1000:.2f}ms")
    print("=" * 80)

    if not response.data:
        print("⚠️  No results found\n")
        return []

    print(f"✅ Found {len(response.data)} playlists\n")

    # 상위 5개 결과 출력
    print("📊 Top 5 Results:")
    print("-" * 80)
    for i, item in enumerate(response.data[:5], 1):
        playlist_id = item.get("playlist_id", "Unknown")
        similarity = item.get("similarity", 0)
        track_count = len(item.get("track_ids", [])) if item.get("track_ids") else 0
        print(f"{i}. Playlist ID: {playlist_id}")
        print(f"   Similarity: {similarity:.4f}")
        print(f"   Tracks: {track_count}")
        print()

    return response.data


def run_benchmark(queries, iterations=3):
    """여러 쿼리로 벤치마크 실행"""
    print("\n" + "=" * 80)
    print("🚀 BENCHMARK TEST")
    print("=" * 80 + "\n")

    all_times = []

    for query in queries:
        query_times = []

        print(f"\n📝 Testing query: '{query}'")
        print("-" * 80)

        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}")

            t_start = time.time()

            # OpenAI 임베딩
            embedding_response = openai_client.embeddings.create(
                model="text-embedding-3-large", input=[query]
            )
            keyword_embedding = embedding_response.data[0].embedding

            # CLIP projection
            keyword_tensor = (
                torch.tensor(keyword_embedding, dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                projected_query = playlist_clip_model.caption_proj(keyword_tensor)
                projected_embedding = projected_query.cpu().numpy()[0].tolist()

            # Supabase 검색 (측정 대상)
            t_db_start = time.time()
            response = supabase.rpc(
                "match_playlist_embeddings",
                {
                    "query_embedding": projected_embedding,
                    "match_count": 50,
                },
            ).execute()
            t_db_end = time.time()

            t_end = time.time()

            db_time = (t_db_end - t_db_start) * 1000
            total_time = (t_end - t_start) * 1000

            query_times.append(db_time)

            print(f"  DB search time: {db_time:.2f}ms")
            print(f"  Total time: {total_time:.2f}ms")
            print(f"  Results: {len(response.data) if response.data else 0}")

        avg_time = sum(query_times) / len(query_times)
        min_time = min(query_times)
        max_time = max(query_times)

        print(f"\n📈 Statistics for '{query}':")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Min: {min_time:.2f}ms")
        print(f"  Max: {max_time:.2f}ms")

        all_times.extend(query_times)

    # 전체 통계
    print("\n" + "=" * 80)
    print("📊 OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total queries: {len(all_times)}")
    print(f"Average DB search time: {sum(all_times)/len(all_times):.2f}ms")
    print(f"Min time: {min(all_times):.2f}ms")
    print(f"Max time: {max(all_times):.2f}ms")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # 단일 검색 테스트
    print("\n" + "=" * 80)
    print("🎯 SINGLE SEARCH TEST")
    print("=" * 80 + "\n")

    search_playlists_with_timing("jazz music", top_k=50)

    # 벤치마크 테스트
    test_queries = [
        "happy upbeat music",
        "relaxing jazz",
        "workout motivation",
        "sad emotional songs",
        "party dance music",
    ]

    run_benchmark(test_queries, iterations=3)
