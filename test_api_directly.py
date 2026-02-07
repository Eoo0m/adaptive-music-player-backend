"""
실제 백엔드 API를 직접 호출해서 결과 확인
"""
import requests
import json

API_URL = "https://api.dynplayer.win/search-by-keyword"

print("=" * 80)
print("🔍 백엔드 API 직접 테스트")
print("=" * 80)

# 테스트 쿼리
test_queries = ["happy music", "sad songs", "workout"]

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"📝 쿼리: '{query}'")
    print(f"{'='*80}")

    try:
        response = requests.post(
            API_URL,
            json={"keyword": query, "top_k": 50},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            # 결과 확인
            results = data.get("results", [])
            debug = data.get("debug", {})

            print(f"\n✅ 성공! (HTTP 200)")
            print(f"   반환된 트랙 수: {len(results)}")

            if debug:
                print(f"\n📊 디버그 정보:")
                print(f"   검색된 플레이리스트: {debug.get('playlists_found', 'N/A')}")
                print(f"   추천된 트랙: {debug.get('tracks_recommended', 'N/A')}")

                top_playlists = debug.get('top_playlists', [])
                if top_playlists:
                    print(f"\n   📋 상위 5개 플레이리스트:")
                    for i, pl in enumerate(top_playlists[:5], 1):
                        print(f"      {i}. [{pl.get('similarity', 0):.4f}] {pl.get('title', 'Unknown')}")
                        print(f"         ID: {pl.get('playlist_id', 'N/A')}")
            else:
                print(f"\n⚠️ 디버그 정보 없음! (백엔드 구버전)")

            # 트랙 결과
            if results:
                print(f"\n   🎵 반환된 트랙 (상위 5개):")
                for i, track in enumerate(results[:5], 1):
                    print(f"      {i}. {track.get('track_name', 'Unknown')} - {track.get('artist', 'Unknown')}")
            else:
                print(f"\n   ❌ 트랙 결과 없음!")

        else:
            print(f"❌ 실패! (HTTP {response.status_code})")
            print(f"   에러: {response.text[:200]}")

    except Exception as e:
        print(f"❌ 예외 발생: {e}")

print("\n" + "=" * 80)
print("✅ 테스트 완료")
print("=" * 80)
