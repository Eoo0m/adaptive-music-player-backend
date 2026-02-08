#!/usr/bin/env python3
"""
제목 검색 성능 벤치마크 스크립트
다양한 제목 키워드로 검색 속도를 측정합니다.
"""

import requests
import time
import statistics
from typing import List, Dict
import json

# 테스트할 제목 키워드 (다양한 패턴)
TEST_QUERIES = [
    # 한글자
    "love", "you", "me", "we", "I", "be", "the", "my", "to", "all",
    # 일반적인 단어
    "home", "heart", "dream", "night", "light", "time", "life", "world",
    "song", "music", "dance", "party", "girl", "boy", "baby", "summer",
    # 아티스트 이름
    "taylor", "drake", "billie", "ariana", "ed", "justin", "beyonce",
    "adele", "bruno", "lady", "katy", "rihanna", "kanye", "eminem",
    # 한국어
    "사랑", "너", "나", "우리", "밤", "별", "꿈", "노래", "그대",
    "마음", "기억", "시간", "이별", "추억", "봄", "가을", "겨울",
    # 복합 키워드
    "in the", "of the", "love me", "take me", "give me", "with me",
    "my heart", "your love", "never gonna", "wanna be", "don't know",
]

def test_title_search(base_url: str, query: str) -> Dict:
    """단일 제목 검색 테스트"""
    url = f"{base_url}/search-songs"

    start_time = time.time()
    try:
        response = requests.post(
            url,
            json={"query": query},
            timeout=60,  # 60초 타임아웃
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            num_results = len(data.get("results", []))

            return {
                "query": query,
                "success": True,
                "elapsed": elapsed,
                "num_results": num_results,
                "status_code": 200,
            }
        else:
            return {
                "query": query,
                "success": False,
                "elapsed": elapsed,
                "error": f"HTTP {response.status_code}",
                "status_code": response.status_code,
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "query": query,
            "success": False,
            "elapsed": elapsed,
            "error": str(e),
            "status_code": None,
        }


def run_benchmark(base_url: str, queries: List[str] = None) -> Dict:
    """벤치마크 실행"""
    if queries is None:
        queries = TEST_QUERIES

    print(f"🚀 Starting title search benchmark with {len(queries)} queries")
    print(f"📍 Target: {base_url}")
    print("-" * 80)

    results = []
    success_times = []

    for i, query in enumerate(queries, 1):
        print(f"[{i:3d}/{len(queries)}] Testing: '{query[:30]:30s}'", end=" ... ")

        result = test_title_search(base_url, query)
        results.append(result)

        if result["success"]:
            success_times.append(result["elapsed"])
            print(f"✅ {result['elapsed']:.2f}s - {result['num_results']} results")
        else:
            print(f"❌ {result['elapsed']:.2f}s - {result['error']}")

        # 서버 부하 방지를 위한 짧은 딜레이
        time.sleep(0.1)

    # 통계 계산
    total_tests = len(results)
    successful_tests = len(success_times)
    failed_tests = total_tests - successful_tests

    stats = {
        "total_tests": total_tests,
        "successful": successful_tests,
        "failed": failed_tests,
        "success_rate": (
            (successful_tests / total_tests * 100) if total_tests > 0 else 0
        ),
    }

    if success_times:
        stats.update(
            {
                "min_time": min(success_times),
                "max_time": max(success_times),
                "mean_time": statistics.mean(success_times),
                "median_time": statistics.median(success_times),
                "stdev_time": (
                    statistics.stdev(success_times) if len(success_times) > 1 else 0
                ),
            }
        )

    return {"stats": stats, "results": results}


def print_summary(benchmark_result: Dict):
    """결과 요약 출력"""
    stats = benchmark_result["stats"]

    print("\n" + "=" * 80)
    print("📊 TITLE SEARCH BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Tests:     {stats['total_tests']}")
    print(f"Successful:      {stats['successful']} ({stats['success_rate']:.1f}%)")
    print(f"Failed:          {stats['failed']}")

    if stats["successful"] > 0:
        print(f"\n⏱️  Response Times:")
        print(f"  Minimum:       {stats['min_time']:.2f}s")
        print(f"  Maximum:       {stats['max_time']:.2f}s")
        print(f"  Mean:          {stats['mean_time']:.2f}s")
        print(f"  Median:        {stats['median_time']:.2f}s")
        print(f"  Std Dev:       {stats['stdev_time']:.2f}s")

    print("=" * 80)


def save_results(benchmark_result: Dict, filename: str):
    """결과를 JSON 파일로 저장"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {filename}")


if __name__ == "__main__":
    import sys

    # 기본 URL (로컬 또는 서버)
    DEFAULT_URL = "http://localhost:8000"

    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = DEFAULT_URL

    print(f"🎵 DynPlayer Title Search Benchmark")
    print(f"🌐 Testing URL: {base_url}")
    print()

    # 벤치마크 실행
    start = time.time()
    benchmark_result = run_benchmark(base_url)
    total_time = time.time() - start

    # 결과 출력
    print_summary(benchmark_result)
    print(f"\n⏰ Total benchmark time: {total_time:.2f}s")

    # 결과 저장
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_title_search_{timestamp}.json"
    save_results(benchmark_result, filename)

    print("\n✨ Benchmark complete!")
