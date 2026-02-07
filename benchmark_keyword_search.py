#!/usr/bin/env python3
"""
키워드 검색 성능 벤치마크 스크립트
100개의 다양한 키워드로 검색 속도를 측정합니다.
"""

import requests
import time
import statistics
from typing import List, Dict
import json

# 테스트할 키워드 100개 (다양한 장르, 분위기, 아티스트)
TEST_KEYWORDS = [
    "happy"
    # 감정/분위기
    #     , "sad", "energetic", "calm", "romantic", "angry", "peaceful", "exciting",
    #     "melancholic", "cheerful", "dreamy", "intense", "relaxing", "uplifting", "dark",
    #     # 장르
    #     "rock", "pop", "jazz", "classical", "hip hop", "electronic", "country", "blues",
    #     "reggae", "metal", "folk", "r&b", "soul", "funk", "disco", "techno",
    #     # 활동/상황
    #     "workout", "study", "party", "sleep", "driving", "cooking", "meditation", "dance",
    #     "running", "yoga", "focus", "chill", "morning", "night", "rain", "summer",
    #     # 악기
    #     "piano", "guitar", "drums", "violin", "saxophone", "bass", "synth", "flute",
    #     # 음악 특성
    #     "acoustic", "instrumental", "vocal", "orchestral", "beat", "melody", "harmony", "rhythm",
    #     # 시대/스타일
    #     "80s", "90s", "retro", "modern", "vintage", "contemporary", "classic", "new",
    #     # 인기 아티스트/밴드
    #     "beatles", "queen", "elvis", "madonna", "michael jackson", "led zeppelin", "pink floyd",
    #     "david bowie", "nirvana", "radiohead", "coldplay", "imagine dragons", "maroon 5",
    #     # 한국어 키워드
    #     "사랑", "이별", "행복", "슬픔", "희망", "그리움", "추억", "여행",
    #     "봄", "가을", "겨울", "비", "바다", "밤", "별", "달",
    #     # 복합 키워드
    #     "love song", "party music", "sad ballad", "upbeat pop", "smooth jazz",
    #     "hard rock", "indie folk", "ambient music", "latin dance", "gospel music"
]


def test_keyword_search(base_url: str, keyword: str) -> Dict:
    """단일 키워드 검색 테스트"""
    url = f"{base_url}/search-by-keyword"

    start_time = time.time()
    try:
        response = requests.post(
            url,
            json={"keyword": keyword},  # 'query'가 아니라 'keyword' 필드 사용
            timeout=60,  # 60초 타임아웃
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            num_results = len(data.get("results", []))

            # 각 단계별 시간 추출
            timings = data.get("timings", {})

            return {
                "keyword": keyword,
                "success": True,
                "elapsed": elapsed,
                "num_results": num_results,
                "status_code": 200,
                "timings": timings,  # OpenAI, Projection, DB 등 각 단계 시간
            }
        else:
            return {
                "keyword": keyword,
                "success": False,
                "elapsed": elapsed,
                "error": f"HTTP {response.status_code}",
                "status_code": response.status_code,
            }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "keyword": keyword,
            "success": False,
            "elapsed": elapsed,
            "error": str(e),
            "status_code": None,
        }


def run_benchmark(base_url: str, keywords: List[str] = None) -> Dict:
    """벤치마크 실행"""
    if keywords is None:
        keywords = TEST_KEYWORDS

    print(f"🚀 Starting benchmark with {len(keywords)} keywords")
    print(f"📍 Target: {base_url}")
    print("-" * 80)

    results = []
    success_times = []

    for i, keyword in enumerate(keywords, 1):
        print(f"[{i:3d}/{len(keywords)}] Testing: '{keyword[:30]:30s}'", end=" ... ")

        result = test_keyword_search(base_url, keyword)
        results.append(result)

        if result["success"]:
            success_times.append(result["elapsed"])
            timings = result.get("timings", {})
            timing_str = ""
            if timings:
                parts = []
                if "openai" in timings:
                    parts.append(f"OpenAI:{timings['openai']:.2f}s")
                if "projection" in timings:
                    parts.append(f"Proj:{timings['projection']:.2f}s")
                if "db" in timings:
                    parts.append(f"DB:{timings['db']:.2f}s")
                if parts:
                    timing_str = f" ({', '.join(parts)})"
            print(
                f"✅ {result['elapsed']:.2f}s{timing_str} - {result['num_results']} results"
            )
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

        # 각 단계별 평균 시간 계산
        openai_times = [
            r.get("timings", {}).get("openai")
            for r in results
            if r["success"] and r.get("timings", {}).get("openai")
        ]
        projection_times = [
            r.get("timings", {}).get("projection")
            for r in results
            if r["success"] and r.get("timings", {}).get("projection")
        ]
        db_times = [
            r.get("timings", {}).get("db")
            for r in results
            if r["success"] and r.get("timings", {}).get("db")
        ]

        if openai_times:
            stats["avg_openai_time"] = statistics.mean(openai_times)
        if projection_times:
            stats["avg_projection_time"] = statistics.mean(projection_times)
        if db_times:
            stats["avg_db_time"] = statistics.mean(db_times)

    return {"stats": stats, "results": results}


def print_summary(benchmark_result: Dict):
    """결과 요약 출력"""
    stats = benchmark_result["stats"]

    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Tests:     {stats['total_tests']}")
    print(f"Successful:      {stats['successful']} ({stats['success_rate']:.1f}%)")
    print(f"Failed:          {stats['failed']}")

    if stats["successful"] > 0:
        print(f"\n⏱️  Total Response Times:")
        print(f"  Minimum:       {stats['min_time']:.2f}s")
        print(f"  Maximum:       {stats['max_time']:.2f}s")
        print(f"  Mean:          {stats['mean_time']:.2f}s")
        print(f"  Median:        {stats['median_time']:.2f}s")
        print(f"  Std Dev:       {stats['stdev_time']:.2f}s")

        if (
            "avg_openai_time" in stats
            or "avg_projection_time" in stats
            or "avg_db_time" in stats
        ):
            print(f"\n🔍 Average Time by Stage:")
            if "avg_openai_time" in stats:
                print(f"  OpenAI API:    {stats['avg_openai_time']:.2f}s")
            if "avg_projection_time" in stats:
                print(f"  Projection:    {stats['avg_projection_time']:.2f}s")
            if "avg_db_time" in stats:
                print(f"  Database:      {stats['avg_db_time']:.2f}s")

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

    print(f"🎵 DynPlayer Keyword Search Benchmark")
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
    filename = f"benchmark_results_{timestamp}.json"
    save_results(benchmark_result, filename)

    print("\n✨ Benchmark complete!")
