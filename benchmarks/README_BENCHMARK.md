# 벤치마크 실행 가이드

## 키워드 검색 벤치마크

110개의 다양한 키워드로 `/search-by-keyword` 엔드포인트 성능 측정

```bash
# 로컬 서버 테스트
python3 benchmarks/benchmark_keyword_search.py

# 배포 서버 테스트
python3 benchmarks/benchmark_keyword_search.py https://api.dynplayer.win
```

**측정 항목**:
- 총 응답 시간 (min, max, mean, median, stdev)
- 단계별 시간 (OpenAI, Projection, DB)
- 성공률

**결과 파일**: `benchmark_results_YYYYMMDD_HHMMSS.json`

---

## 제목 검색 벤치마크

60개의 다양한 제목/아티스트 키워드로 `/search-songs` 엔드포인트 성능 측정

```bash
# 로컬 서버 테스트
python3 benchmarks/benchmark_title_search.py

# 배포 서버 테스트
python3 benchmarks/benchmark_title_search.py https://api.dynplayer.win
```

**측정 항목**:
- 총 응답 시간 (min, max, mean, median, stdev)
- 결과 개수
- 성공률

**결과 파일**: `benchmark_title_search_YYYYMMDD_HHMMSS.json`

---

## 빠른 실행 예시

```bash
# 두 벤치마크 모두 실행 (배포 서버)
python3 benchmarks/benchmark_keyword_search.py https://api.dynplayer.win
python3 benchmarks/benchmark_title_search.py https://api.dynplayer.win
```
