# HNSW 인덱스 성능 비교 테스트 가이드

이 가이드는 HNSW 벡터 인덱스 적용 전후 성능을 정량적으로 비교하는 절차를 설명합니다.

## 📋 테스트 목표

- **현재 상태**: 시퀀셜 스캔으로 인한 느린 벡터 검색 (첫 쿼리 7-13초, 이후 2-3초)
- **목표**: HNSW 인덱스 적용으로 10-100배 속도 향상 (첫 쿼리 1초 이내, 이후 0.2-0.5초)

---

## 🧪 Step 1: 인덱스 적용 전 벤치마크 (Baseline)

### 1-1. 현재 인덱스 상태 확인

Supabase SQL Editor에서 실행:

```sql
-- 현재 적용된 인덱스 확인
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename IN ('new_playlists', 'new_track_embeddings')
ORDER BY tablename, indexname;
```

**예상 결과**: HNSW 인덱스 없음 (`idx_new_playlists_embedding_hnsw`, `idx_new_track_embeddings_embedding_hnsw` 없음)

### 1-2. Baseline 벤치마크 실행

```bash
# 키워드 검색 벤치마크 (110개 키워드)
python3 benchmarks/benchmark_keyword_search.py https://api.dynplayer.win

# 결과 파일 이름 변경 (나중에 비교하기 위해)
mv benchmark_results_*.json benchmark_results_BEFORE_HNSW.json
```

**기록할 메트릭**:
- `mean_total_time`: 평균 전체 응답 시간
- `mean_db_time`: 평균 DB 처리 시간 ← **가장 중요**
- `min_db_time` / `max_db_time`: DB 시간 범위

**예상 값** (인덱스 없음):
- 첫 쿼리 DB 시간: 7-13초
- 이후 쿼리 평균 DB 시간: 2-3초

---

## 🚀 Step 2: HNSW 인덱스 적용

### 2-1. 인덱스 생성 준비

**중요**: 각 인덱스를 **개별적으로** 실행해야 타임아웃 방지됩니다.

Supabase SQL Editor 설정:
1. Statement timeout: **10분** (600초) 이상으로 설정
2. 각 인덱스를 **별도의 쿼리**로 실행

### 2-2. Step 1 - Playlist 벡터 인덱스 (가장 중요)

`sql_optimization/optimize_keyword_search_step1.sql` 파일 내용을 Supabase SQL Editor에서 실행:

```sql
-- Step 1: new_playlists 벡터 인덱스만 먼저 생성
-- 가장 중요한 인덱스 (키워드 검색에서 가장 많이 사용)

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_new_playlists_embedding_hnsw
ON new_playlists
USING hnsw (embedding vector_ip_ops)
WITH (m = 16, ef_construction = 64);

-- 인덱스 확인
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'new_playlists'
  AND indexname = 'idx_new_playlists_embedding_hnsw';
```

**예상 소요 시간**: 1-3분
**성공 확인**: 쿼리 결과에 인덱스 정보가 표시됨

### 2-3. Step 2 - Track Key 인덱스 (JOIN 최적화)

`sql_optimization/optimize_keyword_search_step2.sql` 파일 내용을 실행:

```sql
-- Step 2: track_key 인덱스 생성 (빠름)

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_new_track_embeddings_track_key
ON new_track_embeddings(track_key);

-- 인덱스 확인
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'new_track_embeddings'
  AND indexname = 'idx_new_track_embeddings_track_key';
```

**예상 소요 시간**: 10-30초 (B-tree 인덱스라 빠름)

### 2-4. Step 3 - Track Embedding 벡터 인덱스 (평균 기반 추천)

`sql_optimization/optimize_keyword_search_step3.sql` 파일 내용을 실행:

```sql
-- Step 3: new_track_embeddings 벡터 인덱스 생성

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_new_track_embeddings_embedding_hnsw
ON new_track_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 인덱스 확인
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'new_track_embeddings'
  AND indexname = 'idx_new_track_embeddings_embedding_hnsw';
```

**예상 소요 시간**: 2-5분

### 2-5. 최종 인덱스 확인

모든 인덱스가 성공적으로 생성되었는지 확인:

```sql
-- 모든 벡터 인덱스 확인
SELECT
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE indexname LIKE '%hnsw%'
   OR indexname = 'idx_new_track_embeddings_track_key'
ORDER BY tablename, indexname;
```

**예상 결과**: 3개의 인덱스 모두 표시되어야 함

---

## 📊 Step 3: 인덱스 적용 후 벤치마크

### 3-1. 서버 재시작 (워밍업 실행)

NCP 서버에 접속하여:

```bash
# 최신 코드 pull (5회 워밍업 포함)
cd ~/adaptive-music-player-backend
git pull

# 서비스 재시작
sudo systemctl restart dynplayer-backend

# 워밍업 로그 확인 (30초 대기 후)
sleep 30
sudo journalctl -u dynplayer-backend -n 50 | grep -A 3 "Warming up"
```

**확인 사항**:
- "✅ Database fully warmed up (X.XXs, 5 queries)" 로그가 보여야 함
- 워밍업 시간이 대폭 감소했는지 확인 (13초 → 1-2초 예상)

### 3-2. After 벤치마크 실행

```bash
# 같은 키워드 검색 벤치마크 실행
python3 benchmarks/benchmark_keyword_search.py https://api.dynplayer.win

# 결과 파일 이름 변경
mv benchmark_results_*.json benchmark_results_AFTER_HNSW.json
```

---

## 📈 Step 4: 성능 비교 분석

### 4-1. Before vs After 비교

두 JSON 파일을 열어서 비교:

```bash
# Before
cat benchmark_results_BEFORE_HNSW.json | grep -E "mean_total_time|mean_db_time|max_db_time"

# After
cat benchmark_results_AFTER_HNSW.json | grep -E "mean_total_time|mean_db_time|max_db_time"
```

### 4-2. 예상 개선 효과

| 메트릭 | Before (인덱스 없음) | After (HNSW) | 개선 배수 |
|--------|---------------------|--------------|----------|
| 첫 쿼리 DB 시간 | 7-13초 | 0.5-1초 | **10-20배** |
| 평균 DB 시간 | 2-3초 | 0.2-0.5초 | **5-10배** |
| 최대 DB 시간 | 13초 | 1초 | **13배** |
| 평균 전체 응답 시간 | 3-4초 | 1-1.5초 | **2-3배** |

### 4-3. 성공 기준

✅ **인덱스 적용 성공 기준**:
- 평균 DB 시간 < 1초
- 최대 DB 시간 < 2초
- 첫 쿼리 DB 시간 < 1초 (워밍업 효과)
- DB 시간이 전체 응답 시간의 30% 미만 (OpenAI가 주된 병목으로 전환)

---

## 🔍 Step 5: 워밍업 효과 확인

### 5-1. Cold Start 제거 확인

새 터미널에서 단일 키워드 검색 테스트:

```bash
# 배포 서버 첫 요청 (재시작 후)
curl -X POST https://api.dynplayer.win/search-by-keyword \
  -H "Content-Type: application/json" \
  -d '{"keyword": "여름 해변 분위기"}' \
  | jq '.timing'
```

**기대 결과**:
```json
{
  "openai_time": 0.8,
  "projection_time": 0.02,
  "db_time": 0.3,  // ← 1초 미만!
  "total_time": 1.12
}
```

### 5-2. 워밍업 검증

`sql_optimization/check_warmup.sh` 스크립트를 NCP 서버에서 실행:

```bash
bash sql_optimization/check_warmup.sh
```

**확인 사항**:
- 5회 워밍업 쿼리 모두 실행됨
- 각 워밍업 쿼리가 1초 이내에 완료됨

---

## 🎯 최종 체크리스트

- [ ] Baseline 벤치마크 실행 완료 (`benchmark_results_BEFORE_HNSW.json`)
- [ ] 3개 HNSW 인덱스 모두 생성 확인
- [ ] 서버 재시작 및 워밍업 로그 확인
- [ ] After 벤치마크 실행 완료 (`benchmark_results_AFTER_HNSW.json`)
- [ ] DB 시간이 10배 이상 개선됨 확인
- [ ] 첫 쿼리도 1초 이내로 응답 (Cold Start 제거)

---

## ⚠️ 문제 해결

### 인덱스 생성 타임아웃 발생 시

```sql
-- 현재 진행 중인 인덱스 생성 확인
SELECT pid, now() - query_start as duration, state, query
FROM pg_stat_activity
WHERE query LIKE '%CREATE INDEX%';

-- 필요시 취소 (오래 걸리면)
SELECT pg_cancel_backend(pid);
```

### 인덱스 삭제 (재생성 필요 시)

```sql
-- 인덱스 삭제 (CONCURRENTLY로 안전하게)
DROP INDEX CONCURRENTLY IF EXISTS idx_new_playlists_embedding_hnsw;
DROP INDEX CONCURRENTLY IF EXISTS idx_new_track_embeddings_track_key;
DROP INDEX CONCURRENTLY IF EXISTS idx_new_track_embeddings_embedding_hnsw;
```

---

## 📝 참고 자료

- HNSW 알고리즘: Hierarchical Navigable Small World graphs for ANN search
- pgvector 문서: https://github.com/pgvector/pgvector
- `vector_ip_ops`: Inner product similarity (playlists)
- `vector_cosine_ops`: Cosine similarity (track embeddings)
- `m=16`: HNSW graph 연결 수 (정확도 vs 속도 균형)
- `ef_construction=64`: 인덱스 생성 시 탐색 깊이 (높을수록 정확하지만 느림)
