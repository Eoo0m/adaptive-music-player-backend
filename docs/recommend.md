# 추천

## `/next-track` — 유사곡

현재 곡과 **가까운** 트랙을 반환. 취향지도 내 트랙 중 상위 10%에서 랜덤 4개 선택.

```sql
-- 1. 현재 트랙 임베딩 조회
SELECT embedding::text FROM track_vectors_simgcl WHERE track_key = $1;

-- 2. 가까운 순 ANN
SELECT t.*, (1 - (v.embedding <=> $2::vector))::float AS similarity
FROM track_vectors_simgcl v
JOIN tracks t ON v.track_key = t.track_key
WHERE v.track_key != $1 AND v.track_key = ANY($3)  -- candidate_track_keys 필터
ORDER BY v.embedding <=> $2::vector
LIMIT top_k  -- 전체 후보의 10%
```

결과 중 랜덤 4개 반환.

---

## `/dynamic-next` — 다른 곡

현재 곡과 **먼** 트랙을 반환. 취향지도 내 트랙 중 상위 30%에서 랜덤 4개 선택.

```sql
-- 가장 먼 순 ANN (DESC)
ORDER BY v.embedding <=> $2::vector DESC
LIMIT top_k  -- 전체 후보의 30%
```

---

## pgvector ANN 최적화

기존: Python numpy로 300개 임베딩 코사인 유사도 계산 → ~1초
현재: pgvector HNSW 인덱스 `ORDER BY embedding <=> $vector` → 30~40ms

인덱스: `idx_simgcl_embedding_hnsw` (`vector_cosine_ops`, m=32, ef=200)

---

## `/recommend` — Dual Attention 추천

**Auth 필요 (JWT)**. 클릭한 곡(seed)을 기준으로 유저 취향 클러스터와의 Dual Attention 스코어로 랭킹.

### 입력
```json
{ "track_key": "...", "num_recommendations": 5, "exclude_track_keys": [] }
```

### 알고리즘

1. **ANN 100개 후보** — seed 기준 SimGCL 64d cosine ANN
2. **Interest Profile 조회** — `user_interest_clusters` 에서 k개 medoid + global_embedding (없으면 ANN 순서 그대로 반환)
3. **Dual Attention 계산** (τ = 0.2):
   - `SIA = softmax(cos(seed, medoids) / τ)` — seed의 클러스터 관심 분포
   - `CIA = softmax(cos(candidate, medoids) / τ)` — 후보의 클러스터 관심 분포
   - `Combined = normalize(SIA × CIA)` — 두 분포가 겹치는 클러스터 강조
   - `DA Score = Σ_j (Combined_j × cos(candidate, medoid_j))`
4. **Global Pooling Score** — `cos(candidate, global_embedding)` (유저 전체 취향 방향과의 유사도)
5. **Final Score = 0.8 × DA Score + 0.2 × Global Score**
6. Final Score 내림차순 정렬 → `num_recommendations`개 반환

> 클러스터별 가중치(중요도) 반영은 추후 예정.

저장곡 ≤ 5개 (클러스터 미생성) → `mode: "ann_fallback"` 으로 ANN 유사도 순 반환.

### 응답
```json
{
  "recommendations": [...],
  "mode": "dual_attention",   // or "ann_fallback"
  "k": 3                      // 사용된 클러스터 수
}
```

---

## `/interest-profile` — 취향 프로파일 재계산

**Auth 필요 (JWT)**. 수동 트리거용 (보통 `POST /favorites` 에서 자동 호출).

### 알고리즘

- 최근 저장 50곡의 SimGCL 64d 임베딩 사용
- **Ward Hierarchical Clustering** + **거리 임계값 α=2.0** 으로 K 자동 결정
- 각 클러스터의 **Medoid** (클러스터 내 평균과 cosine similarity 가장 가까운 실제 트랙 벡터) 선택
- **Global Pooling 벡터** 계산: 전체 임베딩 L2-정규화 평균 → 다시 L2 정규화
- `user_interest_clusters` 테이블에 upsert

### DB 테이블: `user_interest_clusters`

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `user_id` | UUID PK | 유저 |
| `k` | INT | 결정된 클러스터 수 |
| `medoids` | JSONB | `[[64d], ...]` k개 medoid 벡터 |
| `track_keys` | JSONB | medoid에 해당하는 track_key 목록 |
| `global_embedding` | JSONB | `[64d]` 전체 취향 방향 벡터 (Global Pooling) |
| `updated_at` | TIMESTAMPTZ | 마지막 갱신 시각 |

`POST /favorites` 완료 후 `BackgroundTasks`로 자동 재계산.

---

## `/find-similar-tracks` — 단일 트랙 유사곡

검색 결과에서 트랙 탭 시 호출. SimGCL 임베딩 기반 유사곡 60개 후보에서 랜덤 샘플.
`/next-track`과 달리 candidate 필터 없이 전체 트랙 대상 검색.

---

## 프리페치 (클라이언트)

재생 시작 직후 `/next-track`, `/dynamic-next`를 백그라운드로 미리 요청.
버튼 클릭 시 이미 완료된 Promise/Task를 즉시 소비 → API 왕복 30~40ms가 체감에서 사라짐.

- **웹**: `_prefetchCache[endpoint:trackKey]` = Promise
- **iOS**: `prefetchedNextTracks`, `prefetchedDynamicTracks` + `prefetchedSongs` (MusicCatalogResourceRequest 결과도 미리 캐싱)
