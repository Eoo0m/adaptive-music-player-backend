# 검색

## 자동완성 (`/search-autocomplete`)

### 테이블: `search_keywords`

```sql
CREATE TABLE search_keywords (
    keyword      TEXT PRIMARY KEY,
    search_count INT NOT NULL DEFAULT 0
);
CREATE INDEX idx_search_keywords_prefix ON search_keywords (keyword text_pattern_ops);
CREATE INDEX idx_search_keywords_trgm   ON search_keywords USING GIN (keyword gin_trgm_ops);
```

약 79,672개 키워드 (곡명 + 아티스트). 검색할 때마다 `/log-action`을 통해 `search_count` 자동 증가.

### 스코어링

```sql
score = prefix_bonus                              -- 1.0 (keyword LIKE q%)
      + trigram_similarity(keyword, q) * 0.4
      + ln(search_count + 1) * 0.3               -- 인기도
```

- 최대 8개 반환
- 응답: `{ "suggestions": ["Taylor Swift", ...] }`

---

## 스마트 검색 (`/smart-search`)

검색어 하나로 엔티티(곡명/아티스트) + 무드(분위기) 결과를 동시에 반환.
프론트(웹), iOS 앱 모두 이 엔드포인트만 사용.

### 처리 흐름

```
1. 임베딩 (keyword_embeddings 캐시 hit → miss 시 OpenAI API 호출)
2. Intent Classifier → mood 여부 판정 (~1ms, CPU)
3. CLIP 프로젝션 → 512d (~1ms, CPU)
4. asyncio.gather:
   ├─ 엔티티 검색: trigram 텍스트 쿼리 (항상 실행)
   └─ 무드 검색: projected_embedding ANN + MMR (is_mood일 때만)
5. 통합 반환
```

### 키워드 임베딩 캐시 (`keyword_embeddings`)

```sql
CREATE TABLE keyword_embeddings (
    keyword   TEXT PRIMARY KEY,
    embedding vector(3072) NOT NULL  -- text-embedding-3-large
);
```

약 1,000개 자주 쓰이는 무드 키워드 사전 저장 (rock, chill, lofi, workout, 80s, …).
캐시 hit 시 OpenAI API 호출 없이 즉시 프로젝션 단계로 이동 → ~300ms 절약.
캐시 miss인 신규 키워드는 OpenAI를 호출하되 자동 저장하지 않음 (수동 관리).

```python
# _get_embedding 로직
row = await pool.fetchrow(
    "SELECT embedding::text FROM keyword_embeddings WHERE keyword = $1",
    query.lower()
)
if row:
    return np.array(json.loads(row["embedding"])), True  # cache hit
return np.array(get_openai_embedding_with_retry(query)), False
```

### 무드 검색

1. `projected_embedding` ANN: `ORDER BY projected_embedding <=> $query LIMIT candidates`
2. MMR reranking: `lambda=0.7` (유사도 70% + 다양성 30%)
3. `top_k`개 반환

### 응답

```json
{
  "query": "lofi beats",
  "is_mood": true,
  "mood_prob": 0.923,
  "entity_results": [...],
  "mood_results": [...],
  "query_suggestions": [
    {"query": "chill vibes", "tracks": [...]},
    {"query": "late night jazz", "tracks": [...]},
    {"query": "study beats", "tracks": [...]}
  ]
}
```

### 검색 후 쿼리 추천 (`query_suggestions`)

검색어 임베딩(512d)과 user medoid 임베딩을 하이브리드 스코어로 caption 유사도 계산 → `query_mood_cache`에서 top 3 쿼리+트랙 반환. 별도 API 호출 없이 `smart-search`와 병렬 실행되므로 추가 지연 없음.

- `user_id` 없으면 검색어 임베딩만으로 계산
- `user_id` 있으면 검색어 50% + medoid 평균 50% 하이브리드
- `query_mood_cache`에 없는 쿼리는 빈 tracks 반환

**요청 파라미터 추가**: `user_id: Optional[str]`

---

## `/search-by-keyword` (구버전, 미사용)

무드 검색만 수행, 의도 분류 없음. 현재 프론트/앱 어디서도 호출하지 않음. `smart-search` 도입으로 대체됨.

---

## 자동완성 debounce

| 클라이언트 | debounce |
|---|---|
| iOS (SearchView) | 150ms |
| 웹 (search.js) | 150ms |

---

## iOS 자동완성 흐름

```
타이핑 (onChange) → 0.15s debounce
    → GET /search-autocomplete?q=...
        → 드롭다운 표시 (최대 8개)
            → 탭 → doSearch() 실행
                → POST /log-action (search)  ← search_count +1
```

---

## 액션 로그 (`/log-action`)

### action_type 목록

| action_type | 발생 시점 | search_query | selected_track_key | extra |
|---|---|---|---|---|
| `search` | 검색 실행 | 검색어 | - | - |
| `delete_recent_search` | 최근 검색어 삭제 | 삭제한 검색어 | - | - |
| `select_from_search` | 검색결과에서 트랙 탭 | - | 클릭한 트랙 | - |
| `select_from_recommend` | 추천/유사곡에서 트랙 탭 | - | 클릭한 트랙 | - |
| `favorite_add` | 찜 추가 | - | 찜한 트랙 | - |
| `favorite_remove` | 찜 삭제 | - | 찜 취소한 트랙 | - |
| `play` | 재생 시작 | - | 재생한 트랙 | - |
| `pause` | 일시정지 | - | 현재 트랙 | - |
| `next_similar` | 다음 유사곡 버튼 | - | 현재 트랙(from) | - |
| `next_dynamic` | 다이나믹 버튼 | - | 현재 트랙(from) | - |

`search_count` 증가는 `search` 타입만 해당.
