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
  "mood_results": [...]
}
```

---

## `/search-by-keyword` (구버전, 미사용)

무드 검색만 수행, 의도 분류 없음. 현재 프론트/앱 어디서도 호출하지 않음. `smart-search` 도입으로 대체됨.

---

## iOS 자동완성 흐름

```
타이핑 (onChange) → 0.25s debounce
    → GET /search-autocomplete?q=...
        → 드롭다운 표시 (최대 8개)
            → 탭 → doSearch() 실행
                → POST /log-action (search)  ← search_count +1
```
