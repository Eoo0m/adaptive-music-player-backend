# DB

## 테이블 목록

| 테이블 | 설명 |
|---|---|
| `tracks` | 트랙 메타 (track_key, title, artist, album, cover_image_url, apple_music_id, playlist_count) |
| `track_embeddings` | SimGCL 64d + CLIP 512d + Two-Tower 128d 통합 (레거시, 현재는 분리 테이블 사용) |
| `track_vectors_simgcl` | SimGCL embedding 64d. `/next-track`, `/dynamic-next`, `/find-similar-tracks`, `/music-map` 전용 |
| `track_vectors_clip` | CLIP projected_embedding 512d |
| `track_vectors_twotower` | Two-Tower itemtower_embedding 128d |
| `keyword_embeddings` | 자주 쓰이는 무드 키워드 임베딩 캐시 (3072d). `/smart-search` 가속용 |
| `search_keywords` | 자동완성용 키워드 + 검색수 |
| `users` | Apple OAuth 유저 |
| `user_favorites` | 찜 목록 |
| `user_action_logs` | 행동 로그 (search/play/favorite/select) |
| `user_maps` | 취향지도 DB 캐시 |
| `user_playlists` | 플리 빌더로 저장한 플레이리스트 |
| `listening_logs` | 재생 로그 |

---

## pgvector 연산자

**절대 혼동 금지.**

| 연산자 | 의미 | 필요 인덱스 opclass |
|---|---|---|
| `<=>` | Cosine distance | `vector_cosine_ops` |
| `<#>` | Negative Inner Product | `vector_ip_ops` |
| `<->` | L2 distance | `vector_l2_ops` |

현재 모든 HNSW 인덱스는 `vector_cosine_ops` → **반드시 `<=>` 사용**.
`<#>`로 바꾸면 인덱스 못 타고 풀스캔 → 5~30초 소요.

similarity 공식: `(1 - (embedding <=> query))::float`

---

## HNSW 인덱스

```sql
-- track_vectors_simgcl
CREATE INDEX idx_simgcl_embedding_hnsw
ON track_vectors_simgcl USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 200);

-- track_embeddings (레거시, projected_embedding용)
-- CREATE INDEX idx_clip_projected_hnsw ...
```

인덱스 확인:
```sql
SELECT indexname, indexdef FROM pg_indexes
WHERE tablename IN ('track_vectors_simgcl', 'track_embeddings');
```

---

## 테이블 분리 배경

### 2026-10-30 DB 크래시

`track_embeddings` 한 테이블에 메타 + 벡터 3종이 함께 있어 행이 무거웠음(~3KB+).
배치 UPDATE 시 MVCC 특성상 어떤 컬럼을 바꿔도 새 행 버전을 통째로 쓰고,
HNSW 인덱스 3개가 전부 갱신 → 짧은 간격의 50개 배치 UPDATE로 DB 연결 포화 → 완전 크래시.

HOT update 비율: 6.59% (행이 커서 페이지당 여유 공간 없음 → HOT 불가).

### 현재 구조

`track_embeddings` (레거시) + 분리된 `track_vectors_*` 테이블 병존.
메타 조회는 `tracks`, 벡터 검색은 `track_vectors_simgcl` 사용.

```sql
-- 올바른 쿼리 패턴 (HNSW 인덱스 활용)
-- 1. 현재 트랙 임베딩 먼저 조회
SELECT embedding::text FROM track_vectors_simgcl WHERE track_key = $1;

-- 2. ANN 검색
SELECT t.*, (1 - (v.embedding <=> $2::vector))::float AS similarity
FROM track_vectors_simgcl v
JOIN tracks t ON v.track_key = t.track_key
WHERE v.track_key != $1
ORDER BY v.embedding <=> $2::vector
LIMIT $3;
```

> **주의**: WITH 절(CTE) 패턴은 HNSW 인덱스를 못 타는 경우 있음 → 2-step 쿼리 사용.

### 미완료: track_vectors_clip, track_vectors_twotower 전환

`track_embeddings.projected_embedding`, `itemtower_embedding` 조회는 아직 레거시 테이블 사용.
분리 테이블(`track_vectors_clip`, `track_vectors_twotower`)은 생성됐으나 코드 전환 미완료.
배치 UPDATE가 없는 상태라 당장 급하지 않음.
