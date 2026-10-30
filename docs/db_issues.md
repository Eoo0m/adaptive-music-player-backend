# DB 이슈 기록

## 2026-10-30 DB 크래시

### 증상
- `57014: canceling statement due to statement timeout` (21:58~22:16)
- `57014: canceling authentication due to timeout` (22:48~23:05)
- `57P03: the database system is not accepting connections` (23:10, 완전 크래시)

### 원인
`map_apple_music_ids.py` 배치 매핑 스크립트가 매 50개 배치마다 DB UPDATE를 연속으로 날린 것이 원인.

**연쇄 흐름:**
1. 50개 배치 UPDATE가 짧은 간격으로 연속 실행
2. `track_embeddings` 테이블이 벡터 컬럼(64d + 512d + 128d) 때문에 행이 무거워 UPDATE 하나에 시간이 오래 걸림
3. UPDATE가 쌓이면서 DB 연결 포화
4. statement timeout → 인증 timeout → 완전 크래시

### HOT Update 현황
```sql
SELECT relname, n_tup_upd, n_tup_hot_upd,
       ROUND(100.0 * n_tup_hot_upd / NULLIF(n_tup_upd, 0), 2) AS hot_update_pct
FROM pg_stat_user_tables
WHERE relname = 'track_embeddings';
-- 결과: hot_update_pct = 6.59% (매우 낮음)
```

HOT update가 6%밖에 안 되는 이유: 벡터 컬럼들 때문에 행 크기가 커서 페이지(8KB)당 여유 공간이 없음. fillfactor 조정으로도 해결 어려움.

### HNSW 인덱스 갱신 비용
PostgreSQL은 MVCC 방식으로 UPDATE 시 어떤 컬럼을 바꾸든 **새 행 버전(tuple)을 통째로 씁니다.**
새 tuple이 생기면 해당 행을 가리키는 **모든 인덱스**가 새 위치를 가리키도록 갱신해야 합니다.

`apple_music_id`, `isrc`(TEXT)만 바꿔도 HNSW 인덱스 3개가 전부 갱신됩니다:
- `embedding` 64d — HNSW (m=32, ef=200)
- `projected_embedding` 512d — HNSW (m=16, ef=64)
- `itemtower_embedding` 128d — HNSW (m=16, ef=64)

HOT update(같은 페이지 내 in-place 갱신, 인덱스 갱신 생략)가 가능했다면 HNSW 갱신을 피할 수 있었지만,
벡터 컬럼들 때문에 행이 커서 페이지에 여유 공간이 없어 HOT이 불가능했음 (HOT 비율 6.59%).

→ 매 UPDATE마다 HNSW 인덱스 3개를 갱신하는 비용이 더해져 느려짐.

### 인덱스 현황
`track_key`에 인덱스는 있음 (문제 아님):
- `track_embeddings_track_key_key` — UNIQUE btree
- `idx_track_embeddings_track_key` — btree (중복)

### 검증
1개씩 UPDATE + 1.5초 sleep → 정상 동작 확인 (2026-10-31)
배치 UPDATE → DB 크래시 재현

### 근본 원인
`track_embeddings` 한 테이블에 메타 + 벡터 3종이 다 있어서 행이 무거움.
- `embedding` 64d (~256 bytes)
- `projected_embedding` 512d (~2KB)
- `itemtower_embedding` 128d (~512 bytes)
- 행 하나가 ~3KB 이상 → 페이지당 2~3행

### 해결책 (단기)
CSV 누적 후 마지막에 단 1번 COPY → UPDATE (`map_apple_music_ids.py --apply`)

### 해결책 (장기, 미적용)
테이블 분리:
- `tracks` — 메타만 (가벼움, HOT update 잘 됨)
- `track_embeddings_simgcl` — embedding 64d
- `track_embeddings_clip` — projected_embedding 512d
- `track_embeddings_twotower` — itemtower_embedding 128d

메타 테이블이 가벼워지면 배치 UPDATE도 빠르게 끝나고 timeout 위험 없음.
