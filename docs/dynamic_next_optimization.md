# dynamic-next 최적화: 백엔드 API → 프론트 JS 계산 이전

**날짜:** 2026-11-01

---

## 문제

다음 곡 재생 시마다 `/dynamic-next` API를 호출하고 있었음.
매 호출마다 DB 쿼리 + 서버 왕복 RTT가 발생해 체감 지연이 컸음.

---

## 기존 방식

### API: `POST /dynamic-next`

**요청:**
```json
{
  "track_key": "현재 트랙 key",
  "listened_half": true,
  "candidate_track_keys": ["트랙1", "트랙2", ...]
}
```

**서버 처리:**
1. 현재 트랙의 `embedding` (64d) DB 조회
2. `listened_half=true` → pgvector `<=>` 로 cosine 가장 가까운 트랙 반환
3. `listened_half=false` → `playlist_count DESC` 상위 200개 후보 중 cosine 가장 먼 트랙 반환

**문제점:**
- 프론트에 이미 지도 트랙 목록(`lastMapData`)이 있는데도 매번 API 호출
- embedding은 `/music-map` 응답에 포함되지 않아서 프론트에서 계산 불가였음

---

## 변경 내용

### 1. 백엔드: `/music-map` 응답에 `embedding` 추가

**파일:** `routes/music_map.py`

- fill 쿼리에 `v.embedding::text` 추가 (seed 쿼리는 이미 포함되어 있었음)
- 응답 tracks 배열 각 항목에 `"embedding": [float, ...]` (64d) 포함

**추가 페이로드:** 트랙당 ~300B × 최대 450트랙 ≈ 135KB (gzip 후 ~35KB)
`/music-map`은 지도 열 때 1회만 호출하므로 marginal cost 작음.

### 2. 프론트: `/dynamic-next` fetch 제거 → JS cosine 계산으로 대체

**파일:** `js/main.js` — `playerDynamicNext()` 함수

```js
const cosine = (a, b) => {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
};

// listened_half=true: 가장 가까운 트랙
// listened_half=false: playlist_count 상위 200개 중 가장 먼 트랙
```

백엔드의 DB 쿼리 로직을 JS로 그대로 재현. 네트워크 호출 없이 처리.

---

## 결과

| | 기존 | 변경 후 |
|---|---|---|
| 다음 곡 선택 지연 | 서버 RTT + DB 쿼리 (수백ms) | < 1ms (JS 연산) |
| `/music-map` 응답 크기 | ~수KB | +135KB (gzip ~35KB) |
| `/dynamic-next` 호출 | 매 곡마다 | 없음 |

---

## 주의

- `/dynamic-next` API는 삭제하지 않음 (다른 클라이언트 호환성)
- `lastMapData`가 없는 상태(지도 미오픈)에서는 다음 곡 선택 안 됨 — 기존과 동일한 동작
