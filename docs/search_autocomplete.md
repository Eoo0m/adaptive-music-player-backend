# 검색 자동완성 시스템

## 목적
검색창에 타이핑하는 동안 실시간으로 키워드를 제안한다.
곡명, 아티스트, 분위기 키워드를 한 곳에서 제안하므로 사용자가 뭘 입력해야 할지 모를 때도 자연스럽게 검색을 시작할 수 있다.

## 테이블: `search_keywords`

```sql
CREATE TABLE search_keywords (
    keyword      TEXT PRIMARY KEY,
    search_count INT NOT NULL DEFAULT 0
);
CREATE INDEX idx_search_keywords_prefix ON search_keywords (keyword text_pattern_ops);
CREATE INDEX idx_search_keywords_trgm   ON search_keywords USING GIN (keyword gin_trgm_ops);
```

## 초기 seed 구성

| 소스 | 조건 | search_count 초기값 |
|---|---|---|
| `tracks.title` | 전체 | 해당 곡의 `playlist_count` |
| `tracks.artist` | 아티스트 총 저장수 ≥ 100 | 아티스트 소속 곡 `playlist_count` 합산 |

총 약 79,672개 (곡명 60,103 + 아티스트 2,439, 중복 제거 후).

seed는 2026-11-10 서버에서 1회 수동 실행. 이후 검색마다 `/log-action`을 통해 카운트 자동 증가.

## 스코어링 공식

```sql
score = prefix_bonus                              -- 1.0 (keyword LIKE q%)
      + trigram_similarity(keyword, q) * 0.4      -- pg_trgm similarity()
      + ln(search_count + 1) * 0.3               -- 인기도
```

prefix 일치가 가장 강한 신호, trigram은 오타/부분일치 보완, 인기도는 같은 조건일 때 타이브레이커 역할.

## API

### `GET /search-autocomplete?q=...`
- 인증 불필요
- 최대 8개 반환
- 응답: `{ "suggestions": ["Taylor Swift", "Taylor Dayne", ...] }`

### `POST /log-action` (search 타입)
검색 실행 시 `action_type=search`로 호출하면 `search_keywords`에 자동 upsert:
```sql
INSERT INTO search_keywords (keyword, search_count) VALUES ($query, 1)
ON CONFLICT (keyword) DO UPDATE SET search_count = search_count + 1
```

### `GET /recent-searches`
로그인 유저의 최근 검색 키워드 (최대 10개, 최신순).
`user_action_logs`의 search 로그에서 쿼리.

### `DELETE /recent-searches/{query}`
특정 키워드 기록 삭제.

## iOS 동작 흐름

```
타이핑 (onChange)
  └─ 0.25s debounce
      └─ GET /search-autocomplete?q=...
          └─ 드롭다운 표시 (최대 8개)
              └─ 탭 → query 채우고 doSearch() 실행
                  └─ POST /log-action (search)  ← search_count +1
                     GET /search-autocomplete 결과 숨김
```

## 추후 확장 가능한 것들
- 분위기 태그(GPT 생성 플리 캡션)를 별도 마이그레이션으로 추가
- 개인화: 로그인 유저에겐 본인 검색 기록 가중치 추가
- 검색어 트렌드 대시보드 (`search_keywords` ORDER BY search_count)
