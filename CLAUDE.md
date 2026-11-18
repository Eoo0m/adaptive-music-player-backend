# Dynplayer Backend

## Overview
AI 기반 음악 추천 서비스 백엔드. FastAPI + PostgreSQL(Supabase) + PyTorch 모델 서빙.

## Tech Stack
- **Framework**: FastAPI (Python)
- **DB**: Supabase PostgreSQL + asyncpg (직접 연결, PgBouncer transaction mode)
- **Vector Search**: pgvector + HNSW 인덱스 (`hnsw.ef_search=300` — recommend에서 SET LOCAL로 설정)
- **ML Models**: CLIP (키워드→트랙 프로젝션), Dual Attention + Global Pooling (찜 기반 추천)
- **Auth**: Apple Sign In → JWT (python-jose)
- **Deploy**: NCP Ubuntu, systemd 서비스 (`dynplayer`), `bash deploy.sh`로 배포

## Project Structure
```
main.py              # app 생성, CORS, startup/shutdown, 라우터 등록
database.py          # asyncpg connection pool
models.py            # CLIP 모델 정의 + 로드 (.pt 파일)
utils.py             # mmr_rerank, OpenAI 임베딩 헬퍼
routes/
  smart_search.py    # /smart-search (의도 분류 → 엔티티 + 무드 결과)
  autocomplete.py    # /search-autocomplete
  recommend.py       # /recommend (Dual Attention + Global Pooling, 찜 기반)
  auth.py            # /auth/apple/mobile, /auth/me
  favorites.py       # /favorites (GET/POST/DELETE)
  action_log.py      # /log-action (유저 행동 로그)
  music_map.py       # /music-map, /top-tracks (취향 지도)
  dynamic_streaming.py  # /next-track, /dynamic-next — pgvector ANN
  playlist_builder.py   # /playlist-builder
  interest_profile.py   # /interest-profile (찜 기반 클러스터링)
  user_maps.py       # /user-maps (뮤직맵 저장/불러오기)
sql/                 # DB 스키마, 인덱스, 테이블 정의
experiments/         # HNSW recall 실험 등
docs/                # 변경 이력, 이슈 기록
```

## Key APIs
| Endpoint | Method | Auth | Description |
|---|---|---|---|
| /search-autocomplete | GET | X | 자동완성 |
| /smart-search | POST | X | 의도 분류 → 엔티티(trigram) + 무드(CLIP ANN+MMR) 결과 동시 반환 |
| /recommend | POST | X/JWT | Dual Attention + Global Pooling 추천 (비로그인 시 ANN fallback) |
| /next-track | POST | X | 유사곡 — 현재 곡과 가까운 상위 10% 중 랜덤 4개 (pgvector ANN) |
| /dynamic-next | POST | X | 다른 곡 — 현재 곡과 먼 상위 30% 중 랜덤 4개 (pgvector ANN) |
| /music-map | POST | JWT | 취향 지도 (MDS + 격자 배치) |
| /top-tracks | GET | X | 저장수 높은 트랙 |
| /favorites | GET/POST/DELETE | JWT | 찜 목록 CRUD |
| /recent-searches | GET/DELETE | JWT | 최근 검색 키워드 |
| /recent-plays | GET | JWT | 최근 재생 트랙 |
| /log-action | POST | X | 유저 행동 로그 |
| /auth/apple/mobile | POST | X | Apple Sign In → JWT 발급 |
| /auth/me | GET | JWT | 유저 정보 |
| /playlist-builder/* | POST/GET | JWT | 플리 빌더 (start/next/save/my-playlists) |
| /interest-profile | POST | JWT | 찜 기반 클러스터링 → user_interest_clusters 갱신 |
| /user-maps | GET/POST | JWT | 뮤직맵 저장/불러오기 |

## DB Tables
- `tracks` — 트랙 메타 (track_key, title, artist, album, cover_image_url, apple_music_id, playlist_count)
- `track_vectors_simgcl` — SimGCL 64d embedding. recommend, music-map, next-track, dynamic-next, interest-profile 전용. HNSW 인덱스 (`vector_cosine_ops`, m=32, ef=200). **기본 ef_search=40 → recommend에서 SET LOCAL hnsw.ef_search=300**
- `track_vectors_clip` — CLIP 512d projected_embedding. smart-search 무드 검색, interest-profile medoid 프로젝션 전용. HNSW 인덱스 (`vector_cosine_ops`)
- `keyword_embeddings` — 검색어 OpenAI 임베딩 캐시 (smart-search 속도 최적화)
- `users` — Apple OAuth 유저
- `user_favorites` — 찜 목록
- `user_action_logs` — 행동 로그 (search/play/favorite/select)
- `user_interest_clusters` — 찜 기반 클러스터 (k, medoids 64d, global_embedding 64d) — recommend Dual Attention 입력
- `user_maps` — 뮤직맵 저장
- `search_keywords` — 자동완성용 키워드 + 검색수
- `user_playlists` — 플리 빌더 저장 플레이리스트

**삭제된 테이블**: `track_vectors_twotower`(미사용), `listening_logs`(미사용), `tmp_cover_url_fix`(임시)

> `track_embeddings`는 현재도 사용 중 — recommend, music-map, dynamic_streaming 모두 이 테이블 사용

## 임베딩 테이블 용도

| 테이블 | 차원 | 모델 | 용도 |
|---|---|---|---|
| `track_vectors_simgcl` | 64d | SimGCL | 트랙 간 음악적 유사도 — recommend, music-map, next/dynamic-next |
| `track_vectors_clip` | 512d | CLIP ProjectionMLP | 텍스트→음악 매핑 — smart-search 무드, interest-profile medoid |

**혼용 금지**: SimGCL을 키워드 검색에, CLIP을 유사곡 검색에 쓰면 결과 엉뚱해짐.

학습 코드 위치: `/Users/eomjoonseo/dynplayer/src/`
- `simgcl/` — SimGCL 학습 → `track_vectors_simgcl.embedding`
- `clip_simgcl/` — CLIP 프로젝션 학습 → `track_vectors_clip.projected_embedding`

## /recommend 아키텍처

1. seed 트랙 SimGCL 임베딩 조회 (`track_vectors_simgcl`, `::text` 캐스팅)
2. `SET LOCAL hnsw.ef_search=300` → ANN 200개 후보 (기본 ef_search=40이라 필수)
3. `user_interest_clusters` 조회 → 없으면 ANN 유사도 순 그대로 반환
4. Dual Attention (SIA × CIA) + Global Pooling 혼합 스코어 (DA 80% + Global 20%)
5. 스코어 내림차순 → `num_recommendations`개 반환 (기본 56)

## Deploy
```bash
# 1. 로컬에서 git push
git push

# 2. NCP 서버 SSH 접속
ssh ncp-dynplayer

# 3. 배포
cd /opt/dynplayer && bash deploy.sh
```

서버 포트: 8889, 도메인: api.caromusic.kr

SSH 설정 (`~/.ssh/config`):
```
Host ncp-dynplayer
    HostName 223.130.128.15
    User root
    IdentityFile ~/.ssh/dynamic-player.pem
    IdentitiesOnly yes
```

> NCP 시리얼 콘솔은 인코딩 깨짐 이슈 있음 → SSH 사용 권장

## .env 필수 변수
DATABASE_URL, SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY,
GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI,
JWT_SECRET, FRONTEND_URL

## pgvector 연산자 규칙

**절대 혼동 금지.**

| 연산자 | 의미 | 필요 인덱스 opclass |
|---|---|---|
| `<=>` | Cosine distance | `vector_cosine_ops` |
| `<#>` | Negative Inner Product | `vector_ip_ops` |
| `<->` | L2 distance | `vector_l2_ops` |

현재 모든 HNSW 인덱스는 `vector_cosine_ops` → **반드시 `<=>` 사용**.
`<#>`로 바꾸면 인덱스 못 타고 풀스캔 → 5~30초 소요.

similarity 공식: `(1 - (embedding <=> query))::float` (cosine distance는 0~1 범위이므로 /2 불필요)

인덱스 확인:
```sql
SELECT indexname, indexdef FROM pg_indexes WHERE tablename IN ('track_vectors_simgcl', 'track_vectors_clip');
```

## 향후 작업

### Two-Tower 스코어 추가 (recommend)
현재 `/recommend`는 `SIA + CIA + Global` 3가지 스코어로 랭킹. 여기에 Two-Tower 스코어 추가 가능:
- API에 `session_track_keys: [String]` 인자 추가 (클라이언트에서 클릭 히스토리 누적해서 전달)
- 백엔드: 시퀀스 → User Tower(`two_tower_best.pt`) → user vec (128d)
- user vec ↔ 후보 `track_vectors_twotower` item vec → TwoTower similarity
- `score = w1*SIA + w2*CIA + w3*Global + w4*TwoTower`
- `track_vectors_twotower` 테이블은 이를 위해 삭제하지 않고 보존

## Notes
- .pt 파일(clip_best.pt, two_tower_best.pt)은 NCP 서버에만 존재
- 로컬에서는 모델 로드 에러 발생 정상 (서버 전용)
- PgBouncer 호환: statement_cache_size=0

---

## Client Apps

### iOS Swift 앱
**위치**: `/Users/eomjoonseo/caro/caro_app_swift/`
- Xcode 프로젝트: `caro_app_swift.xcodeproj`
- 소스: `caro_app_swift/` (git repo: https://github.com/Eoo0m/caro-app-swift)
- Widget 익스텐션: `CaroWidget/`

**주요 파일**:
```
MyApp.swift              # 앱 진입점, AuthStore + MusicPlayerStore 환경 주입
ContentView.swift        # TabView (검색/저장/뮤직맵/프로필) + MiniPlayerView
APIClient.swift          # 모든 API 호출 (base: api.caromusic.kr)
Models.swift             # Track, Playlist, SmartSearchResult, MusicMapResult 등
MusicPlayerStore.swift   # ApplicationMusicPlayer 래퍼, 프리페치 캐시, 곡 전환 감지
AuthStore.swift          # Apple Sign In → JWT 저장 (UserDefaults "jwt")
SearchView.swift         # 검색탭: smartSearch 하나만 호출, favoritedKeys Set으로 favorites 관리
SavedView.swift          # 저장탭: favorites 목록
MusicMapView.swift       # 취향지도탭: music-map API + canvas 렌더링
NowPlayingView.swift     # 재생 상세화면
ProfileView.swift        # 프로필/로그아웃
```

**아키텍처 특이사항**:
- Auth: Apple Sign In → `/auth/apple/mobile` → JWT → `UserDefaults["jwt"]`
- 재생: MusicKit `ApplicationMusicPlayer` (Apple Music 구독 필요)
- 뮤직맵 캐시: `MusicPlayerStore.cachedMapTracks`, 30분 유효 (`isMapCacheValid`)
- Widget → 앱 딥링크: `caro://play?track_key=...&apple_id=...` scheme

**성능 주의**:
- `SearchView`의 `favoritedKeys: Set<String>`으로 favorites를 한 번만 로드 — 카드별 `getFavorites()` 호출 절대 금지
- `smartSearch` 하나만 사용 (`entity_results` + `mood_results` 모두 포함) — `searchSongs` 중복 호출 금지

---

### 웹 프론트엔드
**위치**: `/Users/eomjoonseo/caro/adaptive-music-player/`
- Cloudflare Pages 배포 (바닐라 JS/HTML/CSS, 프레임워크 없음)
- git repo 별도 존재

**주요 파일**:
```
app.html                 # 메인 앱 (검색/저장/뮤직맵 탭)
js/
  main.js                # 탭 전환, createTrackCard, track dots 메뉴 (_openTrackMenu)
  music-map.js           # 취향지도 canvas 렌더링, fill_per_seed 계단식 (10개↓→5, 20개↓→3, 40개↓→2)
  player.js              # MusicKit + playerWidget, _prefetchCache, _fetchAndPlay
  favorites.js           # 찜탭, fav-dots-btn (항상 표시)
  search.js              # 검색 UI
  auth.js                # Apple Sign In (웹)
css/styles.css           # 전체 스타일
```

**뮤직맵 렌더링**:
- canvas 기반 → DOM overlay 좌표계 불일치로 canvas 위에 DOM 버튼 불가
- 초기 카메라: 트랙 평균 좌표(`avgPx/avgPy`)로 중앙 정렬
- 새로고침 시 canvasWrap 숨기지 않고 반투명 블러 오버레이만 표시
