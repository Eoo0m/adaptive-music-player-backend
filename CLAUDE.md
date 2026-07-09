# Dynplayer Backend

## Overview
AI 기반 음악 추천 서비스 백엔드. FastAPI + PostgreSQL(Supabase) + PyTorch 모델 서빙.

## Tech Stack
- **Framework**: FastAPI (Python)
- **DB**: Supabase PostgreSQL + asyncpg (직접 연결, PgBouncer transaction mode)
- **Vector Search**: pgvector + HNSW 인덱스
- **ML Models**: CLIP (키워드→트랙 프로젝션), Two-Tower (세션 기반 추천)
- **Auth**: Google OAuth → JWT (python-jose)
- **Deploy**: NCP Ubuntu, systemd 서비스 (`dynplayer`), `bash deploy.sh`로 배포

## Project Structure
```
main.py              # app 생성, CORS, startup/shutdown, 라우터 등록
database.py          # asyncpg connection pool
models.py            # CLIP, Two-Tower 모델 정의 + 로드 (.pt 파일)
utils.py             # mmr_rerank, OpenAI 임베딩 헬퍼
routes/
  search.py          # /search-songs, /search-by-keyword, /find-similar-tracks
  recommend.py       # /recommend (Two-Tower 세션 추천)
  auth.py            # /auth/google/login, /auth/google/callback, /auth/me
  favorites.py       # /favorites (GET/POST/DELETE)
  home_feed.py       # /home-feed (찜 기반 개인화 피드)
  action_log.py      # /log-action (유저 행동 로그)
sql/                 # DB 스키마, 인덱스, 테이블 정의
experiments/         # HNSW recall 실험 등
```

## Key APIs
| Endpoint | Method | Auth | Description |
|---|---|---|---|
| /search-songs | POST | X | 제목/아티스트 검색 |
| /search-by-keyword | POST | X | 키워드 검색 (CLIP + MMR) |
| /find-similar-tracks | POST | X | 단일 트랙 유사곡 |
| /recommend | POST | X | Two-Tower 세션 추천 |
| /auth/google/login | GET | X | Google OAuth 시작 |
| /auth/google/callback | GET | X | OAuth 콜백 → JWT |
| /auth/me | GET | JWT | 유저 정보 |
| /favorites | GET/POST/DELETE | JWT | 찜 목록 CRUD |
| /home-feed | GET | JWT | 개인화 홈 피드 |
| /log-action | POST | X | 유저 행동 로그 |

## DB Tables
- `track_embeddings` — 트랙 메타 + 벡터 (embedding 64d, projected 512d, itemtower 128d)
- `users` — Google OAuth 유저
- `user_favorites` — 찜 목록
- `user_action_logs` — 유저 행동 로그 (search, select, favorite, play)

## Deploy
```bash
# NCP 서버에서
cd /opt/dynplayer && bash deploy.sh
```
서버 포트: 8889, 도메인: api.dynplayer.win

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
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'track_embeddings';
```

## Notes
- .pt 파일(clip_best.pt, two_tower_best.pt)은 NCP 서버에만 존재
- 로컬에서는 모델 로드 에러 발생 정상 (서버 전용)
- PgBouncer 호환: statement_cache_size=0
