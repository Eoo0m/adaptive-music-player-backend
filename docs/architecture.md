# Caro 백엔드 아키텍처

## API 명세

### 검색
| Endpoint | Method | Auth | 설명 | 주요 파라미터 |
|---|---|---|---|---|
| `/search-autocomplete` | GET | X | 타이핑 중 실시간 키워드 제안 (prefix + trigram + 인기도) | `q` |
| `/search-songs` | POST | X | 곡명/아티스트 텍스트 매칭 (pg_trgm) | `query` |
| `/smart-search` | POST | X | 의도 분류 → 엔티티 결과 + 무드 결과 동시 반환 | `query`, `entity_limit`, `mood_top_k` |
| `/find-similar-tracks` | POST | X | 단일 트랙 기반 유사곡 (SimGCL 임베딩) | `track_key`, `n` |

### 추천
| Endpoint | Method | Auth | 설명 | 주요 파라미터 |
|---|---|---|---|---|
| `/recommend` | POST | X | 여러 트랙 키 기반 Two-Tower 세션 추천 | `track_keys`, `num_recommendations` |
| `/dynamic-next` | POST | X | 다음 곡 선택. `listened_half=true`이면 유사곡, false이면 먼 곡 | `track_key`, `listened_half`, `candidate_track_keys` |
| `/home-feed-v2` | GET | JWT | 인기곡 + 유사곡 + 계절 트랙 혼합 피드 | - |
| `/top-tracks` | GET | X | playlist_count 기준 인기 트랙 | `limit` |

### 취향 지도
| Endpoint | Method | Auth | 설명 | 주요 파라미터 |
|---|---|---|---|---|
| `/music-map` | POST | JWT | 찜한 곡 기반 MDS 취향 지도. 격자 배치 + 임베딩 반환 | `track_keys`, `favorite_keys`, `fill_per_seed` |

### 사용자
| Endpoint | Method | Auth | 설명 |
|---|---|---|---|
| `/auth/apple/mobile` | POST | X | Apple Sign In → JWT 발급 |
| `/auth/me` | GET | JWT | 현재 로그인 유저 정보 |
| `/favorites` | GET/POST/DELETE | JWT | 찜 목록 CRUD |
| `/recent-plays` | GET | JWT | 최근 재생 트랙 목록 |
| `/recent-searches` | GET/DELETE | JWT | 최근 검색 키워드 |
| `/log-action` | POST | X | 행동 로그 저장 (search 타입은 search_keywords 카운트도 증가) |

### 플레이리스트 빌더
| Endpoint | Method | Auth | 설명 |
|---|---|---|---|
| `/playlist-builder/start` | POST | JWT | 키워드 기반 후보곡 첫 세트 반환 |
| `/playlist-builder/next` | POST | JWT | 선택 트랙 기반 다음 후보 반환 |
| `/playlist-builder/save` | POST | JWT | 플레이리스트 저장 |
| `/playlist-builder/my-playlists` | GET | JWT | 내 플레이리스트 목록 |
| `/playlist-builder/playlist/{id}` | DELETE | JWT | 플레이리스트 삭제 |

---

## DB 테이블

| 테이블 | 주요 컬럼 | 설명 |
|---|---|---|
| `tracks` | `track_key`, `title`, `artist`, `album`, `cover_image_url`, `apple_music_id`, `playlist_count` | 트랙 메타. `playlist_count` = 저장수 (인기도 기준) |
| `track_embeddings` | `track_key`, `embedding`(64d), `projected_embedding`(512d), `itemtower_embedding`(128d) | 모델별 임베딩 통합 테이블. HNSW 인덱스 (`vector_cosine_ops`) |
| `track_vectors_simgcl` | `track_key`, `embedding`(64d) | SimGCL 임베딩 단독 테이블. find-similar-tracks, music-map 전용 |
| `users` | `id`, `email`, `name` | Apple OAuth 유저 |
| `user_favorites` | `user_id`, `track_key` | 찜 목록. 취향지도 seed 데이터 |
| `user_action_logs` | `session_id`, `user_id`, `action_type`, `search_query`, `selected_track_key`, `created_at` | 행동 로그. action_type: search/play/favorite/select |
| `search_keywords` | `keyword`, `search_count` | 자동완성용. 곡명/아티스트 초기 seed + 검색마다 카운트 증가 |
| `user_playlists` | `user_id`, `name`, `track_keys[]` | 플리 빌더로 저장한 플레이리스트 |

---

## ML 모델 구조

### 학습 데이터
- 플레이리스트 26,096개 / 트랙 81,699개
- 플레이리스트-트랙 상호작용 그래프 기반
- LOO(Leave-One-Out) 평가: 각 플레이리스트에서 1개 트랙을 test로 보유

### 모델 1: SimGCL
**목적**: 트랙 간 음악적 유사도 임베딩 생성

```
플레이리스트-트랙 이분 그래프
    → LightGCN 전파 (3 layers)
    → 노이즈 증강 (Uniform[-0.1, 0.1])
    → BPR Loss + InfoNCE Loss (λ=0.1, τ=0.2)
    → 트랙 임베딩 64d
```

- 출력: `track_embeddings.embedding` (64d)
- 용도: `/find-similar-tracks`, `/music-map`

### 모델 2: CLIP (ProjectionMLP)
**목적**: 텍스트 키워드 → 음악 의미 공간 매핑

```
플레이리스트 태그 → OpenAI text-embedding-3-large (3072d)
        ↕ InfoNCE Loss (τ=0.07, 가중치: log(saves+1))
SimGCL 플레이리스트 임베딩 (64d) → ProjectionMLP → 512d
```

- ProjectionMLP: `input → 2048 (3블록, GELU + LayerNorm + 잔차) → 512d`
- 학습 데이터: 플레이리스트별 태그 목록 (각 태그가 독립 학습 쌍)
- 출력: `track_embeddings.projected_embedding` (512d)
- 용도: `/smart-search` (무드 검색), `/search-by-keyword`

### 모델 3: Two-Tower
**목적**: 플레이리스트 시퀀스 기반 세션 추천

```
User Tower:  트랙 시퀀스 (N × 64d)
             → Linear (64→128) + Positional Encoding
             → Transformer Encoder (2 layers, 4 heads)
             → CLS 토큰 집계 → 128d

Item Tower:  트랙 임베딩 (64d)
             → MLP (64→256→128, GELU + Dropout 0.1)
             → 128d

InfoNCE Loss (τ=0.07, in-batch negatives, 가중치: log(saves+1))
```

- 출력: `track_embeddings.itemtower_embedding` (128d)
- 용도: `/recommend`

### 모델 연쇄 구조

```
playlist_tracks 데이터
        ↓
   [SimGCL]  → 64d 트랙/플리 임베딩
        ↓
 [Two-Tower] → 128d 아이템 임베딩 (SimGCL 64d 입력)
        ↓
    [CLIP]   → 512d 프로젝션 (SimGCL 플리 임베딩 + 태그 텍스트)
```

### 임베딩 혼용 금지
| 임베딩 | 올바른 용도 | 잘못된 용도 |
|---|---|---|
| `embedding` (64d, SimGCL) | 유사곡 검색, 취향지도 | 키워드 검색 |
| `projected_embedding` (512d, CLIP) | 키워드/분위기 검색 | 유사곡 검색 |
| `itemtower_embedding` (128d, Two-Tower) | 세션 기반 추천 | 유사곡 검색 |

학습 코드: `/Users/eomjoonseo/caro/ml_pipeline/src/`
- `simgcl/` → `embedding`
- `clip_simgcl/` → `projected_embedding`
- `two_tower/` → `itemtower_embedding`
