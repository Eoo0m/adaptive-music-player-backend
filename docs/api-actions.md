# Action APIs — 플레이리스트 / 코멘트

> 유저가 직접 저장·관리하는 데이터에 관한 API 모음.
> 모든 엔드포인트는 `Authorization: Bearer <jwt>` 헤더 필요 (예외 명시).

---

## 플레이리스트

### GET `/playlist-builder/my-playlists`

저장된 플레이리스트 목록 + 트랙 전체 데이터 반환.

**Auth**: JWT (없으면 guest 계정으로 fallback)

**Response**
```json
{
  "playlists": [
    {
      "id": 1,
      "name": "내 플레이리스트",
      "created_at": "2026-11-23T12:00:00",
      "tracks": [
        {
          "track_key": "abc123",
          "track_name": "곡 제목",
          "artist": "아티스트",
          "album": "앨범",
          "cover_image_url": "https://...",
          "playlist_count": 100,
          "apple_music_id": "12345"
        }
      ]
    }
  ]
}
```

---

### POST `/playlist-builder/playlist/create`

빈 플레이리스트 새로 생성.

**Auth**: JWT 필수

**Request**
```json
{ "name": "플레이리스트 이름" }
```

**Response**
```json
{ "success": true, "playlist_id": 42, "name": "...", "created_at": "..." }
```

---

### POST `/playlist-builder/playlist/{playlist_id}/add-track`

기존 플레이리스트에 단일 곡 추가. 이미 담긴 곡은 무시(중복 없음).

**Auth**: JWT 필수

**Request**
```json
{ "track_key": "abc123" }
```

**Response**
```json
{ "success": true, "track_count": 5 }
```

---

### POST `/playlist-builder/save`

플레이리스트 빌더(빌드2)에서 완성한 플레이리스트 저장.
`track_keys` 배열을 그대로 저장함.

**Auth**: JWT (없으면 guest)

**Request**
```json
{ "name": "플레이리스트 이름", "track_keys": ["key1", "key2"] }
```

**Response**
```json
{ "success": true, "playlist_id": 42, "created_at": "..." }
```

---

### DELETE `/playlist-builder/playlist/{playlist_id}`

플레이리스트 삭제. 본인 소유인 경우만 삭제됨.

**Auth**: JWT 필수

**Response**
```json
{ "success": true }
```

---

## 코멘트

### POST `/comments`

트랙에 코멘트 추가.

**Auth**: JWT 필수

**Request**
```json
{ "track_key": "abc123", "text": "이 곡 진짜 좋다" }
```

**Response**
```json
{ "success": true, "comment_id": 7, "created_at": "2026-11-23T20:00:00" }
```

---

### GET `/comments/my`

내가 남긴 코멘트 목록 (최신순). 트랙 메타(title, artist, cover_image_url, apple_music_id) 포함.

**Auth**: JWT 필수

**Response**
```json
{
  "comments": [
    {
      "id": 7,
      "track_key": "abc123",
      "text": "이 곡 진짜 좋다",
      "created_at": "2026-11-23T20:00:00",
      "title": "곡 제목",
      "artist": "아티스트",
      "cover_image_url": "https://...",
      "apple_music_id": "12345"
    }
  ]
}
```

---

### DELETE `/comments/{comment_id}`

코멘트 삭제. 본인 소유인 경우만 삭제됨.

**Auth**: JWT 필수

**Response**
```json
{ "success": true }
```

---

## DB 테이블

### `user_playlists`

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | SERIAL PK | |
| `user_id` | UUID FK→users | |
| `name` | TEXT | 플레이리스트 이름 |
| `track_keys` | TEXT[] | 순서 있는 트랙 키 배열 |
| `created_at` | TIMESTAMPTZ | |

### `track_comments`

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `id` | SERIAL PK | |
| `user_id` | UUID FK→users CASCADE | |
| `track_key` | TEXT | |
| `text` | TEXT | 코멘트 본문 |
| `created_at` | TIMESTAMPTZ | |

인덱스: `idx_track_comments_user_id`, `idx_track_comments_track_key`

---

## 관련 파일

| 파일 | 역할 |
|---|---|
| `routes/playlist_builder.py` | 플레이리스트 CRUD + 빌더 로직 |
| `routes/comments.py` | 코멘트 CRUD |
