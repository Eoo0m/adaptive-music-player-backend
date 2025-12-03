# 🎵 DynPlayer API

대조학습 기반 음악 추천 · 검색 모델


https://dynplayer.win

![video_3x](https://github.com/user-attachments/assets/c6fdbc9d-1f6a-4fa3-aa49-c39b7634b802)

- 앨범 커버 클릭시 검색창 노출
- 너무 많은 클릭시 스포티파이 오류 발생

<img width="921" height="499" alt="Screenshot 2025-11-30 at 8 30 31 PM" src="https://github.com/user-attachments/assets/3fc00617-038a-496e-a4a4-00741dadb91a" />


## 🔑 Spotify OAuth
	•	/login
	•	Spotify OAuth 로그인 시작
	•	/callback
	•	Code → Access Token, Refresh Token 교환
	•	/refresh_token
	•	Refresh Token으로 Access Token 재발급


## 🔍 검색 기능

### 🔎 /search-songs — 제목 기반 검색
	•	입력: query (곡 제목)
	•	Supabase 함수 search_tracks_by_title 호출
	•	유사 제목 10개 반환



### 🧠 /search-by-keyword — 키워드 기반 벡터 검색
	•	OpenAI text-embedding-3-large → 3072차원 텍스트 임베딩 생성
	•	playlist_clip_model 로 텍스트 → playlist 공간(512차원) projection
	•	Supabase 함수 match_playlist_embeddings으로 가장 유사한 playlist TOP 50 조회
	•	playlist 내 트랙들을 similarity × frequency 기반으로 랭킹
	•	상위 10개 곡 반환



### 🎵 /find-spotify-tracks — 추천 결과 Spotify 매핑
	•	추천된 트랙(title + artist) → Spotify Search API로 실제 트랙 매핑
	•	Spotify track object, URI, preview_url 반환
	•	음원 재생을 위한 필수 단계


## 🎧 추천 기능

### 🎧 /recommend — 특정 트랙 기반 추천
	•	입력: track_key
	•	Supabase 함수 match_tracks_by_key
→ pgvector 코사인 유사도로 가장 가까운 embedding N개 추천
	•	결과는 /find-spotify-tracks 로 Spotify 트랙 정보 매핑하여 재생 가능하게 처리



## 📡 Logging

📝 /log-listening — 사용자 청취 기록 저장



## 🧠 모델 구조

✔ playlist_clip_model
	•	Caption(text embedding 3072) → playlist embedding 공간(512) projection
	•	Playlist embedding(256→512 projection)과 cosine similarity로 검색
	•	Residual block + GELU + LayerNorm 기반 MLP



## 🗄 DB 구조 (Supabase + pgvector)

✔ playlists 테이블
	•	playlist_id (PK)
	•	track_ids (JSON array)
	•	embedding (vector 512) ← playlist projector 출력

✔ track_embeddings 테이블
	•	track_key
	•	title, artist, album
	•	embedding (vector 256) ← 음악 모델 embedding
	•	pos_count (playlist 허브곡 조정용)
