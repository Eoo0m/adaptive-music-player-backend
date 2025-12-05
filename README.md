# 🎵 DynPlayer API

대조학습 기반 음악 추천 · 검색 모델


https://dynplayer.win

- 앨범 커버 클릭시 검색창 노출

![video_3x](https://github.com/user-attachments/assets/c6fdbc9d-1f6a-4fa3-aa49-c39b7634b802)



## architecture

<img width="391" height="224" alt="image" src="https://github.com/user-attachments/assets/8bb8a8ef-581c-42d1-b88a-ce78f3ede42c" />


## Spotify OAuth
	•	/login
	•	Spotify OAuth 로그인 시작
	•	/callback
	•	Code → Access Token, Refresh Token 교환
	•	/refresh_token
	•	Refresh Token으로 Access Token 재발급


## 검색 기능

### /search-songs — 제목 기반 검색
	•	입력: query (곡 제목)
	•	Supabase 함수 search_tracks_by_title 호출
	•	유사 제목 10개 반환



### /search-by-keyword — 키워드 기반 벡터 검색
	•	OpenAI text-embedding-3-large → 3072차원 텍스트 임베딩 생성
	•	playlist_clip_model 로 텍스트 → playlist 공간(512차원) projection
	•	Supabase 함수 match_playlist_embeddings으로 가장 유사한 playlist TOP 50 조회
	•	playlist 내 트랙들을 similarity × frequency 기반으로 랭킹
	•	상위 10개 곡 반환



### /find-spotify-tracks — 추천 결과 Spotify 매핑
	•	추천된 트랙(title + artist) → Spotify Search API로 실제 트랙 매핑
	•	Spotify track object, URI, preview_url 반환
	•	음원 재생을 위한 필수 단계


## 추천 기능

### /recommend — 특정 트랙 기반 추천
	•	입력: track_key
	•	Supabase 함수 match_tracks_by_key
→ pgvector 코사인 유사도로 가장 가까운 embedding N개 추천
	•	결과는 /find-spotify-tracks 로 Spotify 트랙 정보 매핑하여 재생 가능하게 처리



## 🧠 모델 구조

✔ playlist_clip_model
	•	Caption(text embedding 3072) → playlist embedding 공간(512) projection
	•	Playlist embedding(256→512 projection)과 cosine similarity로 검색
	•	Residual block + GELU + LayerNorm 기반 MLP



## 🗄 DB 구조 (Supabase + pgvector)

### playlists 테이블: 검색 쿼리와 비교를 위해 투영된 임베딩
	•	playlist_id (PK)
	•	track_ids (JSON array)
	•	embedding (vector 512)

### track_embeddings 테이블: 대조학습으로 생성된 트랙 임베딩
	•	track_key
	•	title, artist, album
	•	embedding (vector 256)
	•	pos_count 



# Experiment
## 텍스트 검색으로 플레이리스트를 찾아도되는데 굳이 임베딩을 학습하는 이유
→ 텍스트로 나타낼 수 없는 음악적 특성 반영

### **query: 외힙 (Top 10 Playlists)**

| **순위** | **점수** | **플레이리스트 이름** | **Playlist ID** |
| --- | --- | --- | --- |
| 1 | **0.6885** | Best Rap Songs of 2019 | 0mPHjoMfKNHbVzo2U4LGqK |
| 2 | 0.6835 | Pop Smoke Radio | 37i9dQZF1E4pSe4zcPlJan |
| 3 | 0.6806 | Billboards R&B/Hip-Hop Top 100 | 3qVzuSvpTZZ7EVlmn1gQ5r |
| 4 | 0.6756 | 외힙 입문하기 (Hip Hop Starter Pack) | 5zUqj3NUgAb2voGZHDMwlc |
| 5 | 0.6730 | 운동할때 듣는 외힙 | 56A97TgFcRRLzWCCcVB1Vs |
| 6 | 0.6717 | Hip Hop Hits / Pop Rap Mix | 5oUjcXbrveXjPvBy9udX76 |
| 7 | 0.6637 | 외국힙합갤러리 | 2Y9oAbVygOt0QA2EqO5Hho |
| 8 | 0.6633 | 좆되게 힙한 팝 | 7A4axQQsLUwtyKrcP5oTwJ |
| 9 | 0.6575 | 카리나가 좋아하는 외힙 플리 | 63AmXMQbA09OwYzs5PiOty |
| 10 | 0.6575 | Unknown | 0vIm6DBpjcaYAuwGQCQda4 |

---

### **query: 발라드 (Top 10 Playlists)**

| **순위** | **점수** | **플레이리스트 이름** | **Playlist ID** |
| --- | --- | --- | --- |
| 1 | **0.7666** | 노래방에서 부르기 딱!! 고음 발라드 | 2ohiol7bCGozbMjwvsMLoM |
| 2 | 0.7647 | 노래방 작살내는 차트(남자버전) | 6tLL7auLLu0vUjb3Q3w93z |
| 3 | 0.7629 | 비오는날 감성발라드 | 4SvgdVWTQzvsuYGw1hbXWG |
| 4 | 0.7616 | 2000년대 발라드 명곡 | 13t0ABFN6fTdSyz3ISPlxN |
| 5 | 0.7490 | 밥만 잘 먹더라 | 4lkkOJ8ZugaS7r6bOAWBhZ |
| 6 | 0.7464 | Unknown | 5NI4eSsPnzaFjrZEl1iw6X |
| 7 | 0.7461 | 가을에 꺼내듣기 좋은 발라드 | 6c9RSN6hEHAPdWbbPzBkpE |
| 8 | 0.7422 | 극 고음 발라드 개띵곡 | 1Eh3lW7cG5wJNE7cuJ48tY |
| 9 | 0.7417 | 술 마실 때 틀려고 만든 플리 | 2tBCPWD3hyfJZoSSPcxhDS |
| 10 | 0.7399 | 엠씨더맥스 플레이리스트 | 3ENgMmp2wXEkkR90Xmq6JZ |

---

### **query: cozy pop (Top 10 Playlists)**

| **순위** | **점수** | **플레이리스트 이름** | **Playlist ID** |
| --- | --- | --- | --- |
| 1 | **0.6203** | Cozy Pop Mix | 37i9dQZF1EIgUNZWgFoh9c |
| 2 | 0.6118 | Soft aesthetic songs ♡ | 7snUniDZ1aZhbbGJvw6KlF |
| 3 | 0.5880 | cute romantic songs for fake scenarios | 4vSobDIRZvZk9Hfx8Yk36z |
| 4 | 0.5838 | love songs <3 | 0nevMyChAKVxX6sGwby5A6 |
| 5 | 0.5816 | most love & romantic songs ever | 5NxUcSM8u89MIbdT9Cq78C |
| 6 | 0.5695 | my love mine all mine vibes | 1uAGveL23B7n9q1VR21ZTZ |
| 7 | 0.5662 | I collect these romantic songs just for you | 0COrvGg2BG9O66FLZscNxf |
| 8 | 0.5617 | ✩ — 잠 ⸰ֺ⭑ | 3RFUnP6lbQowqZUD0JqCbO |
| 9 | 0.5558 | calm songs to relax my anxietyy | 3l6b0zuXjgyPxLK6PIAqED |
| 10 | 0.5522 | Winter vibes ⋆⁺₊❅. | 15lesHzhv9X0fbEWztEYon |

---
-> 텍스트의 의미가 유사한 플레이리스트만 검색되는 것이 아니라 장르/무드/특성 등이 유사한 음악이 검색되는 것을 볼 수 있다.
