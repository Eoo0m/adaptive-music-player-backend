# 🎵 DynPlayer API

대조학습 기반 음악 추천 · 검색 모델


https://dynplayer.win

![video_3x](https://github.com/user-attachments/assets/c6fdbc9d-1f6a-4fa3-aa49-c39b7634b802)

- 앨범 커버 클릭시 검색창 노출
- 처음 키워드 검색시 5-10초 소요
- 너무 많은 클릭시 스포티파이 오류 발생




<img width="1" height="1" alt="image" src="https://github.com/user-attachments/assets/f6dc72d3-41de-4016-b63a-bf907b73e4a9" />



### 🔑 Spotify OAuth
	•	Spotify Login
	•	Access Token & Refresh Token 발급

### 🔍 검색 기능
	•	제목 검색 (/search-songs)
	•	키워드 기반 벡터 유사도 검색 (/search-by-keyword)

### 🎧 추천 기능
	•	트랙 벡터 기반 추천 (/recommend)
	•	추천 결과 → Spotify Track 매핑 (/find-spotify-tracks)

### 📡 Logging
	•	유저 재생 기록 저장 (/log-listening)

### 🧠 멀티모달 CLIP 모델
	•	Title(3072D OpenAI embedding) → 512D
	•	Track(256D) → 512D
	•	Multi-head Projection MLP
	•	Cosine similarity 기반 추천
