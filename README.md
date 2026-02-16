# 🎵 DynPlayer API

대조학습 기반 음악 추천 · 검색 모델


https://dynplayer.win

- 앨범 커버 클릭시 검색창 노출
- 음악 재생 개발자 id로만 가능, 재생없이 추천 클릭!

![video_3x](https://github.com/user-attachments/assets/c6fdbc9d-1f6a-4fa3-aa49-c39b7634b802)



# **engagement를 어떻게 올릴 것인가?**

## **UI**

- 재생권한이 없기에, 첫 곡을 검색 이후 그 곡과 유사한 곡을 계속해서 디깅할 수 있는 구조
    - 특정 곡을 seed로 하여 embedding space 상에서 인접한 곡들을 순차적으로 탐색하는 digging UX
    - 사용자가 그래프를 따라 탐색한다는 인식을 줄 수 있도록 transition을 연결감 있게 설계
- 새로운 곡 등장 시 애니메이션 효과
    - 추천 결과가 단절적으로 나타나는 것이 아니라, 이전 곡과의 유사도 기반 연결로 등장한다는 인식을 제공
    - discovery 과정에서의 몰입감 및 탐색 지속 시간 증가 목적

---

## **알고리즘**

### **track**

- 특정 트랙을 검색시에 그 곡과 유사한 곡 15개를 선정하여 제시
- 이때 트랙의 임베딩:
    - playlist 내 co-occurrence를 positive set으로 하여 multipositive InfoNCE를 활용하여 학습
    - 동일 플레이리스트 내 공동 출현 확률 분포를 embedding 공간 거리로 근사하는 것이 목적
- 목적:
    - 특정 곡을 기준으로 동일 맥락에 존재하는 곡 집단을 retrieval 가능하도록 representation 학습
- 문제점: 인기곡(Hub) 쏠림 현상
    - playlist 등장 빈도가 높은 인기곡이 비인기곡에 비해 평균적인 유사도가 너무 높아 쏠림 현상 발생
- 해결:
    - embedding space density regularization 목적의 uniformity loss를 proxy로 추가
    - representation spread 확보 및 hubness 완화
    
    → recall, ndcg 상승, 분포에서 쏠림현상 완화

### **메트릭**

- playlist completion:
    - 각 플레이리스트 내에서 80퍼센트의 곡만을 이용하여 학습 후, 나머지 20퍼센트를 retrieval로 복원
    - leave-k-out split 기반 평가
    - metric: Recall, NDCG
- linear evaluation:
    - 학습된 embedding을 freeze한 뒤, 선형 projection만을 이용하여 Spotify 기준 50가지 장르 classification 수행
    - embedding의 semantic separability 평가 목적
    - metric: Genre prediction Top-5 accuracy
- co-occurrence rate:
    - 플레이리스트 중 80퍼센트만을 이용하여 학습 후, 20퍼센트 구간에서의 query-positive pair retrieval rate 측정
    - pairwise co-save likelihood 복원 성능 평가
    - metric: Recall, NDCG

---

### **keyword**

- 키워드 검색시 쿼리와 유사한 플레이리스트 100개 추출하여 유사도 × 등장 빈도로 트랙 추천
    - playlist title text embedding(OpenAI embedding large) 활용
    - playlist title ↔ playlist track set 간 multimodal contrastive learning 구조로 정렬
    - text semantic과 track co-occurrence representation alignment 목적
### **메트릭**

- 플레이리스트 타이틀 임베딩으로 플레이리스트 검색
    - text query → playlist retrieval 성능 평가
    - semantic title understanding 및 playlist mapping 정확도 측정
    - metric: Recall, NDCG

---
### Ongoing Experiments
	•	Multimodal Track Embedding
Audio(30s preview) + Lyrics + MF(Co-occurrence) concat 기반 임베딩 학습 진행중
(약 80K 트랙 수집 및 전처리 단계)
	•	Graph Embedding Extension
Playlist–Track bipartite graph 기반 임베딩 추가 실험
	•	Retrieval → Ranking Pipeline
Retrieval 후보군에 대해 embedding fusion 기반 ranking 모델 설계 및 실험 진행중
	•	Session-based Recommendation
사용자 세션 시퀀스 기반 next-track 추천 모델 실험 진행중



---
## architecture

<img width="391" height="224" alt="image" src="https://github.com/user-attachments/assets/8bb8a8ef-581c-42d1-b88a-ce78f3ede42c" />



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


### clip_projection
	•	Caption(text embedding 3072) → embedding 공간으로 projection하여 playlist caption - playlist embedding clip 학습
	•	Residual block + GELU + LayerNorm 기반 MLP



## DB 구조 (Supabase + pgvector)

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

## 평가방식

기본 데이터셋

```python
# playlist_id |track_ids
# pl_001      |track_D|track_B|track_C|track_A|track_E
# pl_002      |track_Z|track_F|track_G|track_B|track_X
# pl_003      |track_A|track_C|track_H|track_I|track_J

# track_id |positive_track_ids
track_A  : track_C|track_E
track_B  : track_C|track_D|track_G|track_X
track_C  : track_A|track_B|track_H

```

플레이리스트에서 20퍼센트 제거 후 트랙별 이웃 재구성

```python
# playlist_id | track_ids
# pl_001      |track_D|track_B|track_C|track_A|
# pl_002      |track_Z|track_F|track_G|track_B|
# pl_003      |track_A|track_C|track_H|track_I|

# track_id |positive_track_ids
track_A  : track_C
track_B  : track_C|track_D|track_G
track_C  : track_A|track_B|track_H

```

이과정에서 train에 없는 데이터 생길경우 제거.

## uniformity loss 적용

$$
\mathcal{L}_{\text{uni}} =
\log \mathbb{E}_{x,y}\left[ e^{-t \|x - y\|^2} \right]
$$


많은 벡터가 가까이 몰리면 평균값이 크게 증가 → uniformity loss 커짐
벡터들이 hypersphere에 균일하게 퍼지도록 규제함.


<img width="528" height="229" alt="image" src="https://github.com/user-attachments/assets/855759f6-1c1f-40d0-8a05-d6be903fe2a9" />

<img width="557" height="228" alt="image" src="https://github.com/user-attachments/assets/5952fd86-83ab-46ad-b9a6-2b198b29c2ba" />

기존에는 인기곡이 경우 다른 트랙과의 강한 유사도를 보였음.
-> uniformity loss 적용시, 인기곡과 비인기곡의 차이 감소, 성능도 상승




## 텍스트 검색으로 플레이리스트를 찾아도되는데 굳이 임베딩을 학습하는 이유?
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
