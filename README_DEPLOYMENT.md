# 🚀 NCP Ubuntu 배포 가이드

## 문제 해결: systemctl 변경사항이 반영 안될 때

### 1️⃣ 올바른 재시작 순서

```bash
# 서비스 중지
sudo systemctl stop adaptive-music-player

# 서비스 파일 변경사항 반영 (중요!)
sudo systemctl daemon-reload

# 서비스 시작
sudo systemctl start adaptive-music-player

# 상태 확인
sudo systemctl status adaptive-music-player
```

### 2️⃣ Python 캐시 삭제

변경사항이 반영 안되는 가장 흔한 원인:

```bash
cd /path/to/adaptive-music-player-backend

# __pycache__ 폴더 삭제
find . -type d -name __pycache__ -exec rm -rf {} +

# .pyc 파일 삭제
find . -name "*.pyc" -delete

# 그 다음 서비스 재시작
sudo systemctl restart adaptive-music-player
```

### 3️⃣ 로그 확인 방법

#### A. 실시간 로그 모니터링
```bash
# systemd 로그 (실시간)
sudo journalctl -u adaptive-music-player -f

# 애플리케이션 로그 (실시간)
tail -f /tmp/adaptive-music-player.log
```

#### B. 최근 로그 확인
```bash
# 최근 50줄
sudo journalctl -u adaptive-music-player -n 50

# 특정 시간 이후 로그
sudo journalctl -u adaptive-music-player --since "10 minutes ago"
```

#### C. 에러만 필터링
```bash
sudo journalctl -u adaptive-music-player -p err
```

### 4️⃣ 배포 확인 방법

#### A. 버전 확인
코드 변경 후 실제로 새 버전이 배포되었는지 확인:

```bash
# API 버전 확인
curl https://api.dynplayer.win/

# 또는
curl https://api.dynplayer.win/health
```

응답 예시:
```json
{
  "status": "ok",
  "service": "Adaptive Music Player API",
  "version": "20260123_143052",
  "timestamp": "2026-01-23T14:30:52.123456"
}
```

**버전이 바뀌지 않았다면 = 서비스가 재시작 안된 것**

#### B. 프로세스 확인
```bash
# Python 프로세스 찾기
ps aux | grep main.py

# 프로세스 ID와 시작 시간 확인
ps -p <PID> -o pid,lstart,cmd
```

### 5️⃣ 자동 배포 스크립트 사용

```bash
# 스크립트에 실행 권한 부여
chmod +x deploy.sh

# 스크립트 실행
./deploy.sh
```

또는 한 줄로:

```bash
git pull && find . -type d -name __pycache__ -exec rm -rf {} + && sudo systemctl restart adaptive-music-player && sudo systemctl status adaptive-music-player
```

---

## 🔧 Systemd 서비스 파일 예제

### `/etc/systemd/system/adaptive-music-player.service`

```ini
[Unit]
Description=Adaptive Music Player Backend API
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/adaptive-music-player-backend
Environment="PATH=/home/your-username/adaptive-music-player-backend/venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/home/your-username/adaptive-music-player-backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

# 로그 설정
StandardOutput=journal
StandardError=journal
SyslogIdentifier=adaptive-music-player

[Install]
WantedBy=multi-user.target
```

### 서비스 파일 적용

```bash
# 서비스 파일 복사
sudo cp adaptive-music-player.service /etc/systemd/system/

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable adaptive-music-player
sudo systemctl start adaptive-music-player

# 상태 확인
sudo systemctl status adaptive-music-player
```

---

## 🐛 디버깅: 브라우저 개발자도구에서 로그 확인

### 문제: 백엔드 로그가 브라우저에 안 보임

**원인**: 백엔드 로그는 서버에만 기록됩니다. 브라우저에서는 다음만 확인 가능:
- Network 탭: API 요청/응답
- Console 탭: 프론트엔드 JavaScript 로그

### 해결책: 응답에 디버그 정보 포함

이미 `/search-by-keyword` 엔드포인트에서 `debug` 필드로 반환중:

```json
{
  "results": [...],
  "debug": {
    "keyword": "공부할 때 듣는 재즈",
    "playlists_found": 50,
    "top_playlists": [
      {
        "playlist_id": "...",
        "title": "Chill Jazz for Studying",
        "saves": 12000,
        "similarity": 0.8234,
        "track_count": 45
      }
    ],
    "tracks_recommended": 10,
    "tracks_returned": 10
  }
}
```

**브라우저에서 확인**:
1. 개발자도구 열기 (F12)
2. Network 탭 → `search-by-keyword` 요청 클릭
3. Response 탭에서 `debug` 객체 확인

---

## 📊 로그 출력 예시

변경된 코드로 다음과 같은 로그가 출력됩니다:

```
2026-01-23 14:30:52 - __main__ - INFO - 🔍 Keyword search request: '공부할 때 듣는 재즈' (top_k=200)
2026-01-23 14:30:52 - __main__ - INFO - 📊 Found 50 matching playlists
2026-01-23 14:30:52 - __main__ - INFO - ================================================================================
2026-01-23 14:30:52 - __main__ - INFO - 📋 Top 10 Playlists for keyword: '공부할 때 듣는 재즈'
2026-01-23 14:30:52 - __main__ - INFO - ================================================================================
2026-01-23 14:30:52 - __main__ - INFO - # 1 | Chill Jazz for Studying                           | 유사도: 0.8234 | 저장: 12000  | 트랙:  45
2026-01-23 14:30:52 - __main__ - INFO - # 2 | Study Jazz - Relaxing Piano                       | 유사도: 0.8102 | 저장:  8500  | 트랙:  38
2026-01-23 14:30:52 - __main__ - INFO - # 3 | Smooth Jazz Instrumentals                         | 유사도: 0.7891 | 저장: 15200  | 트랙:  52
...
2026-01-23 14:30:52 - __main__ - INFO - ================================================================================
2026-01-23 14:30:52 - __main__ - INFO -
🎵 Calculating weighted track recommendations...
2026-01-23 14:30:52 - __main__ - INFO - ✅ Recommended 10 tracks
2026-01-23 14:30:52 - __main__ - INFO -
🎼 Final Track Recommendations:
2026-01-23 14:30:52 - __main__ - INFO - --------------------------------------------------------------------------------
2026-01-23 14:30:52 - __main__ - INFO - # 1 | Autumn Leaves                            - Bill Evans Trio
2026-01-23 14:30:52 - __main__ - INFO - # 2 | Blue in Green                            - Miles Davis
2026-01-23 14:30:52 - __main__ - INFO - # 3 | My Favorite Things                       - John Coltrane
...
2026-01-23 14:30:52 - __main__ - INFO - --------------------------------------------------------------------------------
2026-01-23 14:30:52 - __main__ - INFO - ✅ Final selected: 10 tracks
```

---

## 🎯 체크리스트

코드 변경 후 배포 시:

- [ ] Git commit & push
- [ ] NCP 서버에서 `git pull`
- [ ] Python 캐시 삭제 (`__pycache__`, `.pyc`)
- [ ] `sudo systemctl daemon-reload` (서비스 파일 변경 시)
- [ ] `sudo systemctl restart adaptive-music-player`
- [ ] `/health` 엔드포인트로 버전 확인
- [ ] `journalctl` 또는 `/tmp/adaptive-music-player.log`로 로그 확인
- [ ] 브라우저에서 기능 테스트

---

## 🆘 문제 해결

### "Address already in use" 에러
```bash
# 포트 사용중인 프로세스 찾기
sudo lsof -i :8000

# 프로세스 종료
sudo kill -9 <PID>

# 또는 서비스 재시작
sudo systemctl restart adaptive-music-player
```

### 환경변수가 로드 안됨
```bash
# .env 파일 위치 확인
ls -la /path/to/adaptive-music-player-backend/.env

# 서비스 파일에 EnvironmentFile 추가
[Service]
EnvironmentFile=/path/to/adaptive-music-player-backend/.env
```

### 모델 파일 못 찾음
```bash
# 모델 파일 존재 확인
ls -lh /path/to/adaptive-music-player-backend/*.pt

# 권한 확인
sudo chown your-username:your-username *.pt
```
