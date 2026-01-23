#!/bin/bash

# Adaptive Music Player Backend - Deployment Script
# NCP Ubuntu에서 systemctl 서비스 재시작 스크립트

set -e  # 에러 발생시 중단

SERVICE_NAME="adaptive-music-player"  # 실제 서비스 이름으로 변경하세요
APP_DIR="/path/to/adaptive-music-player-backend"  # 실제 경로로 변경하세요

echo "🚀 Starting deployment..."

# 1. 현재 디렉토리로 이동
cd "$APP_DIR"

# 2. Git pull (최신 코드 가져오기)
echo "📥 Pulling latest code from git..."
git pull origin main

# 3. Python 캐시 삭제
echo "🗑️  Cleaning Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 4. 가상환경 활성화 (필요시)
# source venv/bin/activate

# 5. 의존성 업데이트 (필요시)
# pip install -r requirements.txt

# 6. Systemd 서비스 재시작
echo "🔄 Restarting systemd service..."
sudo systemctl stop "$SERVICE_NAME"
sudo systemctl daemon-reload
sudo systemctl start "$SERVICE_NAME"

# 7. 서비스 상태 확인
echo "✅ Checking service status..."
sleep 2
sudo systemctl status "$SERVICE_NAME" --no-pager

# 8. 로그 확인
echo ""
echo "📋 Recent logs:"
sudo journalctl -u "$SERVICE_NAME" -n 20 --no-pager

echo ""
echo "✨ Deployment completed!"
echo "🔍 Check logs with: sudo journalctl -u $SERVICE_NAME -f"
