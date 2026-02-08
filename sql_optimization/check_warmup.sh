#!/bin/bash
# NCP 서버에서 warmup 로그 확인

echo "=== Checking warmup logs ==="
sudo journalctl -u dynplayer-backend -n 100 | grep -A 5 "warmup"

echo ""
echo "=== Recent service status ==="
sudo systemctl status dynplayer-backend | head -20
