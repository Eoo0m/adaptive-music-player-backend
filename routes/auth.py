import os
import logging
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
from jose import jwt, JWTError
from datetime import datetime, timedelta

from database import get_db_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth")

# Google OAuth 설정
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24 * 7  # 7일

# 프론트엔드 URL (로그인 후 리다이렉트)
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://dynplayer.win")


def create_jwt_token(user_id: str, email: str) -> str:
    """JWT 토큰 생성"""
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(request: Request) -> dict:
    """Authorization 헤더에서 JWT를 검증하고 유저 정보 반환"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": payload["sub"], "email": payload["email"]}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.get("/google/login")
async def google_login():
    """Google OAuth 인증 페이지로 리다이렉트"""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    }
    url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    return RedirectResponse(url=url)


@router.get("/google/callback")
async def google_callback(code: str):
    """Google OAuth 콜백 - 토큰 교환 → 유저 생성/조회 → JWT 발급"""
    try:
        # 1. code로 access_token 교환
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
            )
            token_data = token_response.json()

        if "access_token" not in token_data:
            logger.error(f"Google token exchange failed: {token_data}")
            raise HTTPException(status_code=400, detail="Google authentication failed")

        # 2. access_token으로 유저 정보 가져오기
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
            )
            google_user = user_response.json()

        google_id = google_user["id"]
        email = google_user.get("email", "")
        display_name = google_user.get("name", "")
        profile_image_url = google_user.get("picture", "")

        logger.info(f"Google login: {email} ({google_id})")

        # 3. DB에 유저 upsert
        pool = await get_db_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO users (google_id, email, display_name, profile_image_url)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (google_id) DO UPDATE SET
                email = EXCLUDED.email,
                display_name = EXCLUDED.display_name,
                profile_image_url = EXCLUDED.profile_image_url,
                updated_at = NOW()
            RETURNING id::text
            """,
            google_id, email, display_name, profile_image_url
        )

        user_id = row["id"]

        # 4. JWT 발급 → 프론트엔드로 리다이렉트
        token = create_jwt_token(user_id, email)

        redirect_url = f"{FRONTEND_URL}?token={token}"
        return RedirectResponse(url=redirect_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google OAuth error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """현재 로그인 유저 정보 조회"""
    pool = await get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT id::text, google_id, email, display_name, profile_image_url, created_at
        FROM users
        WHERE id = $1::uuid
        """,
        user["user_id"]
    )

    if not row:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": row["id"],
        "email": row["email"],
        "display_name": row["display_name"],
        "profile_image_url": row["profile_image_url"],
        "created_at": row["created_at"].isoformat(),
    }
