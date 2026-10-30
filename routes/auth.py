import os
import logging
import time
import json
from urllib.parse import urlencode
from functools import lru_cache

import httpx
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse, JSONResponse
from jose import jwt, JWTError, jwk
from jose.utils import base64url_decode
from cryptography.hazmat.primitives.serialization import load_pem_private_key
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

# Spotify OAuth 설정
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

# Apple 설정
APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID")
APPLE_MUSICKIT_KEY_ID = os.getenv("APPLE_MUSICKIT_KEY_ID")
APPLE_MUSICKIT_KEY_PATH = os.getenv("APPLE_MUSICKIT_KEY_PATH")
APPLE_SIWA_KEY_ID = os.getenv("APPLE_SIWA_KEY_ID")
APPLE_SIWA_KEY_PATH = os.getenv("APPLE_SIWA_KEY_PATH")
APPLE_SERVICES_ID = os.getenv("APPLE_SERVICES_ID")
APPLE_REDIRECT_URI = os.getenv("APPLE_REDIRECT_URI")

# 프론트엔드 URL (로그인 후 리다이렉트)
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://caromusic.kr")


def create_jwt_token(user_id: str, email: str, login_type: str = "google") -> str:
    """JWT 토큰 생성"""
    payload = {
        "sub": user_id,
        "email": email,
        "login_type": login_type,
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
        return {"user_id": payload["sub"], "email": payload["email"], "login_type": payload.get("login_type", "google")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def get_optional_user(request: Request) -> dict | None:
    """인증 선택적 — 토큰 없으면 None 반환"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {"user_id": payload["sub"], "email": payload["email"], "login_type": payload.get("login_type", "google")}
    except JWTError:
        return None


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
        token = create_jwt_token(user_id, email, login_type="google")

        redirect_url = f"{FRONTEND_URL}?token={token}"
        return RedirectResponse(url=redirect_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google OAuth error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@router.get("/spotify/login")
async def spotify_login():
    """Spotify OAuth 인증 페이지로 리다이렉트"""
    params = {
        "client_id": SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "scope": "user-read-email user-read-private",
    }
    url = f"https://accounts.spotify.com/authorize?{urlencode(params)}"
    return RedirectResponse(url=url)


@router.get("/spotify/callback")
async def spotify_callback(code: str):
    """Spotify OAuth 콜백 - 토큰 교환 → 유저 생성/조회 → JWT 발급"""
    import base64

    try:
        # 1. code로 access_token 교환
        credentials = base64.b64encode(
            f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()
        ).decode()

        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://accounts.spotify.com/api/token",
                data={
                    "code": code,
                    "redirect_uri": SPOTIFY_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            token_data = token_response.json()

        if "access_token" not in token_data:
            logger.error(f"Spotify token exchange failed: {token_data}")
            raise HTTPException(status_code=400, detail="Spotify authentication failed")

        # 2. access_token으로 유저 정보 가져오기
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                "https://api.spotify.com/v1/me",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
            )
            spotify_user = user_response.json()

        spotify_id = spotify_user["id"]
        email = spotify_user.get("email", "")
        display_name = spotify_user.get("display_name", "") or ""
        images = spotify_user.get("images", [])
        profile_image_url = images[0]["url"] if images else ""

        logger.info(f"Spotify login: {email} ({spotify_id})")

        # 3. DB에 유저 upsert (spotify_id 기준)
        pool = await get_db_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO users (spotify_id, email, display_name, profile_image_url, login_type)
            VALUES ($1, $2, $3, $4, 'spotify')
            ON CONFLICT (spotify_id) DO UPDATE SET
                email = EXCLUDED.email,
                display_name = EXCLUDED.display_name,
                profile_image_url = EXCLUDED.profile_image_url,
                login_type = 'spotify',
                updated_at = NOW()
            RETURNING id::text
            """,
            spotify_id, email, display_name, profile_image_url
        )

        user_id = row["id"]

        # 4. JWT 발급 → 프론트엔드로 리다이렉트
        token = create_jwt_token(user_id, email, login_type="spotify")

        redirect_url = f"{FRONTEND_URL}?token={token}"
        return RedirectResponse(url=redirect_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Spotify OAuth error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """현재 로그인 유저 정보 조회"""
    pool = await get_db_pool()
    row = await pool.fetchrow(
        """
        SELECT id::text, google_id, email, display_name, profile_image_url, login_type, created_at
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
        "login_type": row["login_type"] or "google",
        "created_at": row["created_at"].isoformat(),
    }


# ── Apple Music Developer Token ──────────────────────────────────────────────

_musickit_token_cache: dict = {"token": None, "exp": 0}

def _generate_musickit_token() -> str:
    """MusicKit Developer Token 생성 (ES256 JWT, 최대 6개월)"""
    now = int(time.time())
    # 캐시 유효하면 재사용
    if _musickit_token_cache["token"] and _musickit_token_cache["exp"] > now + 60:
        return _musickit_token_cache["token"]

    with open(APPLE_MUSICKIT_KEY_PATH, "rb") as f:
        private_key = f.read()

    payload = {
        "iss": APPLE_TEAM_ID,
        "iat": now,
        "exp": now + 15777000,  # 6개월
    }
    headers = {"alg": "ES256", "kid": APPLE_MUSICKIT_KEY_ID}
    token = jwt.encode(payload, private_key, algorithm="ES256", headers=headers)

    _musickit_token_cache["token"] = token
    _musickit_token_cache["exp"] = now + 15777000
    return token


@router.get("/apple-music/token")
async def get_musickit_token():
    """MusicKit JS 초기화에 필요한 Developer Token 반환"""
    try:
        token = _generate_musickit_token()
        return {"token": token}
    except Exception as e:
        logger.error(f"MusicKit token generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate MusicKit token")


# ── Sign in with Apple ────────────────────────────────────────────────────────

def _generate_siwa_client_secret() -> str:
    """Sign in with Apple Client Secret JWT 생성 (유효기간 6개월)"""
    now = int(time.time())
    with open(APPLE_SIWA_KEY_PATH, "rb") as f:
        private_key = f.read()

    payload = {
        "iss": APPLE_TEAM_ID,
        "iat": now,
        "exp": now + 15777000,
        "aud": "https://appleid.apple.com",
        "sub": APPLE_SERVICES_ID,
    }
    headers = {"alg": "ES256", "kid": APPLE_SIWA_KEY_ID}
    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)


@router.get("/apple/login")
async def apple_login():
    """Sign in with Apple 인증 페이지로 리다이렉트"""
    params = {
        "client_id": APPLE_SERVICES_ID,
        "redirect_uri": APPLE_REDIRECT_URI,
        "response_type": "code id_token",
        "response_mode": "form_post",
        "scope": "name email",
    }
    url = f"https://appleid.apple.com/auth/authorize?{urlencode(params)}"
    return RedirectResponse(url=url)


@router.post("/apple/callback")
async def apple_callback(request: Request):
    """Apple OAuth 콜백 (POST form_post)"""
    try:
        form = await request.form()
        code = form.get("code")
        id_token = form.get("id_token")
        # 최초 로그인 시에만 user JSON이 옴
        user_json = form.get("user")

        if not code or not id_token:
            raise HTTPException(status_code=400, detail="Missing code or id_token")

        # 1. id_token 검증 (Apple 공개키로)
        async with httpx.AsyncClient() as client:
            jwks_resp = await client.get("https://appleid.apple.com/auth/oauth2/v2/keys")
            jwks = jwks_resp.json()

        header = jwt.get_unverified_header(id_token)
        apple_public_key = next(
            (k for k in jwks["keys"] if k["kid"] == header["kid"]), None
        )
        if not apple_public_key:
            raise HTTPException(status_code=400, detail="Apple public key not found")

        claims = jwt.decode(
            id_token,
            apple_public_key,
            algorithms=["RS256"],
            audience=APPLE_SERVICES_ID,
        )

        apple_id = claims["sub"]
        email = claims.get("email", "")

        # 2. 최초 로그인 시 이름 파싱 (이후엔 안 옴)
        apple_name = None
        if user_json:
            try:
                user_data = json.loads(user_json)
                name = user_data.get("name", {})
                apple_name = f"{name.get('firstName', '')} {name.get('lastName', '')}".strip()
            except Exception:
                pass

        logger.info(f"Apple login: {email} ({apple_id})")

        # 3. DB upsert
        pool = await get_db_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO users (apple_id, email, display_name, login_type)
            VALUES ($1, $2, $3, 'apple')
            ON CONFLICT (apple_id) DO UPDATE SET
                email = COALESCE(NULLIF(EXCLUDED.email, ''), users.email),
                display_name = COALESCE(NULLIF(EXCLUDED.display_name, ''), users.display_name),
                login_type = 'apple',
                updated_at = NOW()
            RETURNING id::text, display_name
            """,
            apple_id, email, apple_name or ""
        )
        user_id = row["id"]

        # apple_name은 최초 1회만 오므로 별도 컬럼에도 저장
        if apple_name:
            await pool.execute(
                "UPDATE users SET apple_name = $1 WHERE id = $2::uuid AND apple_name IS NULL",
                apple_name, user_id
            )

        # 4. JWT 발급 → 프론트엔드로 리다이렉트
        token = create_jwt_token(user_id, email, login_type="apple")
        return RedirectResponse(url=f"{FRONTEND_URL}?token={token}", status_code=303)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apple OAuth error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")


# ── Sign in with Apple (Mobile) ───────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel

class AppleMobileLoginRequest(_BaseModel):
    identity_token: str
    full_name: str | None = None  # 최초 로그인 시에만 전달


@router.post("/apple/mobile")
async def apple_mobile_login(body: AppleMobileLoginRequest):
    """모바일 앱용 Sign in with Apple — identityToken JSON으로 직접 수신"""
    try:
        id_token = body.identity_token

        # 1. id_token 검증 (Apple 공개키로)
        async with httpx.AsyncClient() as client:
            jwks_resp = await client.get("https://appleid.apple.com/auth/oauth2/v2/keys")
            jwks = jwks_resp.json()

        header = jwt.get_unverified_header(id_token)
        apple_public_key = next(
            (k for k in jwks["keys"] if k["kid"] == header["kid"]), None
        )
        if not apple_public_key:
            raise HTTPException(status_code=400, detail="Apple public key not found")

        claims = jwt.decode(
            id_token,
            apple_public_key,
            algorithms=["RS256"],
            audience="kr.caromusic.app",
        )

        apple_id = claims["sub"]
        email = claims.get("email", "")

        logger.info(f"Apple mobile login: {email} ({apple_id})")

        # 2. DB upsert
        pool = await get_db_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO users (apple_id, email, display_name, login_type)
            VALUES ($1, $2, $3, 'apple')
            ON CONFLICT (apple_id) DO UPDATE SET
                email = COALESCE(NULLIF(EXCLUDED.email, ''), users.email),
                display_name = COALESCE(NULLIF(EXCLUDED.display_name, ''), users.display_name),
                login_type = 'apple',
                updated_at = NOW()
            RETURNING id::text, display_name
            """,
            apple_id, email, body.full_name or ""
        )
        user_id = row["id"]

        if body.full_name:
            await pool.execute(
                "UPDATE users SET apple_name = $1 WHERE id = $2::uuid AND apple_name IS NULL",
                body.full_name, user_id
            )

        # 3. JWT 반환 (모바일은 리다이렉트 없이 JSON)
        token = create_jwt_token(user_id, email, login_type="apple")
        return {"token": token, "user_id": user_id, "email": email}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apple mobile login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")
