"""
iTunes Search API로 apple_music_id 없는 트랙 채우기

흐름:
  1. tracks 테이블에서 apple_music_id IS NULL인 트랙 조회
  2. iTunes Search API (무료, 토큰 불필요)로 title+artist 검색
  3. 결과 중 title/artist가 가장 근접한 것 선택
  4. tracks 테이블에 UPDATE

사용법:
    python scripts/fill_apple_ids_itunes.py [--limit N] [--dry-run]

옵션:
    --limit N     최대 N개만 처리 (기본: 전체)
    --dry-run     DB 업데이트 없이 로그만 출력
    --storefront  kr|us|jp 등 (기본: kr, 실패 시 us 재시도)
"""

import asyncio
import asyncpg
import httpx
import os
import sys
import time
import logging
import unicodedata
import re
import argparse
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
SLEEP_BETWEEN = 0.35   # iTunes API 초당 약 2~3건 허용
SLEEP_ON_RATE  = 10.0


def normalize(s: str) -> str:
    """비교용 정규화: 소문자, 특수문자 제거, 공백 정리"""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_match(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)


def artist_match(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    return na == nb or na in nb or nb in na


def strip_title_suffix(title: str) -> str:
    """검색용 타이틀: 버전/피처링 suffix 제거"""
    t = title
    # " - " 이후 버전/에디션/리마스터 등 제거
    t = re.sub(r"\s*-\s*[\w\s]*(remaster|remix|edit|version|live|acoustic|radio|extended|deluxe)[\w\s]*$", "", t, flags=re.IGNORECASE)
    # (From ... Movie/Series/OST) 괄호 제거
    t = re.sub(r"\s*\(from\s+.+?\)", "", t, flags=re.IGNORECASE)
    # (feat. ...) / (ft. ...) 괄호 제거
    t = re.sub(r"\s*\([fF](eat|t)\..*?\)", "", t)
    return t.strip(" -")


async def _search_once(client: httpx.AsyncClient, title: str, artist: str, storefronts: list[str]) -> str | None:
    """여러 storefront 순서로 시도, 성공하면 즉시 반환"""
    term = f"{title} {artist}".strip()
    for sf in storefronts:
        for attempt in range(2):
            try:
                resp = await client.get(
                    "https://itunes.apple.com/search",
                    params={"term": term, "media": "music", "entity": "song", "limit": "10", "country": sf},
                    timeout=15,
                )
                if resp.status_code == 429:
                    logger.warning("Rate limit, 대기 중...")
                    await asyncio.sleep(SLEEP_ON_RATE)
                    continue
                if resp.status_code in (403, 404):
                    break  # 이 storefront는 포기, 다음으로
                resp.raise_for_status()
                results = resp.json().get("results", [])

                # 1순위: title + artist 모두 일치
                for r in results:
                    if title_match(r.get("trackName", ""), title) and artist_match(r.get("artistName", ""), artist):
                        return str(r["trackId"])

                # 2순위: title만 일치
                for r in results:
                    if title_match(r.get("trackName", ""), title):
                        return str(r["trackId"])

                break  # 결과는 받았으나 매칭 없음, 다음 storefront 시도 불필요
            except (httpx.TimeoutException, httpx.ConnectError):
                await asyncio.sleep(2)
    return None


async def search_itunes(client: httpx.AsyncClient, title: str, artist: str, storefront: str = "kr") -> str | None:
    """iTunes Search API로 apple_music_id 반환.
    원본 타이틀로 kr→us 시도 후, suffix 제거 타이틀로 us만 재시도."""
    storefronts = [storefront] if storefront != "kr" else ["kr", "us", "jp"]

    # 1차: 원본 타이틀 + 아티스트
    result = await _search_once(client, title, artist, storefronts)
    if result:
        return result

    # 2차: suffix 제거 타이틀 + 아티스트
    stripped = strip_title_suffix(title)
    if stripped != title:
        result = await _search_once(client, stripped, artist, ["us", "jp"])
        if result:
            return result

    # 3차: 타이틀만 (아티스트 없이) — iTunes에서 아티스트명이 다른 경우 대응
    result = await _search_once(client, title, "", ["jp", "us"])
    if result:
        return result

    # 4차: suffix 제거 타이틀만
    if stripped != title:
        result = await _search_once(client, stripped, "", ["jp", "us"])
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--storefront", default="kr")
    args = parser.parse_args()

    pool = await asyncpg.create_pool(DATABASE_URL, statement_cache_size=0, min_size=1, max_size=2)

    query = "SELECT track_key, title, artist FROM tracks WHERE apple_music_id IS NULL AND title IS NOT NULL ORDER BY playlist_count DESC NULLS LAST"
    if args.limit:
        query += f" LIMIT {args.limit}"

    rows = await pool.fetch(query)
    total = len(rows)
    logger.info(f"대상: {total}개 (dry_run={args.dry_run}, storefront={args.storefront})")

    mapped = 0
    failed = 0
    start = time.time()

    async with httpx.AsyncClient() as client:
        for i, row in enumerate(rows):
            track_key = row["track_key"]
            title = row["title"] or ""
            artist = row["artist"] or ""

            apple_id = await search_itunes(client, title, artist, args.storefront)

            if apple_id:
                mapped += 1
                logger.info(f"[{i+1}/{total}] ✅ {title} - {artist} → {apple_id}")
                if not args.dry_run:
                    await pool.execute(
                        "UPDATE tracks SET apple_music_id = $1 WHERE track_key = $2",
                        apple_id, track_key
                    )
            else:
                failed += 1
                logger.info(f"[{i+1}/{total}] ❌ {title} - {artist}")

            await asyncio.sleep(SLEEP_BETWEEN)

            # 100개마다 중간 집계
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate / 60
                logger.info(f"--- 진행: {i+1}/{total} | 성공: {mapped} | 실패: {failed} | 남은시간: {remaining:.1f}분 ---")

    await pool.close()
    logger.info(f"완료 — 성공: {mapped}, 실패: {failed}, 총: {total}")


if __name__ == "__main__":
    asyncio.run(main())
