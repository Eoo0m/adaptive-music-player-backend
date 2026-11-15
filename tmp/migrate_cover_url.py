"""cover_image_url의 Supabase 도메인을 일괄 변경하는 스크립트 (효율적 버전)"""

import asyncio
import os
import time
import asyncpg
from dotenv import load_dotenv

load_dotenv()

OLD_URL = "https://xopigdxsobjbagmazgba.supabase.co/storage/v1/"
NEW_URL = "https://yxihvsnrszafmluyjqke.supabase.co/storage/v1/"
BATCH_SIZE = 100


async def main():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return

    conn = await asyncpg.connect(database_url, statement_cache_size=0)

    # 1단계: 대상 track_key 목록을 한번에 가져오기 (text 컬럼만 읽으니 빠름)
    print("대상 track_key 목록 조회 중...", flush=True)
    keys = await conn.fetch(
        "SELECT track_key FROM track_embeddings WHERE cover_image_url LIKE $1",
        f"%xopigdxsobjbagmazgba%",
    )
    total = len(keys)
    print(f"변경 대상: {total}건", flush=True)

    if total == 0:
        print("변경할 항목이 없습니다.")
        await conn.close()
        return

    # 2단계: 먼저 10건만 테스트
    test_keys = [row["track_key"] for row in keys[:10]]
    t0 = time.time()
    await conn.execute(
        """
        UPDATE track_embeddings
        SET cover_image_url = REPLACE(cover_image_url, $1, $2)
        WHERE track_key = ANY($3::text[])
        """,
        OLD_URL,
        NEW_URL,
        test_keys,
    )
    elapsed = time.time() - t0
    print(f"테스트: 10건 UPDATE에 {elapsed:.2f}초 소요", flush=True)
    print(f"예상 총 시간: {elapsed / 10 * total / 60:.1f}분", flush=True)
    print("계속 진행합니다...", flush=True)

    # 나머지 배치 처리
    updated = 10
    for i in range(10, total, BATCH_SIZE):
        batch_keys = [row["track_key"] for row in keys[i : i + BATCH_SIZE]]
        t1 = time.time()
        result = await conn.execute(
            """
            UPDATE track_embeddings
            SET cover_image_url = REPLACE(cover_image_url, $1, $2)
            WHERE track_key = ANY($3::text[])
            """,
            OLD_URL,
            NEW_URL,
            batch_keys,
        )
        count = int(result.split()[-1])
        updated += count
        print(f"  진행: {updated}/{total} ({updated * 100 // total}%) - {time.time()-t1:.1f}s", flush=True)

    print(f"완료! 총 {updated}건 업데이트", flush=True)
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
