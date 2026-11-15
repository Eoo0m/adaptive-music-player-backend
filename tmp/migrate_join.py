"""CSV 임시테이블 → UPDATE JOIN 방식 마이그레이션"""

import asyncio
import os
import time
import csv
import asyncpg
from dotenv import load_dotenv

load_dotenv()

CSV_PATH = "tmp/cover_url_fix.csv"
OLD_HOST = "xopigdxsobjbagmazgba.supabase.co"
NEW_HOST = "yxihvsnrszafmluyjqke.supabase.co"


async def main():
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"), statement_cache_size=0, timeout=3600)

    # 0. 좀비 정리
    print("좀비 쿼리 정리...", flush=True)
    await conn.execute("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE state = 'active' AND pid != pg_backend_pid()
    """)
    await asyncio.sleep(2)

    # 1. 대상 데이터 가져오기
    print("대상 데이터 조회...", flush=True)
    rows = await conn.fetch(
        "SELECT track_key, cover_image_url FROM track_embeddings WHERE cover_image_url LIKE $1",
        f"%{OLD_HOST}%",
    )
    print(f"대상: {len(rows)}건", flush=True)

    if not rows:
        print("변경할 항목 없음")
        await conn.close()
        return

    # 2. 로컬에서 CSV 생성 (URL 치환)
    print("CSV 생성...", flush=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        for r in rows:
            new_url = r["cover_image_url"].replace(OLD_HOST, NEW_HOST)
            writer.writerow([r["track_key"], new_url])
    print(f"CSV 생성 완료: {len(rows)}건", flush=True)

    # 3. 임시 테이블 생성
    print("임시 테이블 생성...", flush=True)
    await conn.execute("DROP TABLE IF EXISTS tmp_cover_url_fix")
    await conn.execute("""
        CREATE UNLOGGED TABLE tmp_cover_url_fix (
            track_key varchar,
            new_url text
        )
    """)

    # 4. CSV → 임시 테이블 COPY
    print("CSV COPY 중...", flush=True)
    t0 = time.time()
    with open(CSV_PATH, "rb") as f:
        await conn.copy_to_table(
            "tmp_cover_url_fix",
            source=f,
            format="csv",
            columns=["track_key", "new_url"],
        )
    print(f"COPY 완료 ({time.time()-t0:.1f}s)", flush=True)

    # 5. UPDATE JOIN (DB 내부에서 처리)
    print("UPDATE JOIN 실행...", flush=True)
    t0 = time.time()
    result = await conn.execute("""
        UPDATE track_embeddings t
        SET cover_image_url = tmp.new_url
        FROM tmp_cover_url_fix tmp
        WHERE t.track_key = tmp.track_key
    """)
    print(f"UPDATE 완료: {result} ({time.time()-t0:.1f}s)", flush=True)

    # 6. 정리
    await conn.execute("DROP TABLE tmp_cover_url_fix")
    os.remove(CSV_PATH)
    print("임시 데이터 정리 완료", flush=True)

    # 7. 인덱스 재생성
    print("\nHNSW 인덱스 재생성 (1/3: embedding 64d)...", flush=True)
    t0 = time.time()
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_track_embedding_hnsw ON track_embeddings
        USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)
    print(f"  완료 ({time.time()-t0:.1f}s)", flush=True)

    print("HNSW 인덱스 재생성 (2/3: projected 512d)...", flush=True)
    t0 = time.time()
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_track_projected_embedding_hnsw ON track_embeddings
        USING hnsw (projected_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)
    print(f"  완료 ({time.time()-t0:.1f}s)", flush=True)

    print("HNSW 인덱스 재생성 (3/3: itemtower 128d)...", flush=True)
    t0 = time.time()
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_track_itemtower_embedding_hnsw ON track_embeddings
        USING hnsw (itemtower_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    """)
    print(f"  완료 ({time.time()-t0:.1f}s)", flush=True)

    print("\n전체 완료!", flush=True)
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
