"""CSV 임시테이블 → 배치 UPDATE JOIN"""

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
BATCH_SIZE = 500


async def main():
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"), statement_cache_size=0, timeout=3600)

    # 좀비 정리
    print("좀비 쿼리 정리...", flush=True)
    await conn.execute("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE state = 'active' AND pid != pg_backend_pid()
    """)
    await asyncio.sleep(2)

    # 대상 데이터 조회
    print("대상 데이터 조회...", flush=True)
    rows = await conn.fetch(
        "SELECT track_key, cover_image_url FROM track_embeddings WHERE cover_image_url LIKE $1",
        f"%{OLD_HOST}%",
    )
    total = len(rows)
    print(f"대상: {total}건", flush=True)

    if not total:
        print("변경할 항목 없음")
        await conn.close()
        return

    # CSV 생성
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow([r["track_key"], r["cover_image_url"].replace(OLD_HOST, NEW_HOST)])

    # 임시 테이블 + COPY
    print("임시 테이블 COPY...", flush=True)
    await conn.execute("DROP TABLE IF EXISTS tmp_cover_url_fix")
    await conn.execute("""
        CREATE UNLOGGED TABLE tmp_cover_url_fix (
            track_key varchar PRIMARY KEY,
            new_url text
        )
    """)
    with open(CSV_PATH, "rb") as f:
        await conn.copy_to_table("tmp_cover_url_fix", source=f, format="csv", columns=["track_key", "new_url"])
    print("COPY 완료", flush=True)

    # 배치 UPDATE JOIN
    updated = 0
    keys = [r["track_key"] for r in rows]
    for i in range(0, total, BATCH_SIZE):
        batch = keys[i:i + BATCH_SIZE]
        t0 = time.time()
        result = await conn.execute("""
            UPDATE track_embeddings t
            SET cover_image_url = tmp.new_url
            FROM tmp_cover_url_fix tmp
            WHERE t.track_key = tmp.track_key
            AND t.track_key = ANY($1::text[])
        """, batch)
        count = int(result.split()[-1])
        updated += count
        print(f"  진행: {updated}/{total} ({updated*100//total}%) - {time.time()-t0:.1f}s", flush=True)

    print(f"\nUPDATE 완료: {updated}건", flush=True)

    # 정리
    await conn.execute("DROP TABLE tmp_cover_url_fix")
    os.remove(CSV_PATH)

    # 인덱스 재생성
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
