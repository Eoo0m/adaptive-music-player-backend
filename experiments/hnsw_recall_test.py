"""
HNSW vs Brute-force Recall 실험

1. 새 테이블 생성 (hnsw_test_embeddings)
2. SimGCL 임베딩 업로드
3. Brute-force (인덱스 없이) top-100 쿼리 100번
4. HNSW 인덱스 생성
5. HNSW top-100 쿼리 100번
6. Recall@k 계산 및 결과 저장
"""

import asyncio
import asyncpg
import numpy as np
import json
import time
import os
import random
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "/Users/eomjoonseo/dynplayer/src/simgcl/outputs/simgcl_dot_ips/outputs/min5_win10"
DATABASE_URL = os.getenv("DATABASE_URL")
NUM_QUERIES = 100
TOP_K = 100
RESULT_FILE = os.path.join(os.path.dirname(__file__), "hnsw_recall_results.json")


async def main():
    print("=" * 60)
    print("HNSW vs Brute-force Recall 실험")
    print("=" * 60)

    # 데이터 로드
    print("\n[1/6] 데이터 로드 중...")
    track_ids = np.load(os.path.join(DATA_DIR, "model_loo_track_ids.npy"))
    track_embeddings = np.load(os.path.join(DATA_DIR, "model_loo_track_embeddings.npy"))
    num_tracks = len(track_ids)
    dim = track_embeddings.shape[1]
    print(f"  트랙 수: {num_tracks}, 차원: {dim}")

    # DB 연결
    conn = await asyncpg.connect(DATABASE_URL, statement_cache_size=0)
    print("  DB 연결 완료")

    # 테이블 생성
    print("\n[2/6] 테이블 생성 중...")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.execute("DROP TABLE IF EXISTS hnsw_test_embeddings")
    await conn.execute(f"""
        CREATE TABLE hnsw_test_embeddings (
            id SERIAL PRIMARY KEY,
            track_key TEXT UNIQUE NOT NULL,
            embedding vector({dim})
        )
    """)
    print("  테이블 생성 완료")

    # 데이터 업로드 (배치)
    print("\n[3/6] 임베딩 업로드 중...")
    batch_size = 1000
    t_upload = time.time()
    for i in range(0, num_tracks, batch_size):
        batch_end = min(i + batch_size, num_tracks)
        records = []
        for j in range(i, batch_end):
            records.append((
                str(track_ids[j]),
                str(track_embeddings[j].tolist())
            ))
        await conn.executemany(
            "INSERT INTO hnsw_test_embeddings (track_key, embedding) VALUES ($1, $2::vector)",
            records
        )
        if (i // batch_size) % 10 == 0:
            print(f"  {batch_end}/{num_tracks} 업로드됨...")
    upload_time = time.time() - t_upload
    print(f"  업로드 완료: {num_tracks}개 ({upload_time:.1f}s)")

    # ANALYZE
    await conn.execute("ANALYZE hnsw_test_embeddings")

    # 쿼리용 랜덤 트랙 선택
    random.seed(42)
    query_indices = random.sample(range(num_tracks), NUM_QUERIES)
    query_embeddings = track_embeddings[query_indices]
    query_ids = track_ids[query_indices]

    # ============================================================
    # Brute-force (인덱스 없이)
    # ============================================================
    print(f"\n[4/6] Brute-force 쿼리 {NUM_QUERIES}회 실행 중...")
    # 인덱스가 없는 상태에서 seq scan 강제
    await conn.execute("SET enable_indexscan = off")
    await conn.execute("SET enable_bitmapscan = off")

    bruteforce_results = {}
    bruteforce_times = []

    for i, (qid, qemb) in enumerate(zip(query_ids, query_embeddings)):
        emb_str = str(qemb.tolist())
        t_start = time.time()
        rows = await conn.fetch(
            """
            SELECT track_key, (1 - (embedding <=> $1::vector))::float AS similarity
            FROM hnsw_test_embeddings
            WHERE track_key != $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            emb_str, str(qid), TOP_K
        )
        elapsed = time.time() - t_start
        bruteforce_times.append(elapsed)
        bruteforce_results[str(qid)] = [r["track_key"] for r in rows]

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{NUM_QUERIES} 완료 (avg: {np.mean(bruteforce_times):.3f}s)")

    await conn.execute("RESET enable_indexscan")
    await conn.execute("RESET enable_bitmapscan")

    bf_mean = np.mean(bruteforce_times)
    bf_median = np.median(bruteforce_times)
    print(f"  Brute-force 완료: mean={bf_mean:.3f}s, median={bf_median:.3f}s")

    # ============================================================
    # HNSW 인덱스 생성
    # ============================================================
    print(f"\n[5/6] HNSW 인덱스 생성 중 (m=16, ef_construction=64)...")
    t_index = time.time()
    await conn.execute("""
        CREATE INDEX idx_hnsw_test_embedding
        ON hnsw_test_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    index_time = time.time() - t_index
    print(f"  인덱스 생성 완료 ({index_time:.1f}s)")

    # ANALYZE 재실행
    await conn.execute("ANALYZE hnsw_test_embeddings")

    # ============================================================
    # HNSW 쿼리
    # ============================================================
    print(f"\n[6/6] HNSW 쿼리 {NUM_QUERIES}회 실행 중...")
    hnsw_results = {}
    hnsw_times = []

    for i, (qid, qemb) in enumerate(zip(query_ids, query_embeddings)):
        emb_str = str(qemb.tolist())
        t_start = time.time()
        rows = await conn.fetch(
            """
            SELECT track_key, (1 - (embedding <=> $1::vector))::float AS similarity
            FROM hnsw_test_embeddings
            WHERE track_key != $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            emb_str, str(qid), TOP_K
        )
        elapsed = time.time() - t_start
        hnsw_times.append(elapsed)
        hnsw_results[str(qid)] = [r["track_key"] for r in rows]

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{NUM_QUERIES} 완료 (avg: {np.mean(hnsw_times):.3f}s)")

    hnsw_mean = np.mean(hnsw_times)
    hnsw_median = np.median(hnsw_times)
    print(f"  HNSW 완료: mean={hnsw_mean:.3f}s, median={hnsw_median:.3f}s")

    # ============================================================
    # Recall 계산
    # ============================================================
    print("\n" + "=" * 60)
    print("결과 분석")
    print("=" * 60)

    recall_at = {1: [], 5: [], 10: [], 20: [], 50: [], 100: []}

    for qid in bruteforce_results:
        bf_set = bruteforce_results[qid]
        hnsw_set = hnsw_results[qid]

        for k in recall_at:
            bf_top_k = set(bf_set[:k])
            hnsw_top_k = set(hnsw_set[:k])
            if len(bf_top_k) > 0:
                recall = len(bf_top_k & hnsw_top_k) / len(bf_top_k)
                recall_at[k].append(recall)

    results = {
        "config": {
            "num_tracks": num_tracks,
            "embedding_dim": dim,
            "num_queries": NUM_QUERIES,
            "top_k": TOP_K,
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
        },
        "timing": {
            "upload_seconds": round(upload_time, 2),
            "index_build_seconds": round(index_time, 2),
            "bruteforce": {
                "mean": round(bf_mean, 4),
                "median": round(bf_median, 4),
                "min": round(min(bruteforce_times), 4),
                "max": round(max(bruteforce_times), 4),
            },
            "hnsw": {
                "mean": round(hnsw_mean, 4),
                "median": round(hnsw_median, 4),
                "min": round(min(hnsw_times), 4),
                "max": round(max(hnsw_times), 4),
            },
            "speedup": round(bf_mean / hnsw_mean, 2) if hnsw_mean > 0 else 0,
        },
        "recall": {},
    }

    print(f"\n{'k':>5} | {'Recall@k':>10} | {'상세'}")
    print("-" * 50)
    for k in sorted(recall_at.keys()):
        vals = recall_at[k]
        mean_recall = np.mean(vals)
        min_recall = np.min(vals)
        max_recall = np.max(vals)
        results["recall"][f"recall@{k}"] = {
            "mean": round(mean_recall, 4),
            "min": round(min_recall, 4),
            "max": round(max_recall, 4),
        }
        print(f"{k:>5} | {mean_recall:>9.4f} | min={min_recall:.4f}, max={max_recall:.4f}")

    print(f"\n속도:")
    print(f"  Brute-force: {bf_mean:.4f}s (avg)")
    print(f"  HNSW:        {hnsw_mean:.4f}s (avg)")
    print(f"  Speedup:     {results['timing']['speedup']}x")

    # 결과 저장
    with open(RESULT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {RESULT_FILE}")

    # 테스트 테이블 정리
    await conn.execute("DROP TABLE IF EXISTS hnsw_test_embeddings")
    print("테스트 테이블 삭제 완료")

    await conn.close()
    print("\n실험 완료!")


if __name__ == "__main__":
    asyncio.run(main())
