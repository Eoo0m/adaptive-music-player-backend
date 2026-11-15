import numpy as np
import psycopg2
import time
import json

# 30개 랜덤 쿼리 벡터 생성
np.random.seed(42)
queries = [np.random.rand(512).tolist() for _ in range(30)]

conn = psycopg2.connect(
    "postgresql://postgres.yxihvsnrszafmluyjqke:djawnstj13%21@aws-1-ap-northeast-2.pooler.supabase.com:5432/postgres"
)
conn.autocommit = True
cur = conn.cursor()

cur.execute("SET statement_timeout = '300s'")
cur.execute("SET work_mem = '256MB'")

results = []
times = []

print("Running 30 queries (LIMIT 100)...\n")

for i, q in enumerate(queries):
    vec_str = "[" + ",".join(f"{x:.6f}" for x in q) + "]"

    start = time.time()
    cur.execute(
        f"""
        SELECT track_key::text
        FROM track_embeddings
        WHERE projected_embedding IS NOT NULL
        ORDER BY projected_embedding <=> '{vec_str}'::vector
        LIMIT 100
    """
    )
    rows = cur.fetchall()
    elapsed = time.time() - start
    times.append(elapsed * 1000)

    track_keys = [r[0] for r in rows]
    results.append(
        {"query_idx": i, "time_ms": elapsed * 1000, "top100_keys": track_keys}
    )

    print(f"Query {i+1}/30: {elapsed:.2f}s")

conn.close()

times_arr = np.array(times)
avg_time = np.mean(times_arr)
std_time = np.std(times_arr)

print(f"\n{'='*50}")
print(f"Avg: {avg_time:.1f} ms | Std: {std_time:.1f} ms")
print(f"{'='*50}")

output = {
    "config": "no_index_limit100",
    "total_queries": len(results),
    "avg_time_ms": avg_time,
    "std_time_ms": std_time,
    "results": results,
}

with open("tmp/ground_truth_no_index2.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to /tmp/ground_truth_no_index2.json")

# 정답 출력 (처음 5개만)
print("\n" + "=" * 50)
print("Ground Truth (Top 100 per query, showing first 5 queries):")
print("=" * 50)
for r in results[:5]:
    print(f"\nQuery {r['query_idx']} (showing top 10 of 100):")
    for j, key in enumerate(r["top100_keys"][:10], 1):
        print(f"  {j}. {key}")
