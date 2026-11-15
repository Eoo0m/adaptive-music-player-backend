import asyncio, sys, json, numpy as np
sys.path.insert(0, '/opt/dynplayer')
from scipy.cluster.hierarchy import linkage, fcluster
from routes.cluster import _auto_k
from database import get_db_pool

async def show_clusters(uid, label):
    pool = await get_db_pool()
    rows = await pool.fetch("""
        SELECT f.track_key, te.embedding
        FROM user_favorites f
        JOIN track_embeddings te ON f.track_key = te.track_key
        WHERE f.user_id = $1 AND te.embedding IS NOT NULL
        ORDER BY f.created_at DESC LIMIT 50
    """, uid)
    tkeys = [r['track_key'] for r in rows]
    embs = np.array([
        json.loads(r['embedding']) if isinstance(r['embedding'], str) else list(r['embedding'])
        for r in rows
    ], dtype=np.float32)
    Z = linkage(embs, method='ward')
    k = _auto_k(Z)
    labels = fcluster(Z, k, criterion='maxclust')

    clusters = {}
    for i, lbl in enumerate(labels):
        clusters.setdefault(int(lbl), []).append(tkeys[i])

    all_keys = [k for keys in clusters.values() for k in keys]
    track_rows = await pool.fetch(
        "SELECT track_key, title, artist FROM tracks WHERE track_key = ANY($1)", all_keys
    )
    info = {r['track_key']: f"{r['title']} - {r['artist']}" for r in track_rows}

    print(f"\n=== {label} (K={k}) ===")
    for c, keys in sorted(clusters.items()):
        print(f"\n  [클러스터 {c}] {len(keys)}곡")
        for key in keys:
            print(f"    {info.get(key, key)}")

async def main():
    await show_clusters('83488bca-b671-4d84-88db-daa3eb4e4a78', '계정1')
    await show_clusters('63df1fe4-cf92-481d-85fb-0fb891db4545', '계정2')

asyncio.run(main())
