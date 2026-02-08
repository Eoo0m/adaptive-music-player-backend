-- Step 1: new_playlists 벡터 인덱스만 먼저 생성
-- 가장 중요한 인덱스 (키워드 검색에서 가장 많이 사용)

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_new_playlists_embedding_hnsw
ON new_playlists
USING hnsw (embedding vector_ip_ops)
WITH (m = 16, ef_construction = 64);

-- 인덱스 확인
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'new_playlists'
  AND indexname = 'idx_new_playlists_embedding_hnsw';
