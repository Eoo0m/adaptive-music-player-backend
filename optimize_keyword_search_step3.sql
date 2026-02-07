-- Step 3: new_track_embeddings 벡터 인덱스 생성
-- 평균 기반 추천에 사용

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_new_track_embeddings_embedding_hnsw
ON new_track_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 인덱스 확인
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'new_track_embeddings'
  AND indexname = 'idx_new_track_embeddings_embedding_hnsw';
