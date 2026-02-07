-- Step 2: track_key 인덱스 생성 (빠름)

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_new_track_embeddings_track_key
ON new_track_embeddings(track_key);

-- 인덱스 확인
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'new_track_embeddings'
  AND indexname = 'idx_new_track_embeddings_track_key';
