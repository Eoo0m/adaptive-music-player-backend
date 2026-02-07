-- 키워드 검색 최적화: 인덱스 추가

-- 1. new_playlists 테이블에 벡터 인덱스 (HNSW 알고리즘 사용)
-- HNSW는 IVFFlat보다 빠르지만 메모리를 더 사용함
CREATE INDEX IF NOT EXISTS idx_new_playlists_embedding_hnsw
ON new_playlists
USING hnsw (embedding vector_ip_ops)
WITH (m = 16, ef_construction = 64);

-- 2. new_track_embeddings 테이블에 track_key 인덱스 (JOIN 최적화)
CREATE INDEX IF NOT EXISTS idx_new_track_embeddings_track_key
ON new_track_embeddings(track_key);

-- 3. new_track_embeddings 테이블에 벡터 인덱스 (평균 기반 추천 최적화)
CREATE INDEX IF NOT EXISTS idx_new_track_embeddings_embedding_hnsw
ON new_track_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 인덱스 상태 확인
SELECT
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename IN ('new_playlists', 'new_track_embeddings')
ORDER BY tablename, indexname;
