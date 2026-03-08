-- match_tracks_by_projected_embedding
-- 키워드 검색용: 프로젝션된 텍스트 임베딩으로 트랙 직접 검색
--
-- 사용법:
--   SELECT * FROM match_tracks_by_projected_embedding(
--     query_embedding := '[0.1, 0.2, ...]'::vector(512),
--     match_count := 10
--   );

CREATE OR REPLACE FUNCTION match_tracks_by_projected_embedding(
    query_embedding vector(512),
    match_count int DEFAULT 10
)
RETURNS TABLE (
    track_key text,
    title text,
    artist text,
    album text,
    pos_count int,
    cover_image_url text,
    similarity float
)
LANGUAGE sql
AS $$
    SELECT
        track_key,
        title,
        artist,
        album,
        pos_count,
        cover_image_url,
        1 - (projected_embedding <=> query_embedding) AS similarity
    FROM track_embeddings
    WHERE projected_embedding IS NOT NULL
    ORDER BY projected_embedding <=> query_embedding
    LIMIT match_count;
$$;

-- HNSW 인덱스 생성 (성능 최적화)
-- 이미 존재하면 무시됨
CREATE INDEX IF NOT EXISTS idx_track_projected_embedding_hnsw
ON track_embeddings
USING hnsw (projected_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
