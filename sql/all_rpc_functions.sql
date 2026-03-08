-- ============================================================
-- DynPlayer RPC Functions (track_embeddings 테이블 기준)
-- ============================================================

-- 1. match_tracks_by_projected_embedding
-- 키워드 검색용: 프로젝션된 텍스트 임베딩으로 트랙 직접 검색
CREATE OR REPLACE FUNCTION match_tracks_by_projected_embedding(
    query_embedding vector(512),
    match_count int DEFAULT 10
)
RETURNS TABLE (
    track_key text,
    title text,
    artist text,
    album text,
    playlist_count int,
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
        playlist_count,
        cover_image_url,
        1 - (projected_embedding <=> query_embedding) AS similarity
    FROM track_embeddings
    WHERE projected_embedding IS NOT NULL
    ORDER BY projected_embedding <=> query_embedding
    LIMIT match_count;
$$;


-- 2. match_tracks_by_key
-- 트랙 키 기반 유사곡 추천 (embedding 컬럼 사용)
CREATE OR REPLACE FUNCTION match_tracks_by_key(
    input_track_key text,
    match_count int DEFAULT 30
)
RETURNS TABLE (
    id bigint,
    track_key text,
    title text,
    artist text,
    album text,
    playlist_count int,
    cover_image_url text,
    similarity float
)
LANGUAGE sql
AS $$
    WITH query_track AS (
        SELECT embedding
        FROM track_embeddings
        WHERE track_key = input_track_key
        LIMIT 1
    )
    SELECT
        t.id,
        t.track_key,
        t.title,
        t.artist,
        t.album,
        t.pos_count,
        t.cover_image_url,
        1 - (t.embedding <=> q.embedding) AS similarity
    FROM track_embeddings t, query_track q
    WHERE t.track_key != input_track_key
      AND t.embedding IS NOT NULL
    ORDER BY t.embedding <=> q.embedding
    LIMIT match_count;
$$;


-- 3. match_tracks_by_embedding
-- 임베딩 벡터로 직접 유사곡 검색 (평균 임베딩 기반 추천용)
CREATE OR REPLACE FUNCTION match_tracks_by_embedding(
    query_embedding vector(256),
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id bigint,
    track_key text,
    title text,
    artist text,
    album text,
    playlist_count int,
    cover_image_url text,
    similarity float
)
LANGUAGE sql
AS $$
    SELECT
        id,
        track_key,
        title,
        artist,
        album,
        playlist_count,
        cover_image_url,
        1 - (embedding <=> query_embedding) AS similarity
    FROM track_embeddings
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;


-- 4. search_tracks_by_title
-- 제목/아티스트 기반 텍스트 검색
CREATE OR REPLACE FUNCTION search_tracks_by_title(
    query_text text,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id bigint,
    track_key text,
    title text,
    artist text,
    album text,
    playlist_count int,
    cover_image_url text,
    similarity float
)
LANGUAGE sql
AS $$
    SELECT
        id,
        track_key,
        title,
        artist,
        album,
        playlist_count,
        cover_image_url,
        similarity(lower(title || ' ' || artist), lower(query_text)) AS similarity
    FROM track_embeddings
    WHERE lower(title || ' ' || artist) % lower(query_text)
       OR lower(title) ILIKE query_text || '%'
       OR lower(artist) ILIKE query_text || '%'
    ORDER BY
        CASE WHEN lower(title) ILIKE query_text || '%' THEN 0 ELSE 1 END,
        similarity(lower(title || ' ' || artist), lower(query_text)) DESC
    LIMIT match_count;
$$;


-- ============================================================
-- Indexes (성능 최적화)
-- ============================================================

-- embedding 컬럼 HNSW 인덱스 (256차원, 유사곡 추천용)
CREATE INDEX IF NOT EXISTS idx_track_embedding_hnsw
ON track_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- projected_embedding 컬럼 HNSW 인덱스 (512차원, 키워드 검색용)
CREATE INDEX IF NOT EXISTS idx_track_projected_embedding_hnsw
ON track_embeddings
USING hnsw (projected_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 텍스트 검색용 pg_trgm 인덱스
CREATE INDEX IF NOT EXISTS idx_track_title_trgm
ON track_embeddings
USING gin ((lower(title || ' ' || artist)) gin_trgm_ops);

-- prefix 검색용 btree 인덱스
CREATE INDEX IF NOT EXISTS idx_track_title_btree
ON track_embeddings (lower(title) text_pattern_ops);

CREATE INDEX IF NOT EXISTS idx_track_artist_btree
ON track_embeddings (lower(artist) text_pattern_ops);
