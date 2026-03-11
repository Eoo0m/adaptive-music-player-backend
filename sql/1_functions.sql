-- ============================================================
-- DynPlayer RPC Functions (track_embeddings 테이블 기준)
-- ============================================================

-- 기존 함수 삭제 (리턴 타입 변경 시 필요)
DROP FUNCTION IF EXISTS match_tracks_by_projected_embedding(vector(512), int);
DROP FUNCTION IF EXISTS match_tracks_by_key(text, int);
DROP FUNCTION IF EXISTS match_tracks_by_embedding(vector(256), int);
DROP FUNCTION IF EXISTS search_tracks_by_title(text, int);


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
LANGUAGE plpgsql
AS $$
BEGIN
    -- Supabase 무료 플랜 기본 8초 → 30초로 확장
    SET LOCAL statement_timeout = '30s';

    RETURN QUERY
    SELECT
        t.track_key,
        t.title,
        t.artist,
        t.album,
        t.playlist_count,
        t.cover_image_url,
        1 - (t.projected_embedding <=> query_embedding) AS similarity
    FROM track_embeddings t
    WHERE t.projected_embedding IS NOT NULL
    ORDER BY t.projected_embedding <=> query_embedding
    LIMIT match_count;
END;
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
LANGUAGE plpgsql
AS $$
BEGIN
    SET LOCAL statement_timeout = '30s';

    RETURN QUERY
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
        t.playlist_count,
        t.cover_image_url,
        1 - (t.embedding <=> q.embedding) AS similarity
    FROM track_embeddings t, query_track q
    WHERE t.track_key != input_track_key
      AND t.embedding IS NOT NULL
    ORDER BY t.embedding <=> q.embedding
    LIMIT match_count;
END;
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
LANGUAGE plpgsql
AS $$
BEGIN
    SET LOCAL statement_timeout = '30s';

    RETURN QUERY
    SELECT
        t.id,
        t.track_key,
        t.title,
        t.artist,
        t.album,
        t.playlist_count,
        t.cover_image_url,
        1 - (t.embedding <=> query_embedding) AS similarity
    FROM track_embeddings t
    WHERE t.embedding IS NOT NULL
    ORDER BY t.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;


-- 4. match_tracks_by_user_embedding
-- Two-Tower User 임베딩 기반 추천 (itemtower_embedding과 비교)
DROP FUNCTION IF EXISTS match_tracks_by_user_embedding(vector(128), text[], int);

CREATE OR REPLACE FUNCTION match_tracks_by_user_embedding(
    query_embedding vector(128),
    exclude_track_keys text[] DEFAULT '{}',
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
LANGUAGE plpgsql
AS $$
BEGIN
    SET LOCAL statement_timeout = '30s';

    RETURN QUERY
    SELECT
        t.id,
        t.track_key,
        t.title,
        t.artist,
        t.album,
        t.playlist_count,
        t.cover_image_url,
        1 - (t.itemtower_embedding <=> query_embedding) AS similarity
    FROM track_embeddings t
    WHERE t.itemtower_embedding IS NOT NULL
      AND (array_length(exclude_track_keys, 1) IS NULL
           OR t.track_key != ALL(exclude_track_keys))
    ORDER BY t.itemtower_embedding <=> query_embedding
    LIMIT match_count;
END;
$$;


-- 5. search_tracks_by_title
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
LANGUAGE plpgsql
AS $$
BEGIN
    SET LOCAL statement_timeout = '30s';

    RETURN QUERY
    SELECT
        t.id,
        t.track_key,
        t.title,
        t.artist,
        t.album,
        t.playlist_count,
        t.cover_image_url,
        similarity(lower(t.title || ' ' || t.artist), lower(query_text)) AS similarity
    FROM track_embeddings t
    WHERE lower(t.title || ' ' || t.artist) % lower(query_text)
       OR lower(t.title) ILIKE query_text || '%'
       OR lower(t.artist) ILIKE query_text || '%'
    ORDER BY
        CASE WHEN lower(t.title) ILIKE query_text || '%' THEN 0 ELSE 1 END,
        similarity(lower(t.title || ' ' || t.artist), lower(query_text)) DESC
    LIMIT match_count;
END;
$$;
