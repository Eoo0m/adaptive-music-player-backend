-- ============================================================
-- Migration Phase 4: RPC 함수 재작성 (새 테이블 기준)
-- 코드 배포 전 실행
-- ============================================================

DROP FUNCTION IF EXISTS match_tracks_by_projected_embedding(vector(512), int);
DROP FUNCTION IF EXISTS match_tracks_by_key(text, int);
DROP FUNCTION IF EXISTS match_tracks_by_embedding(vector(256), int);
DROP FUNCTION IF EXISTS match_tracks_by_user_embedding(vector(128), text[], int);
DROP FUNCTION IF EXISTS search_tracks_by_title(text, int);

CREATE OR REPLACE FUNCTION match_tracks_by_projected_embedding(
    query_embedding vector(512),
    match_count int DEFAULT 10
)
RETURNS TABLE (
    track_key text, title text, artist text, album text,
    playlist_count int, cover_image_url text, similarity float
)
LANGUAGE sql AS $$
    SELECT
        t.track_key::text, t.title::text, t.artist::text, t.album::text,
        t.playlist_count, t.cover_image_url::text,
        (1 - (v.projected_embedding <=> query_embedding))::float AS similarity
    FROM track_vectors_clip v
    JOIN tracks t ON v.track_key = t.track_key
    ORDER BY v.projected_embedding <=> query_embedding
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION match_tracks_by_key(
    input_track_key text,
    match_count int DEFAULT 30
)
RETURNS TABLE (
    id bigint, track_key text, title text, artist text, album text,
    playlist_count int, cover_image_url text, similarity float
)
LANGUAGE sql AS $$
    SELECT
        t.id, t.track_key::text, t.title::text, t.artist::text, t.album::text,
        t.playlist_count, t.cover_image_url::text,
        (1 - (v.embedding <=> (SELECT embedding FROM track_vectors_simgcl WHERE track_key = input_track_key)))::float AS similarity
    FROM track_vectors_simgcl v
    JOIN tracks t ON v.track_key = t.track_key
    WHERE v.track_key != input_track_key
    ORDER BY v.embedding <=> (SELECT embedding FROM track_vectors_simgcl WHERE track_key = input_track_key)
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION match_tracks_by_user_embedding(
    query_embedding vector(128),
    exclude_track_keys text[] DEFAULT '{}',
    match_count int DEFAULT 30
)
RETURNS TABLE (
    id bigint, track_key text, title text, artist text, album text,
    playlist_count int, cover_image_url text, similarity float
)
LANGUAGE sql AS $$
    SELECT
        t.id, t.track_key::text, t.title::text, t.artist::text, t.album::text,
        t.playlist_count, t.cover_image_url::text,
        (1 - (v.itemtower_embedding <=> query_embedding))::float AS similarity
    FROM track_vectors_twotower v
    JOIN tracks t ON v.track_key = t.track_key
    WHERE (array_length(exclude_track_keys, 1) IS NULL
           OR v.track_key::text != ALL(exclude_track_keys))
    ORDER BY v.itemtower_embedding <=> query_embedding
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION search_tracks_by_title(
    query_text text,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id bigint, track_key text, title text, artist text, album text,
    playlist_count int, cover_image_url text, similarity float
)
LANGUAGE sql AS $$
    SELECT
        t.id, t.track_key::text, t.title::text, t.artist::text, t.album::text,
        t.playlist_count, t.cover_image_url::text,
        similarity(lower(t.title || ' ' || t.artist), lower(query_text))::float AS similarity
    FROM tracks t
    WHERE lower(t.title || ' ' || t.artist) % lower(query_text)
       OR lower(t.title) ILIKE query_text || '%'
       OR lower(t.artist) ILIKE query_text || '%'
    ORDER BY
        CASE WHEN lower(t.title) ILIKE query_text || '%' THEN 0 ELSE 1 END,
        similarity(lower(t.title || ' ' || t.artist), lower(query_text)) DESC
    LIMIT match_count;
$$;
