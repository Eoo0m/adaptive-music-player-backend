-- match_tracks_by_key 함수 수정 - cover_image_url 포함

-- 기존 함수 삭제
DROP FUNCTION IF EXISTS match_tracks_by_key(text, integer);

-- 새 함수 생성
CREATE OR REPLACE FUNCTION match_tracks_by_key(input_track_key text, match_count int)
RETURNS TABLE (
    id bigint,
    track_key text,
    title text,
    artist text,
    album text,
    pos_count int,
    cover_image_url text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        te.id,
        te.track_key,
        te.title,
        te.artist,
        te.album,
        te.pos_count,
        te.cover_image_url,
        1 - (te.embedding <=> (
            SELECT embedding
            FROM track_embeddings
            WHERE track_embeddings.track_key = input_track_key
        )) AS similarity
    FROM track_embeddings te
    WHERE te.track_key != input_track_key
    ORDER BY te.embedding <=> (
        SELECT embedding
        FROM track_embeddings
        WHERE track_embeddings.track_key = input_track_key
    )
    LIMIT match_count;
END;
$$;
