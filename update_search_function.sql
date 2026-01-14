-- search_tracks_by_title 함수 수정 - cover_image 포함

-- 기존 함수 삭제
DROP FUNCTION IF EXISTS search_tracks_by_title(text, integer);

-- 새 함수 생성
CREATE OR REPLACE FUNCTION search_tracks_by_title(query_text text, match_count int)
RETURNS TABLE (
    id int,
    track_key text,
    title text,
    artist text,
    album text,
    pos_count int,
    cover_image bytea,
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
        te.cover_image,
        similarity(te.title, query_text) AS similarity
    FROM track_embeddings te
    WHERE te.title ILIKE '%' || query_text || '%'
    ORDER BY similarity DESC, te.pos_count DESC
    LIMIT match_count;
END;
$$;
