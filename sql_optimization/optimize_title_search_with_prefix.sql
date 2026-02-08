-- 제목 검색 최적화: Prefix 검색 우선 → pg_trgm 폴백
--
-- 전략:
-- 1. 빠른 Prefix 검색 먼저 시도 (B-tree 인덱스 사용)
--    - "love" 입력 시: title이 "love"로 시작하는 곡들
--    - 매우 빠름 (인덱스 스캔)
-- 2. 결과 없으면 pg_trgm similarity 검색으로 폴백
--    - "love" 포함된 모든 곡들 (I Love You, Lovely Day 등)
--    - 느리지만 정확함

-- Step 1: title, artist에 B-tree 인덱스 생성 (Prefix 검색용)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_new_track_embeddings_title_lower
ON new_track_embeddings (LOWER(title) text_pattern_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_new_track_embeddings_artist_lower
ON new_track_embeddings (LOWER(artist) text_pattern_ops);

-- Step 2: pg_trgm GIN 인덱스 생성 (폴백용)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_new_track_embeddings_title_trgm
ON new_track_embeddings USING gin (title gin_trgm_ops);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_new_track_embeddings_artist_trgm
ON new_track_embeddings USING gin (artist gin_trgm_ops);

-- Step 3: 새로운 검색 함수 (Prefix → pg_trgm 폴백)
DROP FUNCTION IF EXISTS search_tracks_by_title(text, integer);
CREATE OR REPLACE FUNCTION search_tracks_by_title(query_text text, match_count int)
RETURNS TABLE (
    id int,
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
DECLARE
    result_count int;
    query_lower text := LOWER(query_text);
BEGIN
    -- 1단계: Prefix 검색 시도 (매우 빠름)
    RETURN QUERY
    SELECT
        te.id,
        te.track_key,
        te.title,
        te.artist,
        te.album,
        te.pos_count,
        te.cover_image_url,
        1.0::float as similarity  -- Prefix 매칭은 완벽한 매칭
    FROM new_track_embeddings te
    WHERE LOWER(te.title) LIKE query_lower || '%'
       OR LOWER(te.artist) LIKE query_lower || '%'
    ORDER BY
        CASE
            WHEN LOWER(te.title) = query_lower THEN 0  -- 완전 일치 우선
            WHEN LOWER(te.title) LIKE query_lower || '%' THEN 1  -- 제목 prefix
            ELSE 2  -- 아티스트 prefix
        END,
        te.pos_count DESC  -- 인기도 순
    LIMIT match_count;

    -- 결과 개수 확인
    GET DIAGNOSTICS result_count = ROW_COUNT;

    -- 2단계: 결과 없으면 pg_trgm 폴백 (느리지만 포괄적)
    IF result_count = 0 THEN
        RETURN QUERY
        SELECT
            te.id,
            te.track_key,
            te.title,
            te.artist,
            te.album,
            te.pos_count,
            te.cover_image_url,
            GREATEST(
                similarity(te.title, query_text),
                similarity(te.artist, query_text)
            ) as similarity
        FROM new_track_embeddings te
        WHERE te.title ILIKE '%' || query_text || '%'
           OR te.artist ILIKE '%' || query_text || '%'
        ORDER BY
            GREATEST(
                similarity(te.title, query_text),
                similarity(te.artist, query_text)
            ) DESC
        LIMIT match_count;
    END IF;
END;
$$;

-- 인덱스 확인
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'new_track_embeddings'
  AND (indexname LIKE '%title%' OR indexname LIKE '%artist%')
ORDER BY indexname;
