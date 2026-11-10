-- search_keywords: 자동완성용 키워드 테이블
-- keyword = 검색어 (곡명, 아티스트, 분위기 키워드 등)
-- search_count = 누적 검색수 (초기값: playlist_count 기반)

CREATE TABLE IF NOT EXISTS search_keywords (
    keyword      TEXT PRIMARY KEY,
    search_count INT NOT NULL DEFAULT 0
);

-- prefix 검색용 B-tree (ILIKE 'q%' 최적화)
CREATE INDEX IF NOT EXISTS idx_search_keywords_prefix
    ON search_keywords (keyword text_pattern_ops);

-- trigram 유사도 검색용 GIN
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_search_keywords_trgm
    ON search_keywords USING GIN (keyword gin_trgm_ops);

-- 초기 데이터: tracks 테이블의 title, artist로 채우기
-- search_count = playlist_count (저장수 = 인기도)
INSERT INTO search_keywords (keyword, search_count)
SELECT title, COALESCE(playlist_count, 0)
FROM tracks
WHERE title IS NOT NULL AND title != ''
ON CONFLICT (keyword) DO UPDATE
    SET search_count = GREATEST(search_keywords.search_count, excluded.search_count);

INSERT INTO search_keywords (keyword, search_count)
SELECT artist, COALESCE(MAX(playlist_count), 0)
FROM tracks
WHERE artist IS NOT NULL AND artist != ''
GROUP BY artist
ON CONFLICT (keyword) DO UPDATE
    SET search_count = GREATEST(search_keywords.search_count, excluded.search_count);
