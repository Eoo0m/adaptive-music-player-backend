-- ============================================================
-- Migration Phase 3a: tracks 메타 인덱스 생성
-- 빠름 (벡터 없음)
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_tracks_title_trgm
ON tracks USING gin ((lower(title || ' ' || artist)) gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_tracks_title_btree
ON tracks (lower(title) text_pattern_ops);

CREATE INDEX IF NOT EXISTS idx_tracks_artist_btree
ON tracks (lower(artist) text_pattern_ops);

CREATE INDEX IF NOT EXISTS idx_tracks_isrc
ON tracks (isrc) WHERE isrc IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tracks_apple_music_id
ON tracks (apple_music_id) WHERE apple_music_id IS NOT NULL;
