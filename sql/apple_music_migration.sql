-- Apple Music 연동을 위한 DB 마이그레이션
-- Supabase SQL Editor에서 실행

-- 1. users 테이블에 apple_id 컬럼 추가
ALTER TABLE users ADD COLUMN IF NOT EXISTS apple_id TEXT UNIQUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS apple_name TEXT;  -- 최초 로그인 때만 오므로 별도 저장

-- login_type에 'apple' 허용 (기존에 CHECK constraint 있으면 수정 필요)
-- login_type은 TEXT라 별도 작업 불필요

-- 2. track_embeddings 테이블에 Apple Music 관련 컬럼 추가
ALTER TABLE track_embeddings ADD COLUMN IF NOT EXISTS isrc TEXT;
ALTER TABLE track_embeddings ADD COLUMN IF NOT EXISTS apple_music_id TEXT;

-- ISRC 인덱스 (배치 매핑 조회 최적화)
CREATE INDEX IF NOT EXISTS idx_track_isrc ON track_embeddings (isrc) WHERE isrc IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_track_apple_music_id ON track_embeddings (apple_music_id) WHERE apple_music_id IS NOT NULL;
