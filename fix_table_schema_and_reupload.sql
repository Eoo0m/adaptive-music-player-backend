-- Step 1: 테이블 완전히 삭제하고 올바른 스키마로 재생성
DROP TABLE IF EXISTS new_playlists CASCADE;

-- Step 2: 올바른 스키마로 테이블 생성
CREATE TABLE new_playlists (
    id SERIAL PRIMARY KEY,
    playlist_id TEXT UNIQUE NOT NULL,
    playlist_title TEXT,
    saves INTEGER DEFAULT 0,
    track_ids TEXT[],
    embedding vector(512) NOT NULL,  -- 중요: vector(512) 타입으로 생성!
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Step 3: 인덱스 생성
CREATE INDEX new_playlists_playlist_id_idx ON new_playlists(playlist_id);
CREATE INDEX new_playlists_embedding_idx ON new_playlists USING hnsw (embedding vector_cosine_ops);

-- Step 4: 확인
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name = 'new_playlists'
AND column_name = 'embedding';

-- 결과가 data_type = 'USER-DEFINED', udt_name = 'vector' 이어야 함!
