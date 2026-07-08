-- 유저 플레이리스트 저장 테이블
CREATE TABLE IF NOT EXISTS user_playlists (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    track_keys  TEXT[] NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_playlists_user_id ON user_playlists(user_id);
