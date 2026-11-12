-- 사용자 취향지도 저장 테이블
CREATE TABLE IF NOT EXISTS user_maps (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT NOT NULL,
    name        TEXT NOT NULL DEFAULT '내 취향지도',
    -- MapTrack 배열을 JSON으로 저장 (track_key, x, y, cover_image_url 등 포함)
    map_data    JSONB NOT NULL DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_maps_user_id ON user_maps(user_id);

-- 업데이트 시 updated_at 자동 갱신
CREATE OR REPLACE FUNCTION update_user_maps_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_user_maps_updated_at
    BEFORE UPDATE ON user_maps
    FOR EACH ROW EXECUTE FUNCTION update_user_maps_updated_at();
