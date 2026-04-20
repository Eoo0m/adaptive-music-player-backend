CREATE TABLE IF NOT EXISTS user_action_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    user_id UUID REFERENCES users(id),
    action_type TEXT NOT NULL,
    search_query TEXT,
    search_mode TEXT,
    selected_track_key TEXT,
    candidate_track_keys TEXT[],
    extra TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_action_logs_session ON user_action_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_action_logs_user ON user_action_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_action_logs_type ON user_action_logs(action_type);
CREATE INDEX IF NOT EXISTS idx_action_logs_created ON user_action_logs(created_at);
