-- 유저 취향 클러스터 테이블
-- Ward Hierarchical Clustering + Medoid 기반
-- favorites POST 시 백그라운드로 갱신

CREATE TABLE IF NOT EXISTS user_interest_clusters (
    user_id     UUID PRIMARY KEY,
    k           INT NOT NULL,                   -- 결정된 클러스터 수 (1~5)
    medoids     JSONB NOT NULL,                 -- [[64d vector], ...] k개 medoid
    track_keys  JSONB NOT NULL,                 -- [track_key, ...] 각 medoid에 해당하는 트랙
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
