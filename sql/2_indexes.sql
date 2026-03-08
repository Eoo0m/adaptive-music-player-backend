-- ============================================================
-- Indexes (성능 최적화)
-- 인덱스 생성은 시간이 오래 걸릴 수 있습니다
-- 각 인덱스를 개별적으로 실행하세요
-- ============================================================

-- embedding 컬럼 HNSW 인덱스 (256차원, 유사곡 추천용)
CREATE INDEX IF NOT EXISTS idx_track_embedding_hnsw
ON track_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- projected_embedding 컬럼 HNSW 인덱스 (512차원, 키워드 검색용)
CREATE INDEX IF NOT EXISTS idx_track_projected_embedding_hnsw
ON track_embeddings
USING hnsw (projected_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 텍스트 검색용 pg_trgm 인덱스
CREATE INDEX IF NOT EXISTS idx_track_title_trgm
ON track_embeddings
USING gin ((lower(title || ' ' || artist)) gin_trgm_ops);

-- prefix 검색용 btree 인덱스
CREATE INDEX IF NOT EXISTS idx_track_title_btree
ON track_embeddings (lower(title) text_pattern_ops);

CREATE INDEX IF NOT EXISTS idx_track_artist_btree
ON track_embeddings (lower(artist) text_pattern_ops);
