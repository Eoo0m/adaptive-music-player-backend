-- ============================================================
-- Migration Phase 3d: Two-Tower HNSW 인덱스 (128d, m=16)
-- 시간 소요 — 단독 실행
-- ============================================================

SET statement_timeout = 0;

CREATE INDEX IF NOT EXISTS idx_twotower_itemtower_hnsw
ON track_vectors_twotower USING hnsw (itemtower_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
