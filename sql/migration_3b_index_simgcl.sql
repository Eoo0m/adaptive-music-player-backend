-- ============================================================
-- Migration Phase 3b: SimGCL HNSW 인덱스 (64d, m=32)
-- 시간 소요 — 단독 실행
-- ============================================================

SET statement_timeout = 0;

CREATE INDEX IF NOT EXISTS idx_simgcl_embedding_hnsw
ON track_vectors_simgcl USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 200);
