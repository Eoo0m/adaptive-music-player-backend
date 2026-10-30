-- ============================================================
-- Migration Phase 3c: CLIP HNSW 인덱스 (512d, m=16)
-- 가장 오래 걸림 (512차원) — 단독 실행
-- ============================================================

SET statement_timeout = 0;

CREATE INDEX IF NOT EXISTS idx_clip_projected_hnsw
ON track_vectors_clip USING hnsw (projected_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
