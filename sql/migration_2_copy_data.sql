-- ============================================================
-- Migration Phase 2: 데이터 복사
-- 순서 중요: tracks 먼저, 그 다음 벡터 테이블들
-- 실행 후 검증 쿼리로 건수 확인
-- ============================================================

-- 2-1. 메타 데이터 복사
INSERT INTO tracks (id, track_key, title, artist, album, cover_image_url, isrc, apple_music_id, playlist_count)
SELECT id, track_key, title, artist, album, cover_image_url, isrc, apple_music_id, playlist_count
FROM track_embeddings
ON CONFLICT (track_key) DO NOTHING;

-- id 시퀀스 재설정
SELECT setval(pg_get_serial_sequence('tracks', 'id'), (SELECT MAX(id) FROM tracks));

-- 2-2. SimGCL 임베딩 복사
INSERT INTO track_vectors_simgcl (track_key, embedding)
SELECT track_key, embedding
FROM track_embeddings
WHERE embedding IS NOT NULL
ON CONFLICT (track_key) DO NOTHING;

-- 2-3. CLIP 임베딩 복사
INSERT INTO track_vectors_clip (track_key, projected_embedding)
SELECT track_key, projected_embedding
FROM track_embeddings
WHERE projected_embedding IS NOT NULL
ON CONFLICT (track_key) DO NOTHING;

-- 2-4. Two-Tower 임베딩 복사
INSERT INTO track_vectors_twotower (track_key, itemtower_embedding)
SELECT track_key, itemtower_embedding
FROM track_embeddings
WHERE itemtower_embedding IS NOT NULL
ON CONFLICT (track_key) DO NOTHING;

-- ============================================================
-- 검증: 복사 후 아래 쿼리로 건수 확인
-- ============================================================
SELECT
    (SELECT COUNT(*) FROM track_embeddings)       AS original,
    (SELECT COUNT(*) FROM tracks)                 AS tracks_cnt,
    (SELECT COUNT(*) FROM track_vectors_simgcl)   AS simgcl_cnt,
    (SELECT COUNT(*) FROM track_vectors_clip)     AS clip_cnt,
    (SELECT COUNT(*) FROM track_vectors_twotower) AS twotower_cnt;
