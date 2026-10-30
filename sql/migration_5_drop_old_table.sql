-- ============================================================
-- Migration Phase 5: 구 테이블 DROP
-- 코드 배포 후 모니터링 완료 후 실행
-- ============================================================

-- FK 참조 확인 먼저
SELECT tc.table_name, kcu.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.referential_constraints rc ON tc.constraint_name = rc.constraint_name
JOIN information_schema.table_constraints tc2 ON rc.unique_constraint_name = tc2.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY' AND tc2.table_name = 'track_embeddings';

-- 위 쿼리 결과가 비어있으면 DROP 실행
-- DROP TABLE track_embeddings CASCADE;
