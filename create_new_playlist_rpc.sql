-- new_playlists 테이블을 위한 RPC 함수 생성
-- 기존 match_playlist_embeddings와 동일한 로직이지만 new_playlists 테이블 사용

create or replace function match_new_playlist_embeddings(query_embedding vector(512), match_count int)
returns table (playlist_id text, track_ids text[], similarity float)
language sql
as $$
  select
    _id as playlist_id,
    track_ids,
    1 - (embedding <=> query_embedding) as similarity
  from new_playlists
  order by embedding <=> query_embedding
  limit match_count;
$$;
