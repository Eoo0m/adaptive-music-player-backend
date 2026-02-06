-- 모든 RPC 함수를 새 테이블(new_playlists, new_track_embeddings)을 사용하도록 업데이트
-- 기존 함수 이름은 유지하고 내부 테이블 참조만 변경

-- 1. match_playlist_embeddings: new_playlists 테이블 사용
-- 로컬 search_playlists.py와 동일하게 내적(inner product) 사용
-- L2 정규화된 벡터의 내적 = 코사인 유사도 (범위: -1 ~ 1, 일반적으로 0 ~ 1)
drop function if exists match_playlist_embeddings(vector, integer);
create or replace function match_playlist_embeddings(query_embedding vector(512), match_count int)
returns table (playlist_id text, track_ids text[], similarity float)
language sql
as $$
  select
    playlist_id::text as playlist_id,
    track_ids,
    (embedding <#> query_embedding) * -1 as similarity
  from new_playlists
  order by embedding <#> query_embedding
  limit match_count;
$$;

-- 2. search_tracks_by_title: new_track_embeddings 테이블 사용
drop function if exists search_tracks_by_title(text, integer);
create or replace function search_tracks_by_title(query_text text, match_count int)
returns table (
    id int,
    track_key text,
    title text,
    artist text,
    album text,
    pos_count int,
    cover_image_url text,
    similarity float
)
language sql
as $$
  select
    id,
    track_key,
    title,
    artist,
    album,
    pos_count,
    cover_image_url,
    similarity(title, query_text) as similarity
  from new_track_embeddings
  where title ilike '%' || query_text || '%'
     or artist ilike '%' || query_text || '%'
  order by similarity(title, query_text) desc
  limit match_count;
$$;

-- 3. match_tracks_by_key: new_track_embeddings 테이블 사용
drop function if exists match_tracks_by_key(text, integer);
create or replace function match_tracks_by_key(input_track_key text, match_count int)
returns table (
    id int,
    track_key text,
    title text,
    artist text,
    album text,
    pos_count int,
    cover_image_url text,
    similarity float
)
language sql
as $$
  select
    te.id,
    te.track_key,
    te.title,
    te.artist,
    te.album,
    te.pos_count,
    te.cover_image_url,
    1 - (te.embedding <=> (
      select embedding
      from new_track_embeddings
      where track_key = input_track_key
    )) as similarity
  from new_track_embeddings te
  where te.track_key != input_track_key
  order by te.embedding <=> (
    select embedding
    from new_track_embeddings
    where track_key = input_track_key
  )
  limit match_count;
$$;

-- 4. search_tracks_by_keyword_fast: 키워드 검색을 한 번에 처리
drop function if exists search_tracks_by_keyword_fast(vector(512), integer, integer);
create or replace function search_tracks_by_keyword_fast(
    query_embedding vector(512),
    playlist_count int default 100,
    track_count int default 10
)
returns table (
    track_key text,
    title text,
    artist text,
    album text,
    pos_count int,
    cover_image_url text
)
language sql
as $$
  with top_playlists as (
    select
      track_ids,
      (embedding <#> query_embedding) * -1 as similarity
    from new_playlists
    order by embedding <#> query_embedding
    limit playlist_count
  ),
  track_scores as (
    select
      unnest(track_ids) as tk,
      sum(similarity) as score
    from top_playlists
    group by tk
  ),
  top_tracks as (
    select tk
    from track_scores
    order by score desc
    limit track_count
  )
  select
    tt.tk as track_key,
    te.title,
    te.artist,
    te.album,
    te.pos_count,
    te.cover_image_url
  from top_tracks tt
  join new_track_embeddings te on te.track_key = tt.tk;
$$;

-- 5. match_tracks_by_embedding: 평균 임베딩으로 유사 트랙 검색
drop function if exists match_tracks_by_embedding(vector(512), integer);
create or replace function match_tracks_by_embedding(
    query_embedding vector(512),
    match_count int default 30
)
returns table (
    id int,
    track_key text,
    title text,
    artist text,
    album text,
    pos_count int,
    cover_image_url text,
    similarity float
)
language sql
as $$
  select
    id,
    track_key,
    title,
    artist,
    album,
    pos_count,
    cover_image_url,
    1 - (embedding <=> query_embedding) as similarity
  from new_track_embeddings
  order by embedding <=> query_embedding
  limit match_count;
$$;
