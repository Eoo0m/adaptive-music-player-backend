from supabase import create_client
from dotenv import load_dotenv
import os
import httpx

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# match_tracks_by_key 함수 수정 - cover_image 포함
sql = """
-- 기존 함수 삭제
DROP FUNCTION IF EXISTS match_tracks_by_key(text, integer);

-- 새 함수 생성
CREATE OR REPLACE FUNCTION match_tracks_by_key(input_track_key text, match_count int)
RETURNS TABLE (
    id int,
    track_key text,
    title text,
    artist text,
    album text,
    pos_count int,
    cover_image bytea,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        te.id,
        te.track_key,
        te.title,
        te.artist,
        te.album,
        te.pos_count,
        te.cover_image,
        1 - (te.embedding <=> (
            SELECT embedding
            FROM track_embeddings
            WHERE track_key = input_track_key
        )) AS similarity
    FROM track_embeddings te
    WHERE te.track_key != input_track_key
    ORDER BY te.embedding <=> (
        SELECT embedding
        FROM track_embeddings
        WHERE track_key = input_track_key
    )
    LIMIT match_count;
END;
$$;
"""

async def update_function():
    try:
        print("🔄 Updating match_tracks_by_key function to include cover_image...")

        # Supabase REST API를 사용하여 SQL 실행
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/exec_sql",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json"
                },
                json={"query": sql}
            )

            if response.status_code == 200:
                print("✅ Function updated successfully!")
            else:
                raise Exception(f"Status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"❌ Error executing via API: {e}")
        print("\n" + "="*80)
        print("📝 Please execute this SQL manually in Supabase SQL Editor:")
        print("="*80)
        print(sql)
        print("="*80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(update_function())
