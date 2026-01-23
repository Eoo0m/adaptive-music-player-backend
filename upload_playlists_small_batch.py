"""
플레이리스트 업로드 - 작은 배치 + 재시도 로직
"""
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm
import time

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 파일 경로
CSV_PATH = "/Users/eomjoonseo/dynplayer/clip/playlists_with_valid_tracks_cleaned.csv"
NPY_EMBEDDING_PATH = "/Users/eomjoonseo/dynplayer/clip/clip_u10_valid_tracks_playlist_projected.npy"
NPY_IDS_PATH = "/Users/eomjoonseo/dynplayer/clip/clip_u10_valid_tracks_playlist_ids.npy"

print("=" * 80)
print("📊 플레이리스트 업로드 (작은 배치 + 재시도)")
print("=" * 80)

# 1. 현재 테이블 상태 확인
print("\n1️⃣ 테이블 임베딩 타입 확인...")
test = supabase.table("new_playlists").select("embedding").limit(1).execute()
if test.data and len(test.data) > 0:
    emb = test.data[0]['embedding']
    if isinstance(emb, str):
        print("   ❌ WARNING: embedding이 문자열 타입입니다!")
        print("   💡 Supabase에서 다음 SQL을 실행해야 합니다:")
        print("   DROP TABLE new_playlists CASCADE;")
        print("   그 다음 fix_table_schema_and_reupload.sql 실행")

        user_input = input("\n   계속하시겠습니까? (y/n): ")
        if user_input.lower() != 'y':
            print("   중단됨")
            exit(0)
    elif isinstance(emb, list):
        print(f"   ✅ embedding이 vector 타입입니다! (차원: {len(emb)})")
else:
    print("   ℹ️  테이블이 비어있음 - 새로 업로드합니다")

# 2. 데이터 로드
print("\n2️⃣ 데이터 로드 중...")
df = pd.read_csv(CSV_PATH)
embeddings = np.load(NPY_EMBEDDING_PATH)
playlist_ids = np.load(NPY_IDS_PATH, allow_pickle=True)
print(f"   ✅ CSV: {len(df):,}개")
print(f"   ✅ Embeddings: {embeddings.shape}")
print(f"   ✅ IDs: {len(playlist_ids):,}개")

# 3. 데이터 매칭
print("\n3️⃣ 데이터 매칭 중...")
ids_df = pd.DataFrame({'playlist_id': playlist_ids, 'npy_index': range(len(playlist_ids))})
merged_df = ids_df.merge(df, on='playlist_id', how='left')
missing_count = merged_df['playlist_title'].isna().sum()

if missing_count > 0:
    print(f"   ⚠️ {missing_count}개 제거")
    merged_df = merged_df.dropna(subset=['playlist_title']).reset_index(drop=True)
    valid_indices = merged_df['npy_index'].tolist()
    embeddings = embeddings[valid_indices]

df = merged_df.drop('npy_index', axis=1)
print(f"   ✅ 최종: {len(df):,}개")

# 4. 업로드
print("\n4️⃣ 배치 업로드 중...")

def parse_track_ids(track_ids_str):
    if pd.isna(track_ids_str):
        return []
    return [t.strip() for t in track_ids_str.split("|") if t.strip()]

def format_pgvector(embedding_array):
    values = ','.join(f"{x:.8f}" for x in embedding_array)
    return f"[{values}]"

# 작은 배치 사이즈 (50개)
batch_size = 50
total_batches = (len(df) + batch_size - 1) // batch_size

success_count = 0
error_count = 0
failed_batches = []

for batch_idx in tqdm(range(total_batches), desc="업로드"):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(df))

    batch_data = []
    for i in range(start_idx, end_idx):
        row = df.iloc[i]
        batch_data.append({
            "playlist_id": row['playlist_id'],
            "playlist_title": row['playlist_title'],
            "saves": int(row['saves']) if pd.notna(row['saves']) else 0,
            "track_ids": parse_track_ids(row['track_ids']),
            "embedding": format_pgvector(embeddings[i])
        })

    # 재시도 로직 (최대 3번)
    retry_count = 0
    max_retries = 3
    success = False

    while retry_count < max_retries and not success:
        try:
            result = supabase.table("new_playlists").insert(batch_data).execute()
            success_count += len(batch_data)
            success = True
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"\n   ⚠️ Batch {batch_idx} 실패 (시도 {retry_count}/{max_retries}): {e}")
                print(f"   🔄 2초 대기 후 재시도...")
                time.sleep(2)
            else:
                print(f"\n   ❌ Batch {batch_idx} 최종 실패: {e}")
                error_count += len(batch_data)
                failed_batches.append(batch_idx)

print(f"\n✅ 업로드 완료!")
print(f"   성공: {success_count:,}개")
print(f"   실패: {error_count:,}개")

if failed_batches:
    print(f"\n   실패한 배치: {failed_batches[:10]}" + ("..." if len(failed_batches) > 10 else ""))

# 5. 검증
print("\n5️⃣ 검증 중...")
result = supabase.table("new_playlists").select("id", count="exact").execute()
print(f"   📊 업로드된 행: {result.count:,}")
print(f"   📊 예상 행: {len(df):,}")

if result.count == len(df):
    print("\n🎉 성공!")
else:
    print(f"\n⚠️ {len(df) - result.count}개 누락")

# 6. 타입 재확인
print("\n6️⃣ 임베딩 타입 재확인...")
sample = supabase.table("new_playlists").select("embedding").limit(1).execute()
if sample.data:
    emb = sample.data[0]['embedding']
    if isinstance(emb, list):
        print(f"   ✅ vector 타입! (차원: {len(emb)})")
    else:
        print(f"   ❌ 여전히 {type(emb).__name__} 타입!")

print("\n" + "=" * 80)
