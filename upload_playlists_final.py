"""
new_playlists 테이블 데이터 업로드 (최종 버전)
- TRUNCATE로 기존 데이터 완전 삭제
- 테이블 스키마 자동 확인
- pgvector 형식으로 올바르게 업로드
"""
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm

# 환경 변수 로드
load_dotenv()

# Supabase 클라이언트
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 파일 경로
CSV_PATH = "/Users/eomjoonseo/dynplayer/clip/playlists_with_valid_tracks_cleaned.csv"
NPY_EMBEDDING_PATH = "/Users/eomjoonseo/dynplayer/clip/clip_u10_valid_tracks_playlist_projected.npy"
NPY_IDS_PATH = "/Users/eomjoonseo/dynplayer/clip/clip_u10_valid_tracks_playlist_ids.npy"

print("=" * 80)
print("📊 new_playlists 테이블 업로드 (최종 버전)")
print("=" * 80)

# 1. 테이블 스키마 확인
print("\n1️⃣ 테이블 스키마 확인 중...")
try:
    schema_query = """
    SELECT column_name, data_type, udt_name
    FROM information_schema.columns
    WHERE table_name = 'new_playlists' AND column_name = 'embedding';
    """
    result = supabase.rpc("exec_sql", {"query": schema_query}).execute()
    print(f"   ⚠️ RPC exec_sql 사용 불가. 스키마 확인 스킵.")
except Exception as e:
    print(f"   ⚠️ 스키마 확인 불가: {e}")
    print(f"   💡 embedding 컬럼이 vector(512) 타입인지 수동 확인 필요!")
    print(f"   💡 SQL: ALTER TABLE new_playlists ALTER COLUMN embedding TYPE vector(512);")

# 2. 테이블 초기화 (TRUNCATE)
print("\n2️⃣ 테이블 초기화 중 (TRUNCATE)...")
try:
    # postgrest를 통해 TRUNCATE 실행 (모든 행 삭제 + ID 리셋)
    # Supabase Python 클라이언트는 TRUNCATE를 직접 지원하지 않으므로
    # 모든 행을 삭제하는 방식 사용

    # 먼저 현재 행 수 확인
    count_result = supabase.table("new_playlists").select("id", count="exact").execute()
    current_count = count_result.count
    print(f"   📊 현재 행 수: {current_count}")

    if current_count > 0:
        print(f"   🗑️  {current_count}개 행 삭제 중...")
        # 모든 행 삭제 (대량 삭제는 시간이 걸릴 수 있음)
        # neq("id", 0)을 사용하여 모든 행 선택
        delete_result = supabase.table("new_playlists").delete().neq("id", 0).execute()
        print(f"   ✅ 테이블 초기화 완료")
    else:
        print(f"   ✅ 테이블이 이미 비어있음")

except Exception as e:
    print(f"   ❌ 테이블 초기화 실패: {e}")
    print(f"\n   💡 Supabase SQL Editor에서 직접 실행하세요:")
    print(f"   TRUNCATE TABLE new_playlists RESTART IDENTITY CASCADE;")
    user_input = input("\n   SQL 실행 후 계속하려면 Enter를 누르세요 (취소: Ctrl+C): ")

# 3. CSV 파일 로드
print("\n3️⃣ CSV 파일 로드 중...")
df = pd.read_csv(CSV_PATH)
print(f"   ✅ {len(df):,} 개 플레이리스트 로드")

# 4. NPY 파일들 로드
print("\n4️⃣ NPY 파일들 로드 중...")
embeddings = np.load(NPY_EMBEDDING_PATH)
playlist_ids = np.load(NPY_IDS_PATH, allow_pickle=True)
print(f"   ✅ Embedding shape: {embeddings.shape}")
print(f"   ✅ Playlist IDs: {len(playlist_ids):,}개")

# 5. 데이터 검증
if embeddings.shape[0] != len(playlist_ids):
    print(f"\n❌ 오류: 임베딩 개수({embeddings.shape[0]})와 ID 개수({len(playlist_ids)})가 다릅니다!")
    exit(1)

# 6. playlist_ids 기준으로 CSV 데이터 매칭
print("\n5️⃣ Playlist ID 기준으로 데이터 매칭 중...")
ids_df = pd.DataFrame({
    'playlist_id': playlist_ids,
    'npy_index': range(len(playlist_ids))
})

merged_df = ids_df.merge(df, on='playlist_id', how='left')
print(f"   📊 매칭 결과: {len(merged_df):,}개")

# 누락된 데이터 확인 및 제거
missing_count = merged_df['playlist_title'].isna().sum()
if missing_count > 0:
    print(f"   ⚠️ {missing_count}개 플레이리스트가 CSV에 없음 (제거)")
    merged_df = merged_df.dropna(subset=['playlist_title']).reset_index(drop=True)
    valid_indices = merged_df['npy_index'].tolist()
    embeddings = embeddings[valid_indices]

df = merged_df.drop('npy_index', axis=1)
print(f"   ✅ 최종 업로드 대상: {len(df):,}개")

# 7. 데이터 업로드
print("\n6️⃣ 데이터 업로드 중...")

def parse_track_ids(track_ids_str):
    """파이프 구분 문자열을 리스트로 변환"""
    if pd.isna(track_ids_str):
        return []
    return [t.strip() for t in track_ids_str.split("|") if t.strip()]

def format_pgvector(embedding_array):
    """NumPy array를 pgvector 문자열 형식으로 변환"""
    # 숫자를 문자열로 변환하되, 과학적 표기법 방지
    values = ','.join(f"{x:.8f}" for x in embedding_array)
    return f"[{values}]"

batch_size = 100
total_batches = (len(df) + batch_size - 1) // batch_size
success_count = 0
error_count = 0

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

    try:
        result = supabase.table("new_playlists").insert(batch_data).execute()
        success_count += len(batch_data)
    except Exception as e:
        print(f"\n   ❌ Batch {batch_idx} 실패: {e}")
        error_count += len(batch_data)
        # 첫 에러 발생 시 중단
        if batch_idx == 0:
            print("\n   💡 첫 번째 배치부터 실패. 테이블 스키마 확인 필요:")
            print("   ALTER TABLE new_playlists ALTER COLUMN embedding TYPE vector(512);")
            break

print(f"\n✅ 업로드 완료!")
print(f"   성공: {success_count:,}개")
print(f"   실패: {error_count:,}개")

# 8. 검증
print("\n7️⃣ 업로드 검증 중...")
result = supabase.table("new_playlists").select("id", count="exact").execute()
print(f"   📊 테이블 총 행 수: {result.count:,}")
print(f"   ✅ 예상 행 수: {len(df):,}")

if result.count == len(df):
    print("\n🎉 성공!")
else:
    print(f"\n⚠️ 경고: 행 수 불일치 ({result.count} vs {len(df)})")

# 9. 임베딩 형식 검증
print("\n8️⃣ 임베딩 형식 검증 중...")
sample = supabase.table("new_playlists").select("playlist_id, embedding").limit(1).execute()

if sample.data:
    emb = sample.data[0]['embedding']
    print(f"   Type: {type(emb)}")

    if isinstance(emb, str):
        print(f"   ❌ 문자열로 저장됨!")
        print(f"   💡 다음 SQL을 실행하세요:")
        print(f"   ALTER TABLE new_playlists ALTER COLUMN embedding TYPE vector(512);")
    elif isinstance(emb, list):
        print(f"   ✅ 리스트로 저장됨 (길이: {len(emb)})")
        if len(emb) == 512:
            print(f"   ✅ 차원 확인: 512 ✓")
            print(f"   📋 샘플 (처음 5개): {emb[:5]}")
        else:
            print(f"   ❌ 차원 오류: {len(emb)} (예상: 512)")
    else:
        print(f"   ⚠️ 알 수 없는 타입: {type(emb)}")

print("\n" + "=" * 80)
