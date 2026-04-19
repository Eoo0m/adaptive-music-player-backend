import time
import json
import logging
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from database import get_db_pool
from models import two_tower_model, device

logger = logging.getLogger(__name__)

router = APIRouter()


class RecommendAverageRequest(BaseModel):
    track_keys: List[str]
    num_recommendations: Optional[int] = 5


@router.post("/recommend")
async def recommend(request: RecommendAverageRequest):
    """세션 기반 추천 (Two-Tower 모델 사용)"""
    if not request.track_keys or len(request.track_keys) == 0:
        raise HTTPException(status_code=400, detail="Missing track_keys")

    try:
        logger.info(f"🎯 Two-Tower recommend with {len(request.track_keys)} session tracks")

        pool = await get_db_pool()

        # 1. 모든 track_key의 임베딩을 한 번에 가져오기 (raw SQL)
        rows = await pool.fetch(
            """
            SELECT track_key, embedding
            FROM track_embeddings
            WHERE track_key = ANY($1)
            """,
            request.track_keys
        )

        # track_key 순서대로 임베딩 정렬
        embedding_map = {}
        for row in rows:
            embedding = row.get("embedding")
            if embedding:
                if isinstance(embedding, str):
                    embedding = json.loads(embedding)
                embedding_map[row["track_key"]] = np.array(embedding, dtype=np.float32)

        # 요청 순서대로 임베딩 리스트 생성
        embeddings = [embedding_map[tk] for tk in request.track_keys if tk in embedding_map]

        if len(embeddings) == 0:
            raise HTTPException(
                status_code=404, detail="No embeddings found for provided track_keys"
            )

        logger.info(f"📦 Fetched {len(embeddings)} embeddings in single batch query")

        # 2. User Tower로 세션 임베딩 생성 (128차원)
        user_seq = torch.tensor(np.stack(embeddings), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            user_embedding = two_tower_model.encode_user(user_seq)  # (1, 128)

        user_embedding_list = user_embedding.squeeze(0).cpu().numpy().tolist()
        logger.info(f"✅ User embedding computed from {len(embeddings)} tracks")

        # 3. DB에서 itemtower_embedding과 직접 비교하여 추천 (raw SQL)
        t_db = time.time()
        rows = await pool.fetch(
            """
            SELECT
                t.id,
                t.track_key::text,
                t.title::text,
                t.artist::text,
                t.album::text,
                t.playlist_count,
                t.cover_image_url::text,
                (1 - (t.itemtower_embedding <=> $1::vector))::float AS similarity
            FROM track_embeddings t
            WHERE t.itemtower_embedding IS NOT NULL
              AND t.track_key::text != ALL($2)
            ORDER BY t.itemtower_embedding <=> $1::vector
            LIMIT $3
            """,
            str(user_embedding_list),
            request.track_keys,
            request.num_recommendations
        )
        db_time = time.time() - t_db
        logger.info(f"📊 /recommend DB query time: {db_time:.3f}s")

        if not rows:
            raise HTTPException(status_code=500, detail="No recommendations found")

        # 4. 결과 포맷 변환
        recommendations = []
        for item in rows:
            recommendations.append({
                "track_id": item["id"],
                "track_key": item["track_key"],
                "track": item["title"],
                "artist": item["artist"],
                "album": item["album"],
                "playlist_count": item["playlist_count"],
                "similarity": item.get("similarity", 0),
                "cover_image_url": item.get("cover_image_url"),
            })

        logger.info(f"Two-Tower recommend: {len(recommendations)} tracks from {len(embeddings)} session tracks")

        return {
            "recommendations": recommendations,
            "num_session_tracks": len(embeddings),
        }

    except Exception as e:
        logger.error(f"Two-Tower recommend error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Session-based recommendation failed: {str(e)}"
        )
