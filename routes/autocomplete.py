import math
import logging
from fastapi import APIRouter, Query
from database import get_db_pool

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/search-autocomplete")
async def search_autocomplete(q: str = Query(..., min_length=1)):
    """
    자동완성 제안 반환.
    스코어 = prefix_bonus(1.0) + trgm_sim*0.4 + log(search_count+1)*0.3
    """
    q = q.strip()
    if not q:
        return {"suggestions": []}

    try:
        pool = await get_db_pool()
        rows = await pool.fetch(
            """
            SELECT
                keyword,
                search_count,
                similarity(lower(keyword), lower($1)) AS trgm_sim,
                CASE WHEN lower(keyword) LIKE lower($1) || '%' THEN 1.0 ELSE 0.0 END AS prefix_bonus
            FROM search_keywords
            WHERE
                lower(keyword) LIKE lower($1) || '%'
                OR similarity(lower(keyword), lower($1)) > 0.2
            ORDER BY
                (
                    CASE WHEN lower(keyword) LIKE lower($1) || '%' THEN 1.0 ELSE 0.0 END
                    + similarity(lower(keyword), lower($1)) * 0.4
                    + ln(GREATEST(search_count, 0) + 1) * 0.3
                ) DESC
            LIMIT 8
            """,
            q
        )
        suggestions = [row["keyword"] for row in rows]
        return {"suggestions": suggestions}

    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        return {"suggestions": []}
