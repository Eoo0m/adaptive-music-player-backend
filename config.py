"""
config.py — 테이블명 중앙 관리

환경변수 TABLE_SUFFIX 로 suffix 지정 (기본: 빈 문자열 = suffix 없음)

예시 (.env):
    TABLE_SUFFIX=         # 기존 테이블 (tracks, track_vectors_simgcl, track_vectors_clip)
    TABLE_SUFFIX=261128   # 새 테이블 (tracks_261128, track_vectors_simgcl_261128, ...)

사용법:
    from config import T, TV_SIMGCL, TV_CLIP
"""

import os

_suffix = os.environ.get("TABLE_SUFFIX", "").strip()

def _tbl(base: str) -> str:
    return f"{base}_{_suffix}" if _suffix else base

# suffix 적용 테이블 (새 학습 결과 반영)
T          = _tbl("tracks")               # tracks / tracks_261128
TV_SIMGCL  = _tbl("track_vectors_simgcl") # track_vectors_simgcl / track_vectors_simgcl_261128
TV_CLIP    = _tbl("track_vectors_clip")   # track_vectors_clip / track_vectors_clip_261128

# suffix 미적용 테이블 (유저 데이터 / 캐시 — 항상 고정)
KEYWORD_EMBEDDINGS    = "keyword_embeddings"
QUERY_MOOD_CACHE      = "query_mood_cache"
USER_QUERY_SUGGESTIONS = "user_query_suggestions"
SEARCH_KEYWORDS       = "search_keywords"
USER_FAVORITES        = "user_favorites"
USER_ACTION_LOGS      = "user_action_logs"
USER_INTEREST_CLUSTERS = "user_interest_clusters"
USER_MAPS             = "user_maps"
USER_PLAYLISTS        = "user_playlists"
USERS                 = "users"
TRACK_COMMENTS        = "track_comments"
