import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============== CaptionPlaylistCLIP 모델 (플레이리스트 검색용) ==============

class PlaylistProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048):
        super().__init__()

        # First projection to hidden dimension
        self.proj_in = nn.Linear(in_dim, hidden_dim)

        # Residual blocks
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
        )

        # Final projection to output dimension
        self.proj_out = nn.Linear(hidden_dim, out_dim)

        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Project to hidden dimension
        h = self.proj_in(x)
        h = self.activation(h)
        h = self.norm(h)

        # Residual block 1
        residual = h
        h = self.block1(h)
        h = h + residual

        # Residual block 2
        residual = h
        h = self.block2(h)
        h = h + residual

        # Project to output dimension
        h = self.proj_out(h)

        # L2 normalize
        return F.normalize(h, dim=-1)


class CaptionPlaylistCLIP(nn.Module):
    def __init__(self, caption_dim, playlist_dim, out_dim=1024, temperature=0.07):
        super().__init__()
        self.caption_proj = PlaylistProjectionMLP(caption_dim, out_dim)
        self.playlist_proj = PlaylistProjectionMLP(playlist_dim, out_dim)
        self.temperature = temperature


# ============== Two-Tower 모델 (세션 기반 추천용) ==============

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class UserTower(nn.Module):
    """User Tower: Encode sequence of item embeddings using a small Transformer."""

    def __init__(self, item_dim, out_dim, hidden_dim=128, n_heads=4, n_layers=2, max_seq_len=16):
        super().__init__()
        self.input_proj = nn.Linear(item_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = self.pos_enc(x)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        x = self.transformer(x, src_key_padding_mask=mask)
        cls_output = x[:, 0, :]
        out = self.output_proj(cls_output)
        out = F.normalize(out, p=2, dim=-1)
        return out


class ItemTower(nn.Module):
    """Item Tower: Project item embedding using MLP."""

    def __init__(self, item_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        out = self.mlp(x)
        out = F.normalize(out, p=2, dim=-1)
        return out


class TwoTowerModel(nn.Module):
    """Two-Tower Model for session-based recommendation."""

    def __init__(self, item_dim, out_dim=128, hidden_dim=128, n_heads=4, n_layers=2, temperature=0.07):
        super().__init__()
        self.user_tower = UserTower(
            item_dim=item_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )
        self.item_tower = ItemTower(
            item_dim=item_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim * 2,
        )
        self.temperature = temperature
        self.out_dim = out_dim

    def encode_user(self, user_seq, user_mask=None):
        """Encode user sequence to embedding."""
        return self.user_tower(user_seq, user_mask)

    def encode_item(self, items):
        """Encode items to embeddings."""
        return self.item_tower(items)


# ============== 모델 로드 ==============

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
title_dim = 3072  # OpenAI text-embedding-3-large
playlist_dim = 64  # SimGCL 트랙 임베딩 차원

# CLIP 모델 로드 (텍스트 → 트랙 임베딩 프로젝션용)
playlist_clip_model = CaptionPlaylistCLIP(
    caption_dim=title_dim, playlist_dim=playlist_dim, out_dim=512
).to(device)
playlist_clip_model.load_state_dict(torch.load("clip_best.pt", map_location=device))
playlist_clip_model.eval()

logger.info(f"✅ CLIP model (clip_best.pt) loaded on {device}")

# Two-Tower 모델 로드 (세션 기반 추천용)
two_tower_model = TwoTowerModel(
    item_dim=64,  # SimGCL 트랙 임베딩 차원
    out_dim=128,
    hidden_dim=128,
    n_heads=4,
    n_layers=2,
).to(device)
two_tower_model.load_state_dict(torch.load("two_tower_best.pt", map_location=device))
two_tower_model.eval()

logger.info(f"✅ Two-Tower model (two_tower_best.pt) loaded on {device}")
