# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Cross-attention fuser that combines SAM3 memory-conditioned features (query)
with DINOv3 patch features (key/value) before the SAM mask decoder.

Uses a single cross-attention layer with residual connection and LayerNorm
(Add & Norm), following the standard Transformer decoder pattern.

Input shapes:
    SAM3 features (pix_feat): [B, 256, 64, 64]
    DINOv3 features (dino_feats): [B, 4096, 256]  (64*64 patches, patch size 16)

Output shape:
    Fused features: [B, 256, 64, 64]
"""

import torch
import torch.nn as nn


class CrossAttentionFuser(nn.Module):
    """Fuses SAM3 memory-conditioned features with DINOv3 patch features
    via a single cross-attention layer.

    The SAM3 features serve as the query; DINOv3 features serve as key and value.
    A residual connection preserves the original SAM3 features, so the module
    defaults to identity-like behavior before training.

    Args:
        embed_dim: Embedding dimension for Q, K, V (must match SAM3's
            hidden_dim, default 256).
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(0.1))  # Scaling factor for the cross-attention output before adding to SAM features

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 0.0)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        sam_features: torch.Tensor,
        dino_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse SAM3 and DINOv3 features via cross-attention.

        Args:
            sam_features: SAM3 memory-conditioned features of shape
                ``[B, C, H, W]`` (BCHW), typically ``[B, 256, 64, 64]``.
            dino_features: DINOv3 projected patch tokens of shape
                ``[B, N_patches, C]``, typically ``[B, 4096, 256]``.

        Returns:
            Fused features of the same shape as ``sam_features`` (BCHW).
        """
        B, C, H, W = sam_features.shape

        # Flatten SAM features: [B, C, H, W] → [B, H*W, C]
        q = sam_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Cross-attention: Q=SAM, K=V=DINO
        attn_out, _ = self.cross_attn(
            query=q,
            key=dino_features,
            value=dino_features,
        )  # [B, H*W, C]

        # Compute gate
        gate = torch.sigmoid(self.gate(torch.cat([q, attn_out], dim=-1)))

        # Add & Norm (gated residual from original SAM features)
        fused = self.norm(q + self.alpha * gate * attn_out)  # [B, H*W, C]

        # Reshape back to BCHW
        return fused.permute(0, 2, 1).reshape(B, C, H, W)
