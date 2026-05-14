# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
DINOv3 encoder wrapper for SAM3.

Extracts patch-level features from an input image and projects them to SAM3's
embedding dimension (256).

SAM3 uses 1008×1008 images with backbone stride 14 → 72×72 feature map.
DINOv3 (ViT-L/16) uses patch size 16; the encoder is configured with
dino_input_size=1024 (default), giving a 64×64 patch grid. This spatial
dimension mismatch is handled by CrossAttentionFuser via cross-attention
(Q=SAM3 72×72 tokens, K/V=DINO 64×64 tokens).

SAM3 normalizes images with (0.5, 0.5, 0.5) mean/std; DINOv3 normalization
is read from AutoImageProcessor and applied after undoing SAM3 normalization.

Output is consumed by CrossAttentionFuser before the SAM mask decoder.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


_DINOV3_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# SAM3 normalization constants: images are normalized to [-1, 1] via (img - 0.5) / 0.5
_SAM3_PIXEL_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_SAM3_PIXEL_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


class DinoEncoder(nn.Module):
    """Wraps DINOv3 (HuggingFace) and projects its patch tokens to ``out_dim``.

    Accepts images normalized with SAM3's convention (ImageNet stats), converts
    them back to [0, 1] range, and re-normalizes with DINOv3's expected stats
    (read from AutoImageProcessor).

    Args:
        model_name: HuggingFace identifier for the DINOv3 model.
        out_dim: Output channel dimension. Must match SAM3's hidden_dim (256).
        freeze_backbone: If True, DINOv3 weights are frozen during training.
        dino_input_size: Spatial size (square) to resize images to before
            passing to DINOv3.  Must be divisible by the patch size (16).
            Default 1024 → 64×64 patch grid (4096 tokens).
            For the conv-based CrossAttentionFuser (which requires pixel-aligned
            features), use 1152 → 72×72 to match SAM3's feature map
            (1008px image / stride 14 = 72).
            For the attention-based CrossAttentionFuser, 1024 works fine since
            cross-attention handles the 72×72 vs 64×64 mismatch.
        sam_pixel_mean: SAM3 pixel mean used to invert SAM normalization.
        sam_pixel_std: SAM3 pixel std used to invert SAM normalization.
    """

    def __init__(
        self,
        model_name: str = _DINOV3_MODEL_ID,
        out_dim: int = 256,
        freeze_backbone: bool = True,
        dino_input_size: int = 1024,
        sam_pixel_mean: Optional[torch.Tensor] = None,
        sam_pixel_std: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        patch_size = 16
        assert dino_input_size % patch_size == 0, (
            f"dino_input_size={dino_input_size} must be divisible by DINOv3 patch size {patch_size}"
        )

        self.dino_input_size = dino_input_size

        processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        dino_embed_dim = self.backbone.config.hidden_size  # 1024 for ViT-L
        self.proj = nn.Linear(dino_embed_dim, out_dim)

        if sam_pixel_mean is None:
            sam_pixel_mean = _SAM3_PIXEL_MEAN.clone()
        if sam_pixel_std is None:
            sam_pixel_std = _SAM3_PIXEL_STD.clone()

        dino_mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        dino_std  = torch.tensor(processor.image_std).view(1, 3, 1, 1)

        self.register_buffer("sam_pixel_mean", sam_pixel_mean)
        self.register_buffer("sam_pixel_std",  sam_pixel_std)
        self.register_buffer("dino_pixel_mean", dino_mean)
        self.register_buffer("dino_pixel_std",  dino_std)

    def _renormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from SAM3 normalization to DINOv3 normalization."""
        x = x * self.sam_pixel_std + self.sam_pixel_mean  # → [0, 1]
        x = x.clamp(0.0, 1.0)
        x = (x - self.dino_pixel_mean) / self.dino_pixel_std
        return x

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Extract and project DINOv3 patch features.

        Args:
            img: SAM3-normalized image tensor of shape ``[B, 3, H, W]``.
                Resized internally to ``dino_input_size × dino_input_size``
                if needed (no-op for SAM3's native 1024×1024 input).

        Returns:
            Projected patch features of shape ``[B, N_patches, out_dim]``,
            where ``N_patches = (dino_input_size // 16) ** 2``.
            For the default 1024 input: ``N_patches = 64 * 64 = 4096``.
        """
        if img.shape[-2] != self.dino_input_size or img.shape[-1] != self.dino_input_size:
            img = F.interpolate(
                img,
                size=(self.dino_input_size, self.dino_input_size),
                mode="bilinear",
                align_corners=False,
            )

        img = self._renormalize(img)

        frozen = not next(iter(self.backbone.parameters())).requires_grad
        if frozen:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=img)
        else:
            outputs = self.backbone(pixel_values=img)

        # last_hidden_state: [B, 1 + N_patches, dino_embed_dim] — skip CLS token
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # [B, N_patches, dino_embed_dim]
        return self.proj(patch_tokens)  # [B, N_patches, out_dim]
