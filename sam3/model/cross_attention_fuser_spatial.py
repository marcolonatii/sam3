import torch
import torch.nn as nn

class CrossAttentionFuser(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

        self.norm = nn.GroupNorm(num_groups=1, num_channels=embed_dim)
        
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, sam_features, dino_features):
        B, C, H, W = sam_features.shape
        dino_map = dino_features.transpose(1, 2).reshape(B, C, H, W)
        x = torch.cat([sam_features, dino_map], dim=1)
        delta = self.fuse(x)
        gate = torch.sigmoid(self.gate(x))
        fused_features = sam_features + self.alpha * gate * delta
        return self.norm(fused_features)
