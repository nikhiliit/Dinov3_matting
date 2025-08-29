"""
Multi-Scale Decoder for DINOv3 Alpha Matting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MultiScaleDecoder(nn.Module):
    """Multi-scale decoder for alpha matting from DINOv3 features"""

    def __init__(self, embed_dim: int = 384, decoder_dims: List[int] = [256, 128, 64]):
        super().__init__()

        self.embed_dim = embed_dim
        self.decoder_dims = decoder_dims

        # Initial projection from patch features
        self.input_proj = nn.Conv2d(embed_dim, decoder_dims[0], 1)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        for i, dim in enumerate(decoder_dims):
            # Decoder block
            block = DecoderBlock(dim, decoder_dims[i+1] if i+1 < len(decoder_dims) else 64)
            self.decoder_blocks.append(block)

            # Upsampling block (except for last layer)
            if i < len(decoder_dims) - 1:
                upsample = nn.ConvTranspose2d(
                    decoder_dims[i+1], decoder_dims[i+1], 4, stride=2, padding=1
                )
                self.upsample_blocks.append(upsample)

        # Final alpha prediction
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()  # Alpha values [0, 1]
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize decoder weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, patch_features: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            patch_features: DINOv3 patch features [B, N, D]
            target_size: Target output size (H, W)
        """
        B, N, D = patch_features.shape

        # Reshape to spatial feature map
        # Assume square patches (14x14 for 224x224 input)
        H_patches = W_patches = int(torch.sqrt(torch.tensor(N).float()))
        patch_features = patch_features.transpose(1, 2).reshape(B, D, H_patches, W_patches)

        # Initial projection
        x = self.input_proj(patch_features)  # [B, decoder_dims[0], H_patches, W_patches]

        # Apply decoder blocks with progressive upsampling
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)

            # Upsample (except for last block)
            if i < len(self.upsample_blocks):
                x = self.upsample_blocks[i](x)
                x = F.relu(x)

        # Final alpha prediction
        alpha = self.final_conv(x)

        # Resize to target size
        if alpha.shape[-2:] != target_size:
            alpha = F.interpolate(alpha, size=target_size, mode='bilinear', align_corners=False)

        return alpha


class DecoderBlock(nn.Module):
    """Decoder block with convolution and upsampling"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class BoundaryRefinement(nn.Module):
    """Boundary refinement module for sharp alpha edges"""

    def __init__(self, in_channels: int = 1, hidden_channels: int = 32):
        super().__init__()

        # Boundary detection
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        )

        # Learnable sharpening operation
        self.sharpening_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detect boundaries
        boundary_response = self.boundary_conv(x)

        # Apply residual refinement
        refined = x + self.sharpening_weight * boundary_response

        # Ensure output is in [0, 1]
        refined = torch.clamp(refined, 0, 1)

        return refined
