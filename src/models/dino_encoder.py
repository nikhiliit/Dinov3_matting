"""
DINOv3 Encoder for Alpha Matting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
from typing import Tuple, Optional

# Add DINOv3 to path
current_dir = os.path.dirname(os.path.abspath(__file__))
dino_path = os.path.join(current_dir, '../../../dinov3')
if dino_path not in sys.path:
    sys.path.insert(0, dino_path)


class DINOv3Encoder(nn.Module):
    """DINOv3 Encoder for alpha matting with patch processing"""

    def __init__(self, model_size='vits16', dinov3_path=None, freeze_encoder=True):
        super().__init__()

        self.model_size = model_size
        self.freeze_encoder = freeze_encoder

        # Set default path if not provided
        if dinov3_path is None:
            dinov3_path = os.path.join(current_dir, '../../../dinov3')

        self.dinov3_path = dinov3_path

        # Load DINOv3 model
        self._load_dinov3_model()

        # DINOv3 patch configuration
        self.patch_size = 16  # DINOv3 uses 16x16 patches
        self.embed_dim = 384 if 'vits' in model_size else 768  # vit-s: 384, vit-b/l: 768

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

    def _load_dinov3_model(self):
        """Load DINOv3 model"""
        # Try to load from local checkpoint first
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')

        model_files = {
            'vits16': 'dinov3_vits16_pretrain_lvd1689m.pth',
            'vitb16': 'dinov3_vitb16_pretrain_lvd1689m.pth',
            'vitl16': 'dinov3_vitl16_pretrain_lvd1689m.pth'
        }

        model_path = os.path.join(local_model_path, model_files[self.model_size])

        if not os.path.exists(model_path):
            print(f"⚠️  Local model not found at {model_path}, trying hub download...")
            model_path = None

        try:
            # Import DINOv3 hub
            dino_import_path = os.path.join(self.dinov3_path)
            if dino_import_path not in sys.path:
                sys.path.insert(0, dino_import_path)

            if model_path and os.path.exists(model_path):
                # Load from local checkpoint
                print(f"📁 Loading DINOv3 {self.model_size} from local checkpoint: {model_path}")

                # Import the model architecture
                from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16

                model_funcs = {
                    'vits16': dinov3_vits16,
                    'vitb16': dinov3_vitb16,
                    'vitl16': dinov3_vitl16
                }

                # Create model without pretrained weights first
                self.dinov3 = model_funcs[self.model_size](pretrained=False)

                # Load state dict
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint

                # Handle state dict key mismatches
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        k = k[7:]  # Remove 'module.' prefix if present
                    new_state_dict[k] = v

                self.dinov3.load_state_dict(new_state_dict, strict=False)
                print(f"✅ DINOv3 {self.model_size} loaded from local checkpoint")

            else:
                # Fallback to hub download
                print(f"🌐 Downloading DINOv3 {self.model_size} from hub...")
                from dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16

                model_funcs = {
                    'vits16': dinov3_vits16,
                    'vitb16': dinov3_vitb16,
                    'vitl16': dinov3_vitl16
                }

                self.dinov3 = model_funcs[self.model_size]()

            self.dinov3.eval()

        except Exception as e:
            print(f"❌ DINOv3 model loading failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _freeze_encoder(self):
        """Freeze DINOv3 encoder parameters"""
        for param in self.dinov3.parameters():
            param.requires_grad = False
        print("🧊 DINOv3 encoder frozen (no gradient updates)")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract DINOv3 features

        Args:
            x: Input RGB image [B, 3, H, W]

        Returns:
            patch_features: Patch token features [B, N, D]
            cls_token: CLS token [B, D] (optional)
        """
        B, C, H, W = x.shape

        # DINOv3 expects input size to be multiple of patch_size
        # Resize if necessary
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            target_h = ((H // self.patch_size) + 1) * self.patch_size
            target_w = ((W // self.patch_size) + 1) * self.patch_size
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)

        with torch.no_grad():  # No gradients for frozen encoder
            # Extract features
            features = self.dinov3.forward_features(x)

            # Get patch tokens and cls token
            if isinstance(features, dict):
                patch_tokens = features.get('x_norm_patchtokens')  # [B, N, D]
                cls_token = features.get('x_norm_clstoken', None)  # [B, D]

                if cls_token is not None and len(cls_token.shape) == 3:
                    cls_token = cls_token.squeeze(1)  # [B, D]
            else:
                # Fallback: assume all tokens are patch tokens
                patch_tokens = features
                cls_token = None

        return patch_tokens, cls_token

    def get_model_info(self) -> dict:
        """Get encoder information"""
        return {
            'model_size': self.model_size,
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size,
            'frozen': self.freeze_encoder,
            'total_params': sum(p.numel() for p in self.dinov3.parameters()),
            'trainable_params': sum(p.numel() for p in self.dinov3.parameters() if p.requires_grad)
        }
