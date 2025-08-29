"""
DINOv3 Alpha Matting Network
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from .dino_encoder import DINOv3Encoder
from .decoder import MultiScaleDecoder, BoundaryRefinement


class DINOv3AlphaMatting(nn.Module):
    """Complete DINOv3-based alpha matting network"""

    def __init__(self,
                 model_size='vits16',
                 dinov3_path=None,
                 decoder_dims: List[int] = [256, 128, 64],
                 use_boundary_refinement: bool = False,
                 freeze_encoder: bool = True):
        super().__init__()

        self.model_size = model_size
        self.use_boundary_refinement = use_boundary_refinement

        # DINOv3 encoder
        self.encoder = DINOv3Encoder(
            model_size=model_size,
            dinov3_path=dinov3_path,
            freeze_encoder=freeze_encoder
        )

        # Multi-scale decoder
        self.decoder = MultiScaleDecoder(
            embed_dim=self.encoder.embed_dim,
            decoder_dims=decoder_dims
        )

        # Optional boundary refinement
        if use_boundary_refinement:
            self.boundary_refiner = BoundaryRefinement(
                in_channels=1,
                hidden_channels=32
            )
        else:
            self.boundary_refiner = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input RGB image [B, 3, H, W]

        Returns:
            alpha: Predicted alpha matte [B, 1, H, W]
        """
        original_size = x.shape[-2:]

        # Extract DINOv3 features
        patch_features, cls_token = self.encoder(x)

        # Decode to alpha map
        alpha = self.decoder(patch_features, original_size)

        # Optional boundary refinement
        if self.boundary_refiner is not None:
            alpha = self.boundary_refiner(alpha)

        return alpha

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        encoder_info = self.encoder.get_model_info()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_type': 'DINOv3_Alpha_Matting',
            'dino_model_size': self.model_size,
            'embed_dim': encoder_info['embed_dim'],
            'encoder_frozen': encoder_info['frozen'],
            'use_boundary_refinement': self.use_boundary_refinement,
            'decoder_dims': self.decoder.decoder_dims,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_parameters': encoder_info['total_params'],
            'decoder_parameters': total_params - encoder_info['total_params'],
            'model_size_mb': total_params * 4 / 1024 / 1024,
        }

    def freeze_encoder(self):
        """Freeze encoder parameters"""
        self.encoder._freeze_encoder()

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters (for fine-tuning)"""
        for param in self.encoder.dinov3.parameters():
            param.requires_grad = True
        print("🔥 DINOv3 encoder unfrozen (parameters will be updated)")

    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer"""
        return filter(lambda p: p.requires_grad, self.parameters())

    def save_checkpoint(self, path: str, optimizer=None, epoch=None, loss=None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_size': self.model_size,
            'decoder_dims': self.decoder.decoder_dims,
            'use_boundary_refinement': self.use_boundary_refinement,
            'epoch': epoch,
            'loss': loss
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, path: str, optimizer=None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')

        self.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"📦 Checkpoint loaded: {path}")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Loss: {checkpoint.get('loss', 'N/A')}")

        return checkpoint


def create_dino_alpha_model(config: Dict[str, Any]) -> DINOv3AlphaMatting:
    """Factory function to create DINOv3 alpha matting model"""

    model_config = config.get('model', {})

    return DINOv3AlphaMatting(
        model_size=model_config.get('dino_model_size', 'vits16'),
        dinov3_path=model_config.get('dinov3_path'),
        decoder_dims=model_config.get('decoder_dims', [256, 128, 64]),
        use_boundary_refinement=model_config.get('use_boundary_refinement', False),
        freeze_encoder=model_config.get('freeze_encoder', True)
    )
