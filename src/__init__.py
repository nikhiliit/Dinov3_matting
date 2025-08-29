"""
DINOv3 Alpha Matting - Source Package
"""

__version__ = "1.0.0"
__author__ = "DINOv3 Alpha Matting Team"

from .models.dino_alpha_net import DINOv3AlphaMatting
from .data.dataset import DINOv3AlphaDataset
from .losses.alpha_losses import DINOv3MattingLoss
from .training.trainer import DINOv3AlphaTrainer

__all__ = [
    'DINOv3AlphaMatting',
    'DINOv3AlphaDataset',
    'DINOv3MattingLoss',
    'DINOv3AlphaTrainer'
]
