"""
Alpha Matting Loss Functions for DINOv3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class AlphaReconstructionLoss(nn.Module):
    """Alpha reconstruction loss with Charbonnier penalty"""

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier loss for alpha reconstruction"""
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon)
        return torch.mean(loss)


class GradientLoss(nn.Module):
    """Multi-scale gradient loss for edge preservation"""

    def __init__(self, scales: list = [1, 2, 4]):
        super().__init__()
        self.scales = scales

        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient magnitude using Sobel operators"""
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return grad_mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale gradient loss"""
        total_loss = 0.0

        for scale in self.scales:
            # Downsample for multi-scale processing
            if scale > 1:
                pred_scaled = F.avg_pool2d(pred, scale)
                target_scaled = F.avg_pool2d(target, scale)
            else:
                pred_scaled, target_scaled = pred, target

            # Compute gradient magnitudes
            pred_grad = self.compute_gradient_magnitude(pred_scaled)
            target_grad = self.compute_gradient_magnitude(target_scaled)

            # Weight by target gradient magnitude (focus on important edges)
            weight = torch.exp(target_grad)
            loss = F.l1_loss(pred_grad * weight, target_grad * weight)

            total_loss += loss / scale  # Scale normalization

        return total_loss / len(self.scales)


class LaplacianLoss(nn.Module):
    """Laplacian smoothness loss"""

    def __init__(self):
        super().__init__()

        # Laplacian kernel
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian smoothness loss"""
        laplacian = self.laplacian.to(pred.device)
        pred_lap = F.conv2d(pred, laplacian, padding=1)
        target_lap = F.conv2d(target, laplacian, padding=1)
        return F.l1_loss(pred_lap, target_lap)


class BoundaryAwareLoss(nn.Module):
    """Boundary-aware loss that emphasizes edge regions"""

    def __init__(self, boundary_weight: float = 5.0, boundary_threshold: float = 0.01):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.boundary_threshold = boundary_threshold

    def compute_boundary_mask(self, alpha: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """Detect boundary regions using local variance"""
        kernel = torch.ones(1, 1, kernel_size, kernel_size).to(alpha.device) / (kernel_size**2)
        alpha_mean = F.conv2d(alpha, kernel, padding=kernel_size//2)
        alpha_var = F.conv2d(alpha**2, kernel, padding=kernel_size//2) - alpha_mean**2
        return (alpha_var > self.boundary_threshold).float()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary-aware loss"""
        boundary_mask = self.compute_boundary_mask(target)

        # Base reconstruction loss
        base_loss = F.smooth_l1_loss(pred, target, reduction='none')

        # Apply higher weight to boundary regions
        weighted_loss = base_loss * (1 + self.boundary_weight * boundary_mask)

        return weighted_loss.mean()


class CompositionLoss(nn.Module):
    """Composition loss using RGB foreground and background"""

    def __init__(self):
        super().__init__()

    def forward(self, pred_alpha: torch.Tensor,
                rgb_foreground: torch.Tensor,
                background: torch.Tensor,
                target_composition: torch.Tensor) -> torch.Tensor:
        """
        Compute composition loss

        Args:
            pred_alpha: Predicted alpha [B, 1, H, W]
            rgb_foreground: Foreground RGB [B, 3, H, W]
            background: Background RGB [B, 3, H, W]
            target_composition: Target composite [B, 3, H, W]
        """
        # Generate composite using predicted alpha
        pred_composition = pred_alpha * rgb_foreground + (1 - pred_alpha) * background

        return F.l1_loss(pred_composition, target_composition)


class DINOv3MattingLoss(nn.Module):
    """Combined loss function for DINOv3-based matting"""

    def __init__(self,
                 alpha_weight: float = 1.0,
                 gradient_weight: float = 0.5,
                 laplacian_weight: float = 0.1,
                 boundary_weight: float = 0.0,
                 composition_weight: float = 0.0,
                 use_charbonnier: bool = True):
        super().__init__()

        # Loss weights
        self.alpha_weight = alpha_weight
        self.gradient_weight = gradient_weight
        self.laplacian_weight = laplacian_weight
        self.boundary_weight = boundary_weight
        self.composition_weight = composition_weight

        # Loss functions
        if use_charbonnier:
            self.alpha_loss = AlphaReconstructionLoss()
        else:
            self.alpha_loss = nn.L1Loss()

        self.gradient_loss = GradientLoss(scales=[1, 2, 4])
        self.laplacian_loss = LaplacianLoss()

        # Optional losses
        if boundary_weight > 0:
            self.boundary_loss = BoundaryAwareLoss()
        else:
            self.boundary_loss = None

        if composition_weight > 0:
            self.composition_loss = CompositionLoss()
        else:
            self.composition_loss = None

    def forward(self,
                pred_alpha: torch.Tensor,
                target_alpha: torch.Tensor,
                rgb_foreground: torch.Tensor = None,
                background: torch.Tensor = None,
                target_composition: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute total loss for DINOv3-based matting

        Args:
            pred_alpha: Predicted alpha [B, 1, H, W]
            target_alpha: Ground truth alpha [B, 1, H, W]
            rgb_foreground: Foreground RGB (for composition loss)
            background: Background RGB (for composition loss)
            target_composition: Target composite (for composition loss)
        """
        losses = {}

        # Alpha reconstruction loss
        losses['alpha'] = self.alpha_loss(pred_alpha, target_alpha)

        # Gradient loss for edge preservation
        losses['gradient'] = self.gradient_loss(pred_alpha, target_alpha)

        # Laplacian smoothness loss
        losses['laplacian'] = self.laplacian_loss(pred_alpha, target_alpha)

        # Optional boundary-aware loss
        if self.boundary_loss is not None and self.boundary_weight > 0:
            losses['boundary'] = self.boundary_loss(pred_alpha, target_alpha)
        else:
            losses['boundary'] = torch.tensor(0.0, device=pred_alpha.device)

        # Optional composition loss
        if (self.composition_loss is not None and self.composition_weight > 0 and
            rgb_foreground is not None and background is not None and target_composition is not None):
            losses['composition'] = self.composition_loss(pred_alpha, rgb_foreground, background, target_composition)
        else:
            losses['composition'] = torch.tensor(0.0, device=pred_alpha.device)

        # Compute total weighted loss
        losses['total'] = (
            self.alpha_weight * losses['alpha'] +
            self.gradient_weight * losses['gradient'] +
            self.laplacian_weight * losses['laplacian'] +
            self.boundary_weight * losses['boundary'] +
            self.composition_weight * losses['composition']
        )

        return losses

    def get_loss_info(self) -> Dict[str, Any]:
        """Get loss configuration information"""
        return {
            'alpha_weight': self.alpha_weight,
            'gradient_weight': self.gradient_weight,
            'laplacian_weight': self.laplacian_weight,
            'boundary_weight': self.boundary_weight,
            'composition_weight': self.composition_weight,
            'use_charbonnier': hasattr(self.alpha_loss, 'epsilon'),
            'use_boundary_loss': self.boundary_loss is not None,
            'use_composition_loss': self.composition_loss is not None
        }


def create_dino_alpha_loss(config: Dict[str, Any]) -> DINOv3MattingLoss:
    """Create DINOv3 alpha matting loss from configuration"""

    loss_config = config.get('loss', {})

    return DINOv3MattingLoss(
        alpha_weight=loss_config.get('alpha_weight', 1.0),
        gradient_weight=loss_config.get('gradient_weight', 0.5),
        laplacian_weight=loss_config.get('laplacian_weight', 0.1),
        boundary_weight=loss_config.get('boundary_weight', 0.0),
        composition_weight=loss_config.get('composition_weight', 0.0),
        use_charbonnier=loss_config.get('use_charbonnier', True)
    )
