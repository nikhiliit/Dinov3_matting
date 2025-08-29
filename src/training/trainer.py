"""
DINOv3 Alpha Matting Trainer
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path
import time
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..models.dino_alpha_net import DINOv3AlphaMatting, create_dino_alpha_model
from ..losses.alpha_losses import DINOv3MattingLoss, create_dino_alpha_loss
from ..data.dataset import create_data_loaders
from ..utils.helpers import save_config, create_output_directories


class DINOv3AlphaTrainer:
    """Trainer class for DINOv3-based alpha matting"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer

        Args:
            config: Training configuration
        """
        self.config = config
        self.device = self._setup_device()

        # Create output directories
        self.output_dirs = create_output_directories(config.get('output_dir', './outputs'))

        # Initialize model, loss, optimizer
        self.model = self._setup_model()
        self.loss_fn = self._setup_loss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Setup data loaders
        self.train_loader, self.val_loader = create_data_loaders(config)

        # Setup logging
        self.writer = self._setup_tensorboard()

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_stats = []

        print("✅ DINOv3 Alpha Matting Trainer initialized")
        print(f"📁 Outputs: {self.output_dirs['root']}")
        print(f"🎯 Device: {self.device}")
        print(f"📊 Train samples: {len(self.train_loader.dataset)}")
        print(f"📊 Val samples: {len(self.val_loader.dataset)}")

    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if self.config.get('device', 'auto') == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config['device'])

        # Move model to device
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return device

    def _setup_model(self) -> DINOv3AlphaMatting:
        """Setup model"""
        model = create_dino_alpha_model(self.config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("🤖 Model created:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Encoder frozen: {model.encoder.freeze_encoder}")

        return model.to(self.device)

    def _setup_loss(self) -> DINOv3MattingLoss:
        """Setup loss function"""
        loss_fn = create_dino_alpha_loss(self.config)
        loss_info = loss_fn.get_loss_info()

        print("📉 Loss function configured:")
        for key, value in loss_info.items():
            print(f"   {key}: {value}")

        return loss_fn.to(self.device)

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        optimizer_config = self.config.get('training', {}).get('optimizer', {})

        if optimizer_config.get('name', 'adamw').lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.get_trainable_parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 0.01),
                betas=optimizer_config.get('betas', [0.9, 0.999])
            )
        else:
            optimizer = optim.Adam(
                self.model.get_trainable_parameters(),
                lr=optimizer_config.get('lr', 1e-4),
                betas=optimizer_config.get('betas', [0.9, 0.999])
            )

        print(f"⚡ Optimizer: {optimizer_config.get('name', 'adamw')}")
        print(f"   Learning rate: {optimizer_config.get('lr', 1e-4)}")

        return optimizer

    def _setup_scheduler(self) -> Optional[CosineAnnealingLR]:
        """Setup learning rate scheduler"""
        scheduler_config = self.config.get('training', {}).get('scheduler', {})

        if scheduler_config.get('name', '').lower() == 'cosineannealinglr':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('training', {}).get('num_epochs', 50),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
            print(f"📅 Scheduler: CosineAnnealingLR (eta_min: {scheduler_config.get('eta_min', 1e-6)})")
        else:
            scheduler = None
            print("📅 Scheduler: None")

        return scheduler

    def _setup_tensorboard(self) -> SummaryWriter:
        """Setup TensorBoard logging"""
        log_dir = os.path.join(self.output_dirs['logs'], 'tensorboard')
        writer = SummaryWriter(log_dir)
        print(f"📝 TensorBoard logging: {log_dir}")
        return writer

    def train(self):
        """Main training loop"""
        print("\n🚀 Starting DINOv3 Alpha Matting Training")
        print("=" * 60)

        num_epochs = self.config.get('training', {}).get('num_epochs', 50)
        save_interval = self.config.get('training', {}).get('save_interval', 5)
        eval_interval = self.config.get('training', {}).get('eval_interval', 5)
        grad_clip_norm = self.config.get('training', {}).get('grad_clip_norm', 1.0)

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_losses = self._train_epoch()
            avg_train_loss = np.mean([loss['total'] for loss in train_losses])

            # Validate
            if epoch % eval_interval == 0:
                val_losses = self._validate_epoch()
                avg_val_loss = np.mean([loss['total'] for loss in val_losses])
            else:
                avg_val_loss = None

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Log progress
            self._log_epoch(epoch, avg_train_loss, avg_val_loss, train_losses, val_losses)

            # Save checkpoint
            if epoch % save_interval == 0:
                self._save_checkpoint()

            # Save best model
            if avg_val_loss is not None and avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self._save_checkpoint(filename='best_model.pth')

        # Final save
        self._save_checkpoint(filename='final_model.pth')

        # Training summary
        total_time = time.time() - start_time
        print("\n🎉 Training completed!")
        print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        print(f"🏆 Best validation loss: {self.best_loss:.4f}")
        print(f"💾 Final model saved: {os.path.join(self.output_dirs['checkpoints'], 'final_model.pth')}")

    def _train_epoch(self) -> List[Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            rgb = batch['rgb'].to(self.device)
            alpha_target = batch['alpha'].to(self.device)

            # Forward pass
            alpha_pred = self.model(rgb)

            # Compute loss
            losses = self.loss_fn(alpha_pred, alpha_target)
            loss = losses['total']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_clip_norm = self.config.get('training', {}).get('grad_clip_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)

            self.optimizer.step()

            # Record losses
            batch_losses = {k: v.item() for k, v in losses.items()}
            epoch_losses.append(batch_losses)

            # Log batch progress
            if batch_idx % self.config.get('logging', {}).get('log_interval', 10) == 0:
                self._log_batch(batch_idx, len(self.train_loader), batch_losses)

        return epoch_losses

    def _validate_epoch(self) -> List[Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        epoch_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                rgb = batch['rgb'].to(self.device)
                alpha_target = batch['alpha'].to(self.device)

                # Forward pass
                alpha_pred = self.model(rgb)

                # Compute loss
                losses = self.loss_fn(alpha_pred, alpha_target)

                # Record losses
                batch_losses = {k: v.item() for k, v in losses.items()}
                epoch_losses.append(batch_losses)

        return epoch_losses

    def _log_batch(self, batch_idx: int, num_batches: int, losses: Dict[str, float]):
        """Log batch progress"""
        progress = (batch_idx + 1) / num_batches * 100
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])

        print(f"Epoch {self.current_epoch+1} | Batch {batch_idx+1:3d}/{num_batches:3d} "
              f"[{progress:5.1f}%] | {loss_str}")

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float],
                   train_losses: List[Dict[str, float]], val_losses: Optional[List[Dict[str, float]]]):
        """Log epoch progress"""
        # Console logging
        if val_loss is not None:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f}")

        # TensorBoard logging
        self.writer.add_scalar('Loss/train_total', train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar('Loss/val_total', val_loss, epoch)

        # Log individual loss components
        if train_losses:
            for loss_name in train_losses[0].keys():
                avg_train_loss = np.mean([loss[loss_name] for loss in train_losses])
                self.writer.add_scalar(f'Loss/train_{loss_name}', avg_train_loss, epoch)

        if val_losses:
            for loss_name in val_losses[0].keys():
                avg_val_loss = np.mean([loss[loss_name] for loss in val_losses])
                self.writer.add_scalar(f'Loss/val_{loss_name}', avg_val_loss, epoch)

        # Log learning rate
        if self.optimizer.param_groups:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', lr, epoch)

    def _save_checkpoint(self, filename: str = None):
        """Save model checkpoint"""
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch+1}.pth'

        checkpoint_path = os.path.join(self.output_dirs['checkpoints'], filename)

        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': self.best_loss if hasattr(self, 'best_loss') else None,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('loss', float('inf'))

        print(f"📦 Checkpoint loaded: {checkpoint_path}")
        print(f"   Epoch: {self.current_epoch}")
        print(f"   Best loss: {self.best_loss}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'model_info': self.model.get_model_info(),
            'loss_info': self.loss_fn.get_loss_info(),
            'device': str(self.device),
            'output_dirs': self.output_dirs
        }
