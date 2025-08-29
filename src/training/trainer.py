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
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path
import time
import yaml
from datetime import datetime
import warnings
import gc
import psutil
import GPUtil
warnings.filterwarnings('ignore')

# Import DINOv3 distributed training utilities
try:
    import sys
    dinov3_path = Path(__file__).parent.parent / 'dinov3'
    if str(dinov3_path) not in sys.path:
        sys.path.insert(0, str(dinov3_path))
    from dinov3.dinov3.distributed.torch_distributed_wrapper import (
        enable_distributed, get_rank, get_world_size, is_main_process
    )
    DINOv3_DISTRIBUTED_AVAILABLE = True
except ImportError:
    DINOv3_DISTRIBUTED_AVAILABLE = False
    print("⚠️  DINOv3 distributed training not available, falling back to single GPU")

from ..models.dino_alpha_net import DINOv3AlphaMatting, create_dino_alpha_model
from ..losses.alpha_losses import DINOv3MattingLoss, create_dino_alpha_loss
from ..data.dataset import create_data_loaders
from ..utils.helpers import save_config, create_output_directories


class MemoryMonitor:
    """Memory monitoring utilities for GPU and system memory"""

    @staticmethod
    def get_gpu_memory_info():
        """Get GPU memory information"""
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None

            gpu = gpus[0]  # Primary GPU
            return {
                'gpu_id': gpu.id,
                'gpu_name': gpu.name,
                'memory_used_mb': gpu.memoryUsed,
                'memory_total_mb': gpu.memoryTotal,
                'memory_free_mb': gpu.memoryFree,
                'memory_utilization_percent': gpu.memoryUtil * 100,
                'gpu_utilization_percent': gpu.load * 100
            }
        except:
            return None

    @staticmethod
    def get_system_memory_info():
        """Get system memory information"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percentage': memory.percent
        }

    @staticmethod
    def log_memory_stats(logger, step_name: str = ""):
        """Log current memory statistics"""
        gpu_info = MemoryMonitor.get_gpu_memory_info()
        system_info = MemoryMonitor.get_system_memory_info()

        if gpu_info:
            logger(f"{step_name} GPU Memory: {gpu_info['memory_used_mb']:.0f}MB/"
                   f"{gpu_info['memory_total_mb']:.0f}MB "
                   f"({gpu_info['memory_utilization_percent']:.1f}%) "
                   f"GPU Util: {gpu_info['gpu_utilization_percent']:.1f}%")

        logger(f"{step_name} System Memory: {system_info['used_gb']:.1f}GB/"
               f"{system_info['total_gb']:.1f}GB "
               f"({system_info['percentage']:.1f}%)")


class DINOv3AlphaTrainer:
    """Enhanced trainer class for DINOv3-based alpha matting with multi-GPU support"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer

        Args:
            config: Training configuration
        """
        self.config = config

        # Initialize distributed training
        self.distributed = self._setup_distributed()

        # Setup device (must be after distributed init for proper GPU assignment)
        self.device = self._setup_device()

        # Create output directories
        self.output_dirs = create_output_directories(config.get('output_dir', './outputs'))

        # Initialize model, loss, optimizer
        self.model = self._setup_model()
        self.loss_fn = self._setup_loss()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Setup memory optimizations
        self.scaler = self._setup_mixed_precision()
        self.memory_monitor = MemoryMonitor()

        # Setup data loaders with optimizations
        self.train_loader, self.val_loader = create_data_loaders(config)

        # Setup logging
        self.writer = self._setup_tensorboard()

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_stats = []
        self.accumulation_steps = config.get('training', {}).get('accumulation_steps', 1)

        # Only print from main process in distributed training
        if not self.distributed or self.is_main_process():
            print("✅ DINOv3 Alpha Matting Trainer initialized")
            print(f"📁 Outputs: {self.output_dirs['root']}")
            print(f"🎯 Device: {self.device}")
            if self.distributed:
                print(f"🔄 Distributed: {self.world_size} GPUs")
            print(f"📊 Train samples: {len(self.train_loader.dataset)}")
            print(f"📊 Val samples: {len(self.val_loader.dataset)}")

            # Log initial memory stats
            if config.get('monitoring', {}).get('enable_memory_tracking', False):
                MemoryMonitor.log_memory_stats(print, "Initial")

    def _setup_distributed(self) -> bool:
        """Setup distributed training"""
        distributed_config = self.config.get('distributed', {})
        if not distributed_config.get('enabled', False):
            return False

        if not DINOv3_DISTRIBUTED_AVAILABLE:
            print("⚠️  Distributed training requested but DINOv3 distributed utils not available")
            return False

        try:
            # Initialize distributed training
            enable_distributed(
                set_cuda_current_device=True,
                nccl_async_error_handling=True,
                restrict_print_to_main_process=True
            )

            self.rank = get_rank()
            self.world_size = get_world_size()
            self.is_main_process_flag = self.rank == 0

            return True

        except Exception as e:
            print(f"⚠️  Failed to initialize distributed training: {e}")
            return False

    def is_main_process(self) -> bool:
        """Check if this is the main process in distributed training"""
        if not self.distributed:
            return True
        return self.is_main_process_flag

    def _setup_device(self) -> torch.device:
        """Setup training device with MPS/CUDA support"""
        device_config = self.config.get('device', 'auto')

        if device_config == 'auto':
            # Auto-detect best available device
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_config)

        # Handle distributed training device assignment
        if self.distributed and device.type == 'cuda':
            # In distributed training, each process gets its own GPU
            device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')

        # Optimize device settings
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.set_device(device)
            # Enable cuDNN benchmarking for faster training
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        elif device.type == 'mps':
            # MPS optimizations for Apple Silicon
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory limits

        return device

    def _setup_mixed_precision(self):
        """Setup mixed precision training"""
        training_config = self.config.get('training', {})
        if training_config.get('mixed_precision', False):
            return GradScaler()
        return None

    def _setup_model(self) -> DINOv3AlphaMatting:
        """Setup model with optimizations"""
        model = create_dino_alpha_model(self.config)

        # Enable gradient checkpointing for memory optimization
        training_config = self.config.get('training', {})
        if training_config.get('gradient_checkpointing', False):
            # Apply gradient checkpointing to the encoder (DINOv3)
            if hasattr(model.encoder, 'dinov3'):
                try:
                    model.encoder.dinov3.gradient_checkpointing_enable()
                    if self.is_main_process():
                        print("🔄 Gradient checkpointing enabled for DINOv3 encoder")
                except AttributeError:
                    # Some DINOv3 versions may not have gradient checkpointing
                    if self.is_main_process():
                        print("⚠️  Gradient checkpointing not available for this DINOv3 version, skipping")

        # Wrap model for distributed training
        if self.distributed:
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.device] if self.device.type == 'cuda' else None,
                output_device=self.device if self.device.type == 'cuda' else None,
                find_unused_parameters=self.config.get('find_unused_parameters', False)
            )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if self.is_main_process():
            print("🤖 Model created:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Encoder frozen: {model.module.encoder.freeze_encoder if self.distributed else model.encoder.freeze_encoder}")
            if self.distributed:
                print(f"   Distributed: {self.world_size} GPUs")

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
        """Train for one epoch with optimizations"""
        self.model.train()
        epoch_losses = []
        accumulation_counter = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            rgb = batch['rgb'].to(self.device)
            alpha_target = batch['alpha'].to(self.device)

            # Mixed precision training
            if self.scaler is not None:
                with autocast():
                    # Forward pass
                    alpha_pred = self.model(rgb)
                    # Compute loss
                    losses = self.loss_fn(alpha_pred, alpha_target)
                    loss = losses['total'] / self.accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                # Forward pass
                alpha_pred = self.model(rgb)
                # Compute loss
                losses = self.loss_fn(alpha_pred, alpha_target)
                loss = losses['total'] / self.accumulation_steps

                # Backward pass
                loss.backward()

            # Gradient accumulation
            accumulation_counter += 1
            if accumulation_counter % self.accumulation_steps == 0:
                # Gradient clipping
                grad_clip_norm = self.config.get('training', {}).get('grad_clip_norm', 1.0)
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Record losses
            batch_losses = {k: v.item() for k, v in losses.items()}
            epoch_losses.append(batch_losses)

            # Memory optimization: periodic cache clearing
            mem_config = self.config.get('memory_optimization', {})
            empty_cache_steps = mem_config.get('empty_cache_every_n_steps', 50)
            if batch_idx % empty_cache_steps == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Log batch progress
            log_interval = self.config.get('training', {}).get('log_interval', 20)
            if batch_idx % log_interval == 0 and self.is_main_process():
                self._log_batch(batch_idx, len(self.train_loader), batch_losses)

                # Memory monitoring
                if self.config.get('monitoring', {}).get('enable_memory_tracking', False):
                    MemoryMonitor.log_memory_stats(print, f"Batch {batch_idx}")

        return epoch_losses

    def _validate_epoch(self) -> List[Dict[str, float]]:
        """Validate for one epoch with optimizations"""
        self.model.eval()
        epoch_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                rgb = batch['rgb'].to(self.device)
                alpha_target = batch['alpha'].to(self.device)

                # Mixed precision validation
                if self.scaler is not None:
                    with autocast():
                        # Forward pass
                        alpha_pred = self.model(rgb)
                        # Compute loss
                        losses = self.loss_fn(alpha_pred, alpha_target)
                else:
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
        if not self.is_main_process():
            return

        progress = (batch_idx + 1) / num_batches * 100
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])

        print(f"Epoch {self.current_epoch+1} | Batch {batch_idx+1:3d}/{num_batches:3d} "
              f"[{progress:5.1f}%] | {loss_str}")

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float],
                   train_losses: List[Dict[str, float]], val_losses: Optional[List[Dict[str, float]]]):
        """Log epoch progress"""
        if not self.is_main_process():
            return

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

        # Memory monitoring
        if self.config.get('monitoring', {}).get('enable_memory_tracking', False):
            MemoryMonitor.log_memory_stats(print, f"Epoch {epoch+1}")

    def _save_checkpoint(self, filename: str = None):
        """Save model checkpoint"""
        if not self.is_main_process():
            return

        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch+1}.pth'

        checkpoint_path = os.path.join(self.output_dirs['checkpoints'], filename)

        # Get model state dict (handle DistributedDataParallel)
        model_state_dict = (self.model.module.state_dict() if self.distributed
                           else self.model.state_dict())

        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
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

        # Load model state dict (handle DistributedDataParallel)
        if self.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('loss', float('inf'))

        if self.is_main_process():
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
