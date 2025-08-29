#!/usr/bin/env python3
"""
Enhanced DINOv3 Alpha Matting Training Script with Multi-GPU Support
Supports distributed training, gradient checkpointing, and memory optimizations

Usage Examples:

1. Single GPU training:
   python train.py --config configs/dinov3_alpha_config.yaml

2. Multi-GPU training (auto-detect GPUs):
   python train.py --config configs/dinov3_alpha_config.yaml

3. Multi-GPU training (specify number of GPUs):
   python train.py --config configs/dinov3_alpha_config.yaml --world_size 4

4. Resume training from checkpoint:
   python train.py --config configs/dinov3_alpha_config.yaml --resume outputs/checkpoints/best_model.pth

5. Override output directory:
   python train.py --config configs/dinov3_alpha_config.yaml --output_dir ./my_experiment

Key Features:
- MPS/CUDA GPU acceleration with auto-detection
- Distributed training across multiple GPUs
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16)
- Gradient accumulation for larger effective batch sizes
- Persistent DataLoader workers
- Memory monitoring and optimization
- Automatic model checkpointing
"""

import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.training.trainer import DINOv3AlphaTrainer
from src.utils.helpers import load_config, validate_config, print_system_info


def setup_distributed_environment():
    """Setup environment variables for distributed training"""
    # Set default values if not already set
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12345'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'


def train_worker(rank: int, world_size: int, config_path: str, args):
    """Worker function for distributed training"""
    # Set environment variables for this process
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank % torch.cuda.device_count())

    try:
        # Load and validate configuration
        print(f"📋 Loading configuration: {config_path}")
        config = load_config(config_path)

        if not validate_config(config):
            print("❌ Configuration validation failed!")
            sys.exit(1)

        # Override configuration if specified
        if args.output_dir:
            config['output_dir'] = args.output_dir

        if args.experiment_name:
            config['experiment_name'] = args.experiment_name

        # Enable distributed training in config
        if world_size > 1:
            config['distributed'] = config.get('distributed', {})
            config['distributed']['enabled'] = True

        # Create trainer
        print(f"\n🚀 Initializing DINOv3 Alpha Matting Trainer (Rank {rank}/{world_size-1})")
        trainer = DINOv3AlphaTrainer(config)

        # Resume from checkpoint if specified
        if args.resume:
            print(f"\n📦 Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Start training
        print(f"\n🏃 Starting training on rank {rank}...")
        trainer.train()

        if trainer.is_main_process():
            print("\n🎉 Training completed successfully!")

    except Exception as e:
        print(f"\n❌ Training failed on rank {rank} with error: {e}")
        import traceback
        traceback.print_exc()

        # Try to save emergency checkpoint
        try:
            trainer._save_checkpoint(filename=f'emergency_checkpoint_rank_{rank}.pth')
            print(f"🚨 Emergency checkpoint saved for rank {rank}")
        except:
            print(f"❌ Failed to save emergency checkpoint for rank {rank}")

        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train DINOv3 Alpha Matting Model with Multi-GPU Support')

    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory override')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name override')

    # Distributed training arguments
    parser.add_argument('--world_size', type=int, default=1,
                       help='Number of processes for distributed training')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1',
                       help='Master node address for distributed training')
    parser.add_argument('--master_port', type=str, default='12345',
                       help='Master node port for distributed training')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'],
                       help='Backend for distributed training')

    args = parser.parse_args()

    # Setup distributed environment variables
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    # Determine world size
    world_size = args.world_size
    if world_size == 1 and torch.cuda.is_available():
        world_size = torch.cuda.device_count()

    print(f"🔄 Distributed training with world_size={world_size}")

    if world_size > 1:
        # Multi-GPU distributed training
        print("🚀 Starting distributed training...")

        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

        # Spawn worker processes
        mp.spawn(
            train_worker,
            args=(world_size, args.config, args),
            nprocs=world_size,
            join=True
        )

    else:
        # Single GPU training
        print("🚀 Starting single-GPU training...")
        setup_distributed_environment()

        try:
            # Load and validate configuration
            print(f"\n📋 Loading configuration: {args.config}")
            config = load_config(args.config)

            if not validate_config(config):
                print("❌ Configuration validation failed!")
                sys.exit(1)

            # Override configuration if specified
            if args.output_dir:
                config['output_dir'] = args.output_dir

            if args.experiment_name:
                config['experiment_name'] = args.experiment_name

            # Create trainer
            print("\n🚀 Initializing DINOv3 Alpha Matting Trainer")
            trainer = DINOv3AlphaTrainer(config)

            # Resume from checkpoint if specified
            if args.resume:
                print(f"\n📦 Resuming from checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)

            # Start training
            print("\n🏃 Starting training...")
            trainer.train()

            print("\n🎉 Training completed successfully!")

        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            trainer._save_checkpoint(filename='interrupted_checkpoint.pth')
            print("💾 Checkpoint saved as 'interrupted_checkpoint.pth'")

        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()

            # Try to save emergency checkpoint
            try:
                trainer._save_checkpoint(filename='emergency_checkpoint.pth')
                print("🚨 Emergency checkpoint saved")
            except:
                print("❌ Failed to save emergency checkpoint")

            sys.exit(1)


if __name__ == "__main__":
    main()
