#!/usr/bin/env python3
"""
DINOv3 Alpha Matting Training Script
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.training.trainer import DINOv3AlphaTrainer
from src.utils.helpers import load_config, validate_config, print_system_info


def main():
    parser = argparse.ArgumentParser(description='Train DINOv3 Alpha Matting Model')

    parser.add_argument('--config', '-c', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory override')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name override')

    args = parser.parse_args()

    # Print system info
    print_system_info()

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
    try:
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
