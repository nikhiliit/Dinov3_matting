#!/usr/bin/env python3

"""
Setup script for DINOv3-Based Alpha Matting

This script helps set up the complete environment for DINOv3 alpha matting,
including downloading required models and setting up the workspace.
"""

import os
import sys
import argparse
import subprocess
import urllib.request
from pathlib import Path


class DINOv3Setup:
    """Setup class for DINOv3 Alpha Matting"""

    def __init__(self, install_dir=None):
        self.install_dir = Path(install_dir) if install_dir else Path.cwd()
        self.models_dir = self.install_dir / "models"
        self.dinov3_dir = self.install_dir / "dinov3"

        # DINOv3 model URLs (Meta Research)
        self.model_urls = {
            "dinov3_vits16_pretrain_lvd1689m.pth": "https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m.pth",
            "dinov3_vitb16_pretrain_lvd1689m.pth": "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m.pth",
            "dinov3_vitl16_pretrain_lvd1689m.pth": "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m.pth",
        }

        # DINOv3 repository
        self.dinov3_repo_url = "https://github.com/facebookresearch/dinov3.git"

    def run(self):
        """Run the complete setup process"""
        print("🚀 DINOv3 Alpha Matting Setup")
        print("=" * 50)

        try:
            self.check_system()
            self.create_directories()
            self.install_dependencies()
            self.download_dinov3_repo()
            self.download_models()
            self.verify_setup()
            self.create_sample_config()

            print("\n✅ Setup completed successfully!")
            print("\n🎯 Next steps:")
            print("1. Prepare your dataset (see README.md for format)")
            print("2. Run: python train.py --config configs/dinov3_alpha_config.yaml")
            print("3. Or try: python demo.py")

        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            sys.exit(1)

    def check_system(self):
        """Check system requirements"""
        print("🖥️  Checking system requirements...")

        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8+ required")

        print(f"✅ Python {sys.version.split()[0]}")

        # Check if torch is available
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__}")
        except ImportError:
            print("⚠️  PyTorch not found - will be installed")

        # Check CUDA availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            else:
                print("⚠️  CUDA not available - using CPU")
        except:
            pass

    def create_directories(self):
        """Create necessary directories"""
        print("📁 Creating directories...")

        directories = [
            self.models_dir,
            self.dinov3_dir,
            self.install_dir / "outputs",
            self.install_dir / "datasets",
            self.install_dir / "logs"
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {dir_path}")

    def install_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing dependencies...")

        # Install PyTorch based on CUDA availability
        try:
            import torch
            print("✅ PyTorch already installed")
        except ImportError:
            print("Installing PyTorch...")
            # Try CUDA version first, fallback to CPU
            torch_commands = [
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
            ]

            for cmd in torch_commands:
                try:
                    subprocess.run(cmd.split(), check=True, capture_output=True)
                    print("✅ PyTorch installed")
                    break
                except subprocess.CalledProcessError:
                    continue
            else:
                print("⚠️  PyTorch installation failed - please install manually")

        # Install other dependencies
        requirements_file = self.install_dir / "requirements.txt"
        if requirements_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
            print("✅ Dependencies installed")

    def download_dinov3_repo(self):
        """Download DINOv3 repository"""
        print("📥 Downloading DINOv3 repository...")

        if (self.dinov3_dir / "dinov3").exists():
            print("✅ DINOv3 repository already exists")
            return

        try:
            subprocess.run([
                "git", "clone", "--depth", "1",
                self.dinov3_repo_url, str(self.dinov3_dir)
            ], check=True)
            print("✅ DINOv3 repository downloaded")
        except subprocess.CalledProcessError:
            print("⚠️  Git clone failed - you may need to download DINOv3 manually")

    def download_models(self):
        """Download DINOv3 models"""
        print("🤖 Downloading DINOv3 models...")
        print("Note: Models are ~3GB total. This may take a while...")

        downloaded_count = 0
        total_models = len(self.model_urls)

        for model_name, url in self.model_urls.items():
            model_path = self.models_dir / model_name

            if model_path.exists():
                print(f"✅ {model_name} already exists")
                downloaded_count += 1
                continue

            try:
                print(f"📥 Downloading {model_name}...")
                urllib.request.urlretrieve(url, model_path)
                downloaded_count += 1
                print(f"✅ Downloaded {model_name}")
            except Exception as e:
                print(f"⚠️  Failed to download {model_name}: {e}")

        if downloaded_count == total_models:
            print(f"✅ All {total_models} models downloaded successfully")
        elif downloaded_count > 0:
            print(f"✅ {downloaded_count}/{total_models} models downloaded")
        else:
            print("⚠️  No models downloaded - you may need to download them manually")

    def verify_setup(self):
        """Verify that setup is complete"""
        print("🔍 Verifying setup...")

        checks = [
            ("Source code", self.install_dir / "src" / "models" / "dino_encoder.py"),
            ("Configuration", self.install_dir / "configs" / "dinov3_alpha_config.yaml"),
            ("Training script", self.install_dir / "train.py"),
            ("DINOv3 repository", self.dinov3_dir / "dinov3"),
            ("Models directory", self.models_dir),
        ]

        all_passed = True
        for check_name, check_path in checks:
            if check_path.exists():
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}: {check_path}")
                all_passed = False

        # Check at least one model
        model_files = list(self.models_dir.glob("*.pth"))
        if model_files:
            print(f"✅ Models: {len(model_files)} model(s) found")
        else:
            print("⚠️  No model files found")
            all_passed = False

        return all_passed

    def create_sample_config(self):
        """Create sample configuration file"""
        print("⚙️  Creating sample configuration...")

        sample_config = f"""# DINOv3-Based Alpha Matting Configuration
# Generated by setup script

# Model Architecture
model:
  dino_model_size: "vits16"  # Options: vits16, vitb16, vitl16
  dinov3_path: "{self.dinov3_dir}"  # Path to DINOv3 installation
  decoder_dims: [256, 128, 64]  # Decoder layer dimensions
  freeze_encoder: true  # Keep DINOv3 encoder frozen

# Loss Function Configuration
loss:
  alpha_weight: 1.0
  gradient_weight: 0.5
  laplacian_weight: 0.1
  use_charbonnier: true

# Training Parameters
training:
  num_epochs: 50
  batch_size: 8
  target_size: [224, 224]
  num_workers: 4

  optimizer:
    name: "adamw"
    lr: 0.0001
    weight_decay: 0.01

  scheduler:
    name: "cosineannealinglr"
    eta_min: 0.0000001

# Data Configuration
data:
  # Update these paths to your dataset
  train_rgb: "./datasets/train/rgb_images"
  train_alpha: "./datasets/train/alpha_maps"
  val_rgb: "./datasets/val/rgb_images"
  val_alpha: "./datasets/val/alpha_maps"

# Output Configuration
output_dir: "./outputs"
experiment_name: "dinov3_alpha_matting_{{model_size}}"
"""

        config_path = self.install_dir / "configs" / "sample_config.yaml"
        with open(config_path, "w") as f:
            f.write(sample_config)

        print(f"✅ Sample config created: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Setup DINOv3 Alpha Matting")
    parser.add_argument("--dir", help="Installation directory (default: current directory)")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")

    args = parser.parse_args()

    setup = DINOv3Setup(args.dir)

    if args.skip_models:
        setup.model_urls = {}  # Skip model downloads

    if args.skip_deps:
        # Skip dependency installation
        original_install_deps = setup.install_dependencies
        setup.install_dependencies = lambda: print("⏭️  Skipping dependency installation")

    setup.run()


if __name__ == "__main__":
    main()
