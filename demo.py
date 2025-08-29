#!/usr/bin/env python3
"""
DINOv3 Alpha Matting Demo Script
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.models.dino_encoder import DINOv3Encoder
from src.models.decoder import MultiScaleDecoder
from src.models.dino_alpha_net import DINOv3AlphaMatting


def create_demo_model():
    """Create a demo model for testing"""
    print("🎭 Creating DINOv3 Alpha Matting Demo Model")

    # Create encoder
    encoder = DINOv3Encoder(
        model_size='vits16',
        dinov3_path=str(current_dir.parent / 'dinov3'),
        freeze_encoder=True
    )

    # Create decoder
    decoder = MultiScaleDecoder(
        embed_dim=encoder.embed_dim,
        decoder_dims=[256, 128, 64]
    )

    # Create complete model
    model = DINOv3AlphaMatting(
        model_size='vits16',
        dinov3_path=str(current_dir.parent / 'dinov3'),
        freeze_encoder=True
    )

    print("✅ Demo model created successfully!")
    return model


def create_synthetic_alpha(image_size=(224, 224)):
    """Create synthetic alpha for demonstration"""
    # Create a simple circular alpha mask
    center = (image_size[0] // 2, image_size[1] // 2)
    radius = min(image_size) // 3

    y, x = np.ogrid[:image_size[0], :image_size[1]]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    alpha = np.zeros(image_size)
    alpha[dist_from_center <= radius] = 1.0

    # Add some noise for realism
    noise = np.random.normal(0, 0.05, image_size)
    alpha = np.clip(alpha + noise, 0, 1)

    return alpha


def demo_forward_pass():
    """Demonstrate forward pass without training"""
    print("\n🔄 Running Forward Pass Demo")

    # Create model
    model = create_demo_model()
    model.eval()

    # Create synthetic input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 224, 224)

    print(f"📊 Input shape: {input_tensor.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    print(f"📊 Output shape: {output.shape}")
    print(f"📊 Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    return model


def demo_image_processing():
    """Demonstrate image processing pipeline"""
    print("\n🖼️  Running Image Processing Demo")

    # Check if test images exist
    test_images_dir = current_dir / 'test_images'
    if not test_images_dir.exists():
        print("⚠️  No test images found. Creating synthetic demo...")
        return demo_synthetic_processing()

    # Find test images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(test_images_dir.glob(f'*{ext}'))

    if not image_files:
        print("⚠️  No test images found. Creating synthetic demo...")
        return demo_synthetic_processing()

    # Process first image
    image_path = image_files[0]
    print(f"📸 Processing: {image_path.name}")

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    print(f"📏 Original size: {image.size}")

    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)
    print(f"📊 Processed tensor shape: {input_tensor.shape}")

    # Create and run model
    model = create_demo_model()
    model.eval()

    with torch.no_grad():
        alpha_pred = model(input_tensor)

    # Convert to numpy
    alpha_pred = alpha_pred.squeeze(0).squeeze(0).numpy()
    alpha_pred = np.clip(alpha_pred, 0, 1)

    print(f"🎭 Predicted alpha shape: {alpha_pred.shape}")
    print(f"🎭 Alpha range: [{alpha_pred.min():.3f}, {alpha_pred.max():.3f}]")

    return image, alpha_pred


def demo_synthetic_processing():
    """Demonstrate with synthetic data"""
    print("\n🎨 Creating Synthetic Processing Demo")

    # Create synthetic RGB image
    rgb_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(rgb_image)

    # Create synthetic alpha
    alpha_gt = create_synthetic_alpha((224, 224))

    print(f"📸 Synthetic image shape: {rgb_image.shape}")
    print(f"🎭 Synthetic alpha shape: {alpha_gt.shape}")

    # Setup transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)
    print(f"📊 Processed tensor shape: {input_tensor.shape}")

    # Create and run model
    model = create_demo_model()
    model.eval()

    with torch.no_grad():
        alpha_pred = model(input_tensor)

    # Convert to numpy
    alpha_pred = alpha_pred.squeeze(0).squeeze(0).numpy()
    alpha_pred = np.clip(alpha_pred, 0, 1)

    print(f"🎭 Predicted alpha shape: {alpha_pred.shape}")
    print(f"🎭 Alpha range: [{alpha_pred.min():.3f}, {alpha_pred.max():.3f}]")

    return image, alpha_pred


def create_visualization(image, alpha_pred, save_path=None):
    """Create visualization of results"""
    print("\n📊 Creating Visualization")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontweight='bold')
    axes[0].axis('off')

    # Predicted alpha
    axes[1].imshow(alpha_pred, cmap='gray')
    axes[1].set_title('Predicted Alpha', fontweight='bold')
    axes[1].axis('off')

    # Alpha statistics
    axes[2].text(0.1, 0.8, f'Alpha Statistics:\n\n'
                          f'Mean: {alpha_pred.mean():.3f}\n'
                          f'Std: {alpha_pred.std():.3f}\n'
                          f'Min: {alpha_pred.min():.3f}\n'
                          f'Max: {alpha_pred.max():.3f}\n'
                          f'Foreground: {(alpha_pred > 0.5).mean():.1%}',
               transform=axes[2].transAxes, fontsize=12,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved: {save_path}")

    return fig


def main():
    """Main demo function"""
    print("🎬 DINOv3 Alpha Matting Demo")
    print("=" * 50)

    try:
        # Demo 1: Forward pass
        model = demo_forward_pass()

        # Demo 2: Image processing
        try:
            image, alpha_pred = demo_image_processing()
        except:
            print("⚠️  Image processing failed, using synthetic data...")
            image, alpha_pred = demo_synthetic_processing()

        # Demo 3: Visualization
        output_dir = current_dir / 'demo_outputs'
        output_dir.mkdir(exist_ok=True)

        viz_path = output_dir / 'demo_visualization.png'
        fig = create_visualization(image, alpha_pred, viz_path)

        # Show model info
        print("\n🤖 Model Information:")
        model_info = model.get_model_info()
        for key, value in model_info.items():
            print(f"   {key}: {value}")

        print("\n✅ Demo completed successfully!")
        print(f"📁 Check demo_outputs/ for visualizations")
        print(f"🎯 Next steps:")
        print(f"   1. Prepare your dataset (RGB + Alpha pairs)")
        print(f"   2. Run: python train.py --config configs/dinov3_alpha_config.yaml")
        print(f"   3. Test: python inference.py --image your_image.jpg --checkpoint best_model.pth")

        plt.show()

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
