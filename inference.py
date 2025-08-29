#!/usr/bin/env python3
"""
DINOv3 Alpha Matting Inference Script
"""

import os
import sys
import argparse
import cv2
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.models.dino_alpha_net import DINOv3AlphaMatting, create_dino_alpha_model
from src.utils.helpers import load_config, get_device_info
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='DINOv3 Alpha Matting Inference')

    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--image_dir', type=str, help='Path to directory of images')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./inference_outputs',
                       help='Output directory')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory in batch mode')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'],
                       help='Device for inference')

    args = parser.parse_args()

    # Validate arguments
    if not args.image and not args.image_dir:
        print("❌ Please provide either --image or --image_dir")
        return

    if args.image and args.batch:
        print("❌ Cannot use --batch with single --image")
        return

    if args.image_dir and not args.batch:
        print("💡 Use --batch flag when processing directories")
        args.batch = True

    # Setup device
    if args.device == 'auto':
        device_info = get_device_info()
        if device_info['cuda_available']:
            device = torch.device('cuda')
        elif device_info['mps_available']:
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"🎯 Using device: {device}")

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Use default configuration
        config = {
            'model': {
                'dino_model_size': 'vits16',
                'dinov3_path': str(current_dir.parent / 'dinov3'),
                'freeze_encoder': True
            }
        }

    # Create model
    print("🤖 Loading DINOv3 Alpha Matting model...")
    model = create_dino_alpha_model(config)
    model.to(device)
    model.eval()

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"📦 Model loaded from: {args.checkpoint}")

    # Setup transforms
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        # Single image processing
        process_single_image(args.image, model, transform, device, args.output_dir)

    elif args.image_dir and args.batch:
        # Batch processing
        process_batch(args.image_dir, model, transform, device, args.output_dir)


def process_single_image(image_path, model, transform, device, output_dir):
    """Process a single image"""
    print(f"🖼️  Processing: {image_path}")

    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"📏 Image size: {image.size}")

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict alpha
    with torch.no_grad():
        alpha_pred = model(input_tensor)

    # Convert to numpy
    alpha_pred = alpha_pred.squeeze(0).squeeze(0).cpu().numpy()

    # Resize back to original size
    alpha_resized = cv2.resize(alpha_pred, image.size, interpolation=cv2.INTER_LINEAR)
    alpha_resized = np.clip(alpha_resized, 0, 1)

    # Save results
    base_name = Path(image_path).stem

    # Save alpha map
    alpha_img = Image.fromarray((alpha_resized * 255).astype(np.uint8), mode='L')
    alpha_path = os.path.join(output_dir, f'{base_name}_alpha.png')
    alpha_img.save(alpha_path)
    print(f"💾 Alpha map saved: {alpha_path}")

    # Save original image
    image.save(os.path.join(output_dir, f'{base_name}_original.jpg'))

    # Create composite with white background
    img_array = np.array(image)
    background = np.full_like(img_array, (255, 255, 255))
    alpha_3ch = np.stack([alpha_resized] * 3, axis=2)
    composite = (img_array.astype(np.float32) * alpha_3ch +
                background.astype(np.float32) * (1 - alpha_3ch))

    comp_img = Image.fromarray(composite.astype(np.uint8))
    comp_path = os.path.join(output_dir, f'{base_name}_composite_white.png')
    comp_img.save(comp_path)
    print(f"💾 White composite saved: {comp_path}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(alpha_resized, cmap='gray')
    axes[1].set_title('Predicted Alpha')
    axes[1].axis('off')

    axes[2].imshow(comp_img)
    axes[2].set_title('White Background Composite')
    axes[2].axis('off')

    plt.tight_layout()
    viz_path = os.path.join(output_dir, f'{base_name}_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"💾 Visualization saved: {viz_path}")
    print("✅ Processing completed!")


def process_batch(image_dir, model, transform, device, output_dir):
    """Process multiple images in batch"""
    image_dir = Path(image_dir)
    image_files = []

    # Find all image files
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(image_dir.glob(f'**/*{ext}'))
        image_files.extend(image_dir.glob(f'**/*{ext.upper()}'))

    image_files = sorted(list(set(image_files)))

    print(f"📂 Found {len(image_files)} images to process")

    successful = 0
    failed = 0

    for i, image_path in enumerate(image_files):
        print(f"\n🔄 Processing {i+1}/{len(image_files)}: {image_path.name}")

        try:
            process_single_image(str(image_path), model, transform, device, output_dir)
            successful += 1

        except Exception as e:
            print(f"❌ Failed to process {image_path}: {e}")
            failed += 1

    print(f"\n📊 Batch processing completed!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
