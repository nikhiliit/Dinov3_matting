"""
DINOv3 Alpha Matting Dataset
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random
import warnings
warnings.filterwarnings('ignore')


class DINOv3AlphaDataset(Dataset):
    """Dataset for DINOv3-based alpha matting training"""

    def __init__(self,
                 rgb_dir: str,
                 alpha_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (224, 224),
                 augment_data: bool = True,
                 cache_images: bool = False):
        """
        Initialize DINOv3 Alpha Matting Dataset

        Args:
            rgb_dir: Directory containing RGB images
            alpha_dir: Directory containing alpha maps
            transform: Image transformations
            target_size: Target image size (must be compatible with DINOv3 patches)
            augment_data: Whether to apply data augmentation
            cache_images: Whether to cache images in memory
        """
        self.rgb_dir = Path(rgb_dir)
        self.alpha_dir = Path(alpha_dir)
        self.transform = transform
        self.target_size = target_size
        self.augment_data = augment_data
        self.cache_images = cache_images

        # Find valid image pairs
        self.image_pairs = self._find_valid_pairs()

        # Setup augmentations
        if augment_data:
            self.augmentation = self._create_augmentation_pipeline()
        else:
            self.augmentation = None

        # Cache
        self.image_cache = {} if cache_images else None

        print(f"Found {len(self.image_pairs)} valid image pairs for DINOv3 alpha matting")

    def _find_valid_pairs(self) -> List[str]:
        """Find RGB-alpha image pairs"""
        rgb_files = set()

        # Get RGB files
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            rgb_files.update([f.stem for f in self.rgb_dir.glob(f'*{ext}')])
            rgb_files.update([f.stem for f in self.rgb_dir.glob(f'*{ext.upper()}')])

        valid_pairs = []

        for rgb_stem in rgb_files:
            # Check for corresponding alpha map
            alpha_file = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_alpha = self.alpha_dir / f"{rgb_stem}{ext}"
                if potential_alpha.exists():
                    alpha_file = potential_alpha
                    break

            if alpha_file is not None:
                valid_pairs.append(rgb_stem)

        return sorted(valid_pairs)

    def _create_augmentation_pipeline(self):
        """Create augmentation pipeline for DINOv3 training"""
        # For DINOv3, we need to maintain patch alignment
        # So we use minimal augmentations that preserve patch structure
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.Resize(self.target_size),
        ])

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset"""
        image_stem = self.image_pairs[idx]

        # Load from cache or disk
        if self.image_cache is not None and image_stem in self.image_cache:
            sample = self.image_cache[image_stem]
        else:
            sample = self._load_sample(image_stem)
            if self.image_cache is not None:
                self.image_cache[image_stem] = sample

        # Apply augmentations
        if self.augmentation is not None:
            sample = self._apply_augmentations(sample)

        # Apply transforms
        sample = self._apply_transforms(sample)

        # Add metadata
        sample['filename'] = image_stem
        sample['idx'] = idx

        return sample

    def _load_sample(self, image_stem: str) -> Dict[str, np.ndarray]:
        """Load RGB image and alpha map"""
        sample = {}

        # Load RGB image
        rgb_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            potential_file = self.rgb_dir / f"{image_stem}{ext}"
            if potential_file.exists():
                rgb_file = potential_file
                break

        if rgb_file is None:
            raise FileNotFoundError(f"RGB image not found for {image_stem}")

        rgb_image = cv2.imread(str(rgb_file))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        sample['rgb'] = rgb_image

        # Load alpha map
        alpha_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_file = self.alpha_dir / f"{image_stem}{ext}"
            if potential_file.exists():
                alpha_file = potential_file
                break

        if alpha_file is None:
            raise FileNotFoundError(f"Alpha map not found for {image_stem}")

        alpha_image = cv2.imread(str(alpha_file), cv2.IMREAD_GRAYSCALE)
        sample['alpha'] = alpha_image

        return sample

    def _apply_augmentations(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply augmentations to the sample"""
        rgb_pil = Image.fromarray(sample['rgb'])
        alpha_pil = Image.fromarray(sample['alpha'])

        # Apply same augmentation to RGB and alpha
        if random.random() < 0.5:  # Horizontal flip
            rgb_pil = rgb_pil.transpose(Image.FLIP_LEFT_RIGHT)
            alpha_pil = alpha_pil.transpose(Image.FLIP_LEFT_RIGHT)

        # Apply color jitter to RGB only
        if random.random() < 0.3:
            rgb_pil = transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            )(rgb_pil)

        sample['rgb'] = np.array(rgb_pil)
        sample['alpha'] = np.array(alpha_pil)

        return sample

    def _apply_transforms(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert to tensors and apply final transforms"""
        # Resize to target size (maintaining aspect ratio for DINOv3)
        rgb_pil = Image.fromarray(sample['rgb']).resize(self.target_size, Image.BILINEAR)
        alpha_pil = Image.fromarray(sample['alpha']).resize(self.target_size, Image.BILINEAR)

        # Convert to tensors
        if self.transform is not None:
            sample['rgb'] = self.transform(rgb_pil)
        else:
            sample['rgb'] = torch.from_numpy(np.array(rgb_pil)).permute(2, 0, 1).float() / 255.0

        sample['alpha'] = torch.from_numpy(np.array(alpha_pil)).unsqueeze(0).float() / 255.0

        return sample


def create_data_transforms(target_size: Tuple[int, int] = (224, 224),
                         normalize: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create standard transforms for RGB and alpha images"""

    rgb_transforms = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]

    if normalize:
        # ImageNet normalization for DINOv3
        rgb_transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    alpha_transforms = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ]

    return transforms.Compose(rgb_transforms), transforms.Compose(alpha_transforms)


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""

    data_config = config.get('data', {})
    training_config = config.get('training', {})

    # Dataset paths
    train_rgb_dir = data_config.get('train_rgb', './data/train/rgb_images')
    train_alpha_dir = data_config.get('train_alpha', './data/train/alpha_maps')
    val_rgb_dir = data_config.get('val_rgb', './data/val/rgb_images')
    val_alpha_dir = data_config.get('val_alpha', './data/val/alpha_maps')

    target_size = tuple(training_config.get('target_size', [224, 224]))
    batch_size = training_config.get('batch_size', 8)
    num_workers = training_config.get('num_workers', 4)

    # Transforms
    rgb_transform, alpha_transform = create_data_transforms(
        target_size=target_size,
        normalize=config.get('normalize_rgb', True)
    )

    # Create datasets
    train_dataset = DINOv3AlphaDataset(
        rgb_dir=train_rgb_dir,
        alpha_dir=train_alpha_dir,
        transform=rgb_transform,
        target_size=target_size,
        augment_data=config.get('augment_data', True),
        cache_images=config.get('cache_images', False)
    )

    val_dataset = DINOv3AlphaDataset(
        rgb_dir=val_rgb_dir,
        alpha_dir=val_alpha_dir,
        transform=rgb_transform,
        target_size=target_size,
        augment_data=False,  # No augmentation for validation
        cache_images=config.get('cache_images', False)
    )

    # Get memory optimization settings
    memory_config = config.get('memory_optimization', {})
    prefetch_factor = memory_config.get('prefetch_factor', 2)
    persistent_workers = memory_config.get('persistent_workers', False)

    # Create data loaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader


def create_test_data_loader(config: Dict[str, Any], test_rgb_dir: str, test_alpha_dir: str) -> DataLoader:
    """Create test data loader"""
    training_config = config.get('training', {})
    target_size = tuple(training_config.get('target_size', [224, 224]))
    batch_size = training_config.get('batch_size', 8)
    num_workers = training_config.get('num_workers', 4)

    # Transforms
    rgb_transform, _ = create_data_transforms(
        target_size=target_size,
        normalize=config.get('normalize_rgb', True)
    )

    # Create test dataset
    test_dataset = DINOv3AlphaDataset(
        rgb_dir=test_rgb_dir,
        alpha_dir=test_alpha_dir,
        transform=rgb_transform,
        target_size=target_size,
        augment_data=False,
        cache_images=config.get('cache_images', False)
    )

    # Get memory optimization settings
    memory_config = config.get('memory_optimization', {})
    prefetch_factor = memory_config.get('prefetch_factor', 2)
    persistent_workers = memory_config.get('persistent_workers', False)

    # Create test data loader with optimizations
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )

    return test_loader
