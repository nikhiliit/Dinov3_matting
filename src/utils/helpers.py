"""
Helper utilities for DINOv3 Alpha Matting
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file"""
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"💾 Configuration saved: {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"📋 Configuration loaded: {config_path}")
    return config


def create_output_directories(base_dir: str) -> Dict[str, str]:
    """
    Create output directory structure

    Args:
        base_dir: Base output directory

    Returns:
        Dictionary with paths to created directories
    """
    base_path = Path(base_dir)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_path / f"dinov3_alpha_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    dirs = {
        'root': str(output_dir),
        'checkpoints': str(output_dir / 'checkpoints'),
        'logs': str(output_dir / 'logs'),
        'samples': str(output_dir / 'samples'),
        'configs': str(output_dir / 'configs'),
        'tensorboard': str(output_dir / 'logs' / 'tensorboard')
    }

    for dir_name, dir_path in dirs.items():
        if dir_name != 'root':
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    print(f"📁 Created output directories:")
    for dir_name, dir_path in dirs.items():
        print(f"   {dir_name}: {dir_path}")

    return dirs


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger('dino_alpha_matting')
    logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add console handler
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def print_model_summary(model, input_size=(1, 3, 224, 224)):
    """Print model summary with parameter count"""
    print("\n🤖 Model Summary:")
    print("=" * 50)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(",.0f")
    print(",.0f")
    print(",.0f")
    print(".1f")

    # Model info
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print("\n📊 Model Details:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")

    print("=" * 50)


def save_training_history(history: Dict[str, list], output_path: str):
    """Save training history to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"📈 Training history saved: {output_path}")


def load_training_history(history_path: str) -> Dict[str, list]:
    """Load training history from JSON file"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    print(f"📈 Training history loaded: {history_path}")
    return history


def create_experiment_summary(config: Dict[str, Any],
                            training_stats: Dict[str, Any],
                            output_path: str):
    """Create experiment summary"""
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'framework': 'DINOv3 Alpha Matting',
            'version': '1.0.0'
        },
        'configuration': config,
        'training_results': training_stats,
        'performance_metrics': {
            'model_size_mb': training_stats.get('model_info', {}).get('model_size_mb', 0),
            'best_loss': training_stats.get('best_loss', 'N/A'),
            'total_epochs': training_stats.get('current_epoch', 0) + 1
        }
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"📋 Experiment summary saved: {output_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration file"""
    required_keys = ['model', 'loss', 'training', 'data']

    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)

    if missing_keys:
        print(f"❌ Missing required configuration keys: {missing_keys}")
        return False

    # Validate model configuration
    model_config = config['model']
    if 'dino_model_size' not in model_config:
        print("❌ Missing 'dino_model_size' in model configuration")
        return False

    if model_config['dino_model_size'] not in ['vits16', 'vitb16']:
        print("❌ Invalid 'dino_model_size'. Must be 'vits16' or 'vitb16'")
        return False

    # Validate training configuration
    training_config = config['training']
    if 'target_size' not in training_config:
        print("❌ Missing 'target_size' in training configuration")
        return False

    print("✅ Configuration validation passed")
    return True


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices"""
    import torch

    info = {
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None
    }

    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name()
        info['memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
        info['memory_reserved'] = torch.cuda.memory_reserved() / 1024**3    # GB

    return info


def print_system_info():
    """Print system and hardware information"""
    print("\n🖥️  System Information:")
    print("=" * 40)

    device_info = get_device_info()

    if device_info['cuda_available']:
        print(f"🎮 GPU: {device_info['device_name']}")
        print(".1f")
        print(".1f")
    else:
        print("🎮 GPU: Not available")

    if device_info['mps_available']:
        print("🍎 MPS: Available")
    else:
        print("🍎 MPS: Not available")

    print("=" * 40)
