#!/usr/bin/env python3

"""
Setup script for DINOv3-Based Alpha Matting
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="dinov3_alpha_matting",
    version="1.0.0",
    author="Nikhil IIT",
    author_email="nikhil@example.com",
    description="DINOv3-Based Alpha Matting: Leveraging Self-Supervised Vision Transformers for High-Quality Alpha Matting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikhiliit/Dinov3_matting",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords="computer-vision deep-learning alpha-matting dinov3 vision-transformer",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "tensorboard": [
            "tensorboard>=2.13.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dinov3-alpha-train=dinov3_alpha_matting.training.trainer:main",
            "dinov3-alpha-infer=dinov3_alpha_matting.models.inference:main",
            "dinov3-alpha-demo=dinov3_alpha_matting.demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/nikhiliit/Dinov3_matting/issues",
        "Source": "https://github.com/nikhiliit/Dinov3_matting",
        "Documentation": "https://github.com/nikhiliit/Dinov3_matting#readme",
    },
)
