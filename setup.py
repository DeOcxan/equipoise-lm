"""
Setup script for EBT-Mastery

This allows installing the project as a package for easier imports.

Usage:
    pip install -e .  # Development mode
    pip install .     # Regular install
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="ebt-mastery",
    version="1.0.0",
    author="EBT-Mastery Project",
    author_email="ArtyomDeAI@proton.me",
    description="Energy-Based Transformers from Scratch: Research-Grade Educational Implementation with MCMC Corruption, vocab_to_embed MLP, and RMSNorm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeOcxan/EBT-Mastery",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "ipywidgets>=8.0.0",
        ],
    },
)

