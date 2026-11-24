"""MNIST denoising: images, splits, noise and PSNR (method.md section 17).

The images are read from the original gzip-compressed IDX files, which are not part of the
repository; `load_images` checks their SHA-256 before use.
"""

import gzip
import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from ebt.training.batch import Batch

TRAIN_IMAGES = "train-images-idx3-ubyte.gz"
TEST_IMAGES = "t10k-images-idx3-ubyte.gz"
SHA256 = {
    TRAIN_IMAGES: "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    TEST_IMAGES: "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
}
IDX_TYPES = {0x08: torch.uint8}
"""IDX element types used by MNIST: 0x08 is unsigned byte."""
DIFFUSION_STEPS = 100
"""The authors' `diffusion_steps`: noise level f uses a schedule of DIFFUSION_STEPS / f
timesteps and noises at timestep DIFFUSION_STEPS − 1 (method.md section 4.2)."""
BETA_START, BETA_END = 1e-4, 2e-2
"""DiT linear schedule for 1000 timesteps; for T timesteps both are scaled by 1000 / T."""


def read_idx(path: Path) -> Tensor:
    """Read a gzip-compressed IDX file of unsigned bytes into a tensor of its stated shape.

    The header is two zero bytes, the element type, the number of dimensions and one
    big-endian 32-bit size per dimension.
    """
    data = gzip.decompress(path.read_bytes())
    zero, kind, dims = struct.unpack(">HBB", data[:4])
    if zero != 0 or kind not in IDX_TYPES:
        raise ValueError(f"{path}: not an IDX file of unsigned bytes")
    shape = struct.unpack(f">{dims}I", data[4 : 4 + 4 * dims])
    body = data[4 + 4 * dims :]
    if len(body) != math.prod(shape):
        raise ValueError(f"{path}: {len(body)} values, header states {math.prod(shape)}")
    return torch.frombuffer(bytearray(body), dtype=IDX_TYPES[kind]).reshape(shape)


def load_images(directory: Path) -> tuple[Tensor, Tensor]:
    """Training and test images as uint8 tensors (N, 28, 28), after checking their SHA-256."""
    images = []
    for name in (TRAIN_IMAGES, TEST_IMAGES):
        path = directory / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != SHA256[name]:
            raise ValueError(f"{path}: SHA-256 {digest}, expected {SHA256[name]}")
        images.append(read_idx(path))
    return images[0], images[1]


def to_signed(images: Tensor) -> Tensor:
    """uint8 (N, H, W) -> float32 (N, 1, H, W) in [−1, 1]: (p / 255 − 0.5) / 0.5."""
    return (images.float().unsqueeze(1) / 255 - 0.5) / 0.5


def to_pixels(images: Tensor) -> Tensor:
    """Inverse scaling for PSNR: x · 0.5 + 0.5, clamped to [0, 1], times 255."""
    return (images * 0.5 + 0.5).clamp(0, 1) * 255


@dataclass
class ImageData:
    """Clean images in [−1, 1], shape (N, 1, 28, 28); the context of every batch is None."""

    train: Batch
    validation: Batch
    test: Batch


def make_data(
    train_images: Tensor, test_images: Tensor, validation_size: int, seed: int
) -> ImageData:
    """Split the official training images by one fixed permutation into training and validation
    images; the official test images are the test split."""
    order = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(seed))
    train = to_signed(train_images[order[validation_size:]])
    validation = to_signed(train_images[order[:validation_size]])
    return ImageData(
        Batch(None, train), Batch(None, validation), Batch(None, to_signed(test_images))
    )


def alpha_bar(level: float) -> float:
    """ᾱ of noise level f: the cumulative product of 1 − β over the first DIFFUSION_STEPS
    steps of the linear schedule of int(DIFFUSION_STEPS / f) timesteps (float64)."""
    timesteps = int(DIFFUSION_STEPS / level)
    scale = 1000 / timesteps
    betas = torch.linspace(scale * BETA_START, scale * BETA_END, timesteps, dtype=torch.float64)
    return torch.cumprod(1 - betas, dim=0)[DIFFUSION_STEPS - 1].item()


def add_noise(images: Tensor, alpha: float, generator: torch.Generator) -> Tensor:
    """√ᾱ · x + √(1 − ᾱ) · ε with ε ~ N(0, I) (method.md section 4.2)."""
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype, device=images.device)
    return math.sqrt(alpha) * images + math.sqrt(1 - alpha) * noise


def psnr(prediction: Tensor, clean: Tensor) -> float:
    """10 log10(255² / MSE), with the MSE over all pixels of all images (method.md 17.6)."""
    error = (to_pixels(prediction).double() - to_pixels(clean).double()) ** 2
    return 10 * math.log10(255**2 / error.mean().item())


def denoising_metrics(prediction: Tensor, clean: Tensor) -> dict[str, float]:
    """Smooth L1 (β = 1) and MSE in the [−1, 1] scale, and PSNR, over a whole split."""
    return {
        "smooth_l1": F.smooth_l1_loss(prediction, clean, beta=1.0).item(),
        "mse": F.mse_loss(prediction, clean).item(),
        "psnr": psnr(prediction, clean),
    }
