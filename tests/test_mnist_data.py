"""MNIST files, splits, noise and PSNR on synthetic data (method.md sections 17.1, 17.2, 17.6)."""

import gzip
import hashlib
import math
import struct

import pytest
import torch

from ebt.tasks import mnist


def write_idx(path, values: torch.Tensor) -> None:
    header = struct.pack(">HBB", 0, 0x08, values.dim()) + struct.pack(
        f">{values.dim()}I", *values.shape
    )
    path.write_bytes(gzip.compress(header + bytes(values.flatten().tolist())))


def test_idx_reader_returns_the_stated_shape_and_values(tmp_path):
    values = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4)
    write_idx(tmp_path / "x.gz", values)
    assert torch.equal(mnist.read_idx(tmp_path / "x.gz"), values)


def test_idx_reader_rejects_a_wrong_length_and_a_wrong_type(tmp_path):
    header = struct.pack(">HBB", 0, 0x08, 1) + struct.pack(">I", 5)
    (tmp_path / "short.gz").write_bytes(gzip.compress(header + bytes(3)))
    with pytest.raises(ValueError, match="values"):
        mnist.read_idx(tmp_path / "short.gz")
    (tmp_path / "float.gz").write_bytes(
        gzip.compress(struct.pack(">HBBI", 0, 0x0D, 1, 1) + bytes(4))
    )
    with pytest.raises(ValueError, match="unsigned bytes"):
        mnist.read_idx(tmp_path / "float.gz")


def test_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    write_idx(tmp_path / mnist.TRAIN_IMAGES, torch.zeros(2, 2, 2, dtype=torch.uint8))
    write_idx(tmp_path / mnist.TEST_IMAGES, torch.zeros(1, 2, 2, dtype=torch.uint8))
    with pytest.raises(ValueError, match="SHA-256"):
        mnist.load_images(tmp_path)
    digests = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in mnist.SHA256
    }
    monkeypatch.setattr(mnist, "SHA256", digests)
    train, test = mnist.load_images(tmp_path)
    assert train.shape == (2, 2, 2) and test.shape == (1, 2, 2)


def test_scaling_to_signed_pixels_and_back():
    pixels = torch.tensor([[[0, 255], [51, 204]]], dtype=torch.uint8)
    signed = mnist.to_signed(pixels)
    assert signed.shape == (1, 1, 2, 2)
    torch.testing.assert_close(signed, torch.tensor([[[[-1.0, 1.0], [-0.6, 0.6]]]]))
    torch.testing.assert_close(mnist.to_pixels(signed), pixels.float().unsqueeze(1))
    torch.testing.assert_close(
        mnist.to_pixels(torch.tensor([-3.0, 3.0])), torch.tensor([0.0, 255.0])
    )


def test_splits_have_the_requested_sizes_and_are_deterministic():
    generator = torch.Generator().manual_seed(0)
    train = torch.randint(0, 256, (50, 4, 4), dtype=torch.uint8, generator=generator)
    test = torch.randint(0, 256, (7, 4, 4), dtype=torch.uint8, generator=generator)
    data = mnist.make_data(train, test, validation_size=10, seed=0)
    assert (len(data.train), len(data.validation), len(data.test)) == (40, 10, 7)
    assert data.train.context is None and data.train.target.shape == (40, 1, 4, 4)
    again = mnist.make_data(train, test, validation_size=10, seed=0)
    assert torch.equal(data.validation.target, again.validation.target)
    every = torch.cat([data.train.target, data.validation.target]).flatten(1)
    assert sorted(map(tuple, every.tolist())) == sorted(
        map(tuple, mnist.to_signed(train).flatten(1).tolist())
    )


def test_alpha_bar_against_the_closed_form_product():
    """ᾱ = Π_{i<100} (1 - β_i), β_i = s(1e-4 + i (2e-2 - 1e-4) / (T - 1)), s = 1000 / T."""
    for level, timesteps, table in [(0.1, 1000, 0.89702), (0.2, 500, 0.65972)]:
        scale = 1000 / timesteps
        product = 1.0
        for i in range(100):
            product *= 1 - scale * (1e-4 + i * (2e-2 - 1e-4) / (timesteps - 1))
        assert mnist.alpha_bar(level) == pytest.approx(product, rel=1e-12)
        assert round(mnist.alpha_bar(level), 5) == table


def test_noise_is_variance_preserving_and_seeded():
    images = torch.full((2, 1, 3, 3), 0.5)
    noisy = mnist.add_noise(images, 0.81, torch.Generator().manual_seed(1))
    noise = torch.randn(images.shape, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(noisy, 0.9 * images + math.sqrt(0.19) * noise)


def test_psnr_on_hand_values():
    """Every pixel 255 · 0.05 = 12.75 off: PSNR = 10 log10(255**2 / 12.75**2) = 20 log10(20)."""
    clean = torch.zeros(2, 1, 2, 2)
    prediction = clean + 0.1
    assert mnist.psnr(prediction, clean) == pytest.approx(20 * math.log10(20), abs=1e-9)
    metrics = mnist.denoising_metrics(prediction, clean)
    assert metrics["mse"] == pytest.approx(0.01, rel=1e-6)
    assert metrics["smooth_l1"] == pytest.approx(0.005, rel=1e-6)


def test_psnr_uses_one_mse_over_the_whole_set():
    """Errors 0.1 and 0.3 on two images: one MSE of 0.05, not the mean of two PSNRs."""
    clean = torch.zeros(2, 1, 1, 1)
    prediction = torch.tensor([0.1, 0.3]).reshape(2, 1, 1, 1)
    mse = (255 * 0.5) ** 2 * (0.1**2 + 0.3**2) / 2
    assert mnist.psnr(prediction, clean) == pytest.approx(10 * math.log10(255**2 / mse), abs=1e-9)
