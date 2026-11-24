"""Explicit attention and the patch-transformer energy and denoiser (method.md section 17.3)."""

import math

import pytest
import torch

from ebt.config import InferenceConfig, PatchTransformerConfig, ThinkingConfig
from ebt.models.attention import attention
from ebt.models.mlp import parameter_count
from ebt.models.patch_transformer import PatchDenoiser, PatchEnergy, sincos_positions, unpatchify
from support import float64_loop

TINY = PatchTransformerConfig(image_size=8, patch=4, dim=8, depth=1, heads=2)


def test_attention_matches_the_fused_kernel_and_applies_the_mask():
    generator = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 3, 5, 4, generator=generator) for _ in range(3))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(attention(q, k, v), expected)
    causal = torch.ones(5, 5, dtype=torch.bool).tril()
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(attention(q, k, v, causal), expected)


def test_attention_first_and_second_derivatives_against_finite_differences():
    generator = torch.Generator().manual_seed(1)
    inputs = tuple(
        torch.randn(1, 3, 2, dtype=torch.float64, generator=generator, requires_grad=True)
        for _ in range(3)
    )
    assert torch.autograd.gradcheck(attention, inputs)
    assert torch.autograd.gradgradcheck(attention, inputs)


def test_sincos_positions_on_hand_values():
    """dim 8, grid 2: per axis 4 channels [sin p, sin(p/100), cos p, cos(p/100)]; the first
    half is the column, the second the row; patch 3 is row 1, column 1."""
    positions = sincos_positions(8, 2)
    assert positions.shape == (4, 8)
    one = [math.sin(1), math.sin(0.01), math.cos(1), math.cos(0.01)]
    zero = [0.0, 0.0, 1.0, 1.0]
    torch.testing.assert_close(positions[1], torch.tensor(one + zero))
    torch.testing.assert_close(positions[2], torch.tensor(zero + one))
    torch.testing.assert_close(positions[3], torch.tensor(one + one))


def test_unpatchify_places_each_patch_in_row_major_order():
    config = PatchTransformerConfig(image_size=4, patch=2, dim=8, heads=2)
    patches = torch.arange(16.0).reshape(1, 4, 4)
    expected = torch.tensor(
        [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], dtype=torch.float32
    )
    torch.testing.assert_close(unpatchify(patches, config)[0, 0], expected)


def test_zero_initialized_head_gives_zero_energy_and_zero_output():
    images = torch.randn(3, 1, 8, 8)
    assert torch.equal(PatchEnergy(TINY).energy(None, images), torch.zeros(3))
    denoiser = PatchDenoiser(TINY)
    with torch.no_grad():
        output = denoiser(images)
    assert output.shape == images.shape
    assert torch.equal(output, torch.zeros_like(images))


def test_energy_contract_batched_step_equals_steps_on_single_images():
    torch.manual_seed(2)
    model = PatchEnergy(TINY).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    images = torch.randn(4, 1, 8, 8, dtype=torch.float64)
    loop = float64_loop(ThinkingConfig(step_size=3.0))
    config = InferenceConfig(steps=2)
    batched = loop.infer(model, None, images, config).candidates[-1]
    single = torch.cat(
        [loop.infer(model, None, images[i : i + 1], config).candidates[-1] for i in range(4)]
    )
    torch.testing.assert_close(batched, single, rtol=1e-12, atol=1e-12)
    assert not torch.equal(batched, images)


def test_size_a_parameter_counts_within_one_percent():
    config = PatchTransformerConfig(patch=4, dim=128, depth=4, heads=4)
    energy, denoiser = parameter_count(PatchEnergy(config)), parameter_count(PatchDenoiser(config))
    assert (energy, denoiser) == (828_032, 829_968)
    assert abs(energy - denoiser) / energy < 0.01


def test_invalid_sizes_are_rejected():
    with pytest.raises(ValueError):
        PatchTransformerConfig(image_size=28, patch=5)
    with pytest.raises(ValueError):
        PatchTransformerConfig(dim=10, heads=4)
