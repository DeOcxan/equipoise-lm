"""Training gradients on the Apple MPS backend against CPU, in float32."""

import pytest
import torch

from ebt.config import ThinkingConfig, TrainingMode
from ebt.core.objective import trajectory_loss
from ebt.core.thinking import ThinkingLoop
from support import TransformedTanhEnergy


def squared_error(prediction, target):
    return ((prediction - target) ** 2).sum()


def parameter_gradients(device, mode):
    model = TransformedTanhEnergy(context_dim=2, candidate_dim=3, hidden=8, seed=15).to(device)
    generator = torch.Generator().manual_seed(16)
    context = torch.randn(6, 2, generator=generator).to(device)
    initial = torch.randn(6, 3, generator=generator).to(device)
    target = torch.randn(6, 3, generator=generator).to(device)
    loop = ThinkingLoop(
        ThinkingConfig(mode=mode, step_size=0.3, step_size_learnable=True, steps_min=3, steps_max=3)
    ).to(device)
    trajectory = loop.train_trajectory(model, context, initial)
    loss = trajectory_loss(trajectory.candidates[1:], target, squared_error, mode)
    parameters = [*model.parameters(), loop.step_size.raw]
    return [gradient.cpu() for gradient in torch.autograd.grad(loss, parameters)]


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend not available")
@pytest.mark.parametrize("mode", list(TrainingMode))
def test_mps_gradients_match_cpu(mode):
    cpu = parameter_gradients("cpu", mode)
    mps = parameter_gradients("mps", mode)
    for reference, computed in zip(cpu, mps, strict=True):
        # float32 defaults of assert_close: rtol 1.3e-6, atol 1e-5 (docs/numerics.md).
        torch.testing.assert_close(computed, reference)


def patch_energy_gradients(device):
    """S1 parameter gradients of a small patch-transformer energy (method.md section 17.3)."""
    from ebt.config import PatchTransformerConfig
    from ebt.models.patch_transformer import PatchEnergy

    torch.manual_seed(17)
    model = PatchEnergy(PatchTransformerConfig(image_size=8, patch=4, dim=16, depth=2, heads=2))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    model = model.to(device)
    generator = torch.Generator().manual_seed(18)
    clean = torch.randn(4, 1, 8, 8, generator=generator)
    noisy = (clean + 0.3 * torch.randn(4, 1, 8, 8, generator=generator)).to(device)
    clean = clean.to(device)
    loop = ThinkingLoop(
        ThinkingConfig(
            mode=TrainingMode.S1, step_size=3.0, step_size_learnable=True, steps_min=2, steps_max=2
        )
    ).to(device)
    trajectory = loop.train_trajectory(model, None, noisy)
    loss = trajectory_loss(trajectory.candidates[1:], clean, squared_error, TrainingMode.S1)
    parameters = [*model.parameters(), loop.step_size.raw]
    return [gradient.cpu() for gradient in torch.autograd.grad(loss, parameters)]


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend not available")
def test_mps_patch_energy_second_order_gradients_match_cpu():
    for reference, computed in zip(
        patch_energy_gradients("cpu"), patch_energy_gradients("mps"), strict=True
    ):
        torch.testing.assert_close(computed, reference)


def text_energy_gradients(device, layout):
    """S2 parameter gradients of a small causal text EBT: the double backward runs through
    RMSNorm, rotary positions, attention, SwiGLU and the norm_pred RMSNorm (method.md 18.3)."""
    from ebt.models.text_transformer import TextEnergy, TextTransformerConfig
    from ebt.tasks.text import cross_entropy

    torch.manual_seed(19)
    model = TextEnergy(TextTransformerConfig(vocabulary=9, dim=64, depth=2, layout=layout))
    model = model.to(device)
    generator = torch.Generator().manual_seed(20)
    context = torch.randint(0, 9, (3, 6), generator=generator).to(device)
    target = torch.randint(0, 9, (3, 6), generator=generator).to(device)
    initial = torch.randn(3, 6, 9, generator=generator).to(device)
    loop = ThinkingLoop(
        ThinkingConfig(
            mode=TrainingMode.S2,
            step_size=5.0,
            energy_scaled_step=True,
            steps_min=3,
            steps_max=3,
        )
    ).to(device)
    trajectory = loop.train_trajectory(model, context, initial)
    loss = cross_entropy(trajectory.candidates[-1], target)
    return [gradient.cpu() for gradient in torch.autograd.grad(loss, list(model.parameters()))]


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend not available")
@pytest.mark.parametrize("layout", ["full", "split"])
def test_mps_text_energy_s2_gradients_match_cpu(layout):
    cpu = text_energy_gradients("cpu", layout)
    mps = text_energy_gradients("mps", layout)
    # Absolute tolerance from the measured float32 rounding error (docs/numerics.md,
    # "Text EBT gradients"); the default rtol of float32.
    for reference, computed in zip(cpu, mps, strict=True):
        torch.testing.assert_close(computed, reference, rtol=1.3e-6, atol=7.2e-05)
