"""IRED energy, schedule, loss and annealed inference (method.md section 16.8)."""

import math

import pytest
import torch
from torch import Tensor, nn

from ebt.config import InferenceConfig, ThinkingConfig
from ebt.core.energy import EnergyModel
from ebt.inference.annealed import (
    CLIP,
    LATER_STEP_FACTOR,
    LandscapeSteps,
    accept_lower_step,
    annealed_sample,
)
from ebt.models.ired import (
    LandscapeEnergy,
    OnLandscape,
    cosine_schedule,
    landscape_context,
    sinusoidal_embedding,
)
from ebt.models.mlp import parameter_count
from ebt.training.ired import IREDObjective, denoising_loss
from support import float64_loop


def curve(t: float, landscapes: int = 10, s: float = 0.008) -> float:
    return math.cos((t / landscapes + s) / (1 + s) * math.pi / 2) ** 2


def test_cosine_schedule_against_its_closed_form():
    """Without clipping the product telescopes: ᾱ_k = f(k + 1) / f(0) (method.md 16.8)."""
    schedule = cosine_schedule(10)
    for k in range(9):
        assert schedule.alpha_bar[k].item() == pytest.approx(curve(k + 1) / curve(0), rel=1e-6)
    measured = [0.9721, 0.8987, 0.7869, 0.6475, 0.4938, 0.3408, 0.2031, 0.0940, 0.0241]
    assert [round(v, 4) for v in schedule.alpha_bar[:9].tolist()] == measured
    assert schedule.betas[9].item() == pytest.approx(0.999)
    expected_steps = schedule.betas / torch.sqrt(1 - schedule.alpha_bar)
    torch.testing.assert_close(schedule.step_size, expected_steps)
    assert schedule.step_size[0].item() == pytest.approx(0.167, abs=5e-4)


def test_sinusoidal_embedding_on_hand_values():
    """dim 4: frequencies 1 and 10**-4; index 1 gives [sin 1, sin 10**-4, cos 1, cos 10**-4]."""
    index = torch.tensor([0.0, 1.0], dtype=torch.float64)
    expected = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [math.sin(1), math.sin(1e-4), math.cos(1), math.cos(1e-4)]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(sinusoidal_embedding(index, 4), expected)


def test_parameter_count_is_within_one_percent_of_the_energy():
    count = parameter_count(LandscapeEnergy(1, 1, hidden=37, embedding=8, time=8))
    assert count == 4437
    assert abs(count - 4416) / 4416 < 0.01


def small_energy(seed: int = 0) -> LandscapeEnergy:
    torch.manual_seed(seed)
    return LandscapeEnergy(1, 1, hidden=6, embedding=4, time=4).double()


def test_energy_contract_with_the_landscape_in_the_context():
    """A batched thinking step equals the steps on single rows, rows on different landscapes."""
    model = small_energy()
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(5, 1, dtype=torch.float64, generator=generator)
    context = landscape_context(x, torch.tensor([0, 3, 9, 3, 1]))
    initial = torch.randn(5, 1, dtype=torch.float64, generator=generator)
    loop = float64_loop(ThinkingConfig(step_size=0.4))
    config = InferenceConfig(steps=2)
    batched = loop.infer(model, context, initial, config).candidates[-1]
    single = torch.cat(
        [
            loop.infer(model, context[i : i + 1], initial[i : i + 1], config).candidates[-1]
            for i in range(5)
        ]
    )
    torch.testing.assert_close(batched, single, rtol=1e-14, atol=1e-14)
    on_landscape = OnLandscape(model, 3).energy(x, initial)
    torch.testing.assert_close(on_landscape, model.energy(landscape_context(x, 3), initial))


class QuadraticEnergy(EnergyModel):
    """E(x, y; k) = c y**2, with ∇_y E = 2 c y."""

    candidate_event_dims = 1

    def __init__(self, c: float) -> None:
        super().__init__()
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float64))

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        return self.c * (encoded**2).sum(-1)


def test_denoising_loss_and_its_gradient_on_a_quadratic_energy():
    """loss = mean((2c ỹ - ε)**2); ∂loss/∂c = mean(4 ỹ (2c ỹ - ε))."""
    model = QuadraticEnergy(0.7)
    noisy = torch.tensor([[0.5], [-1.0], [2.0]], dtype=torch.float64)
    noise = torch.tensor([[0.1], [0.3], [-0.2]], dtype=torch.float64)
    context = landscape_context(torch.zeros(3, 1, dtype=torch.float64), 0)
    loss = denoising_loss(model, context, noisy, noise)
    residual = 2 * 0.7 * noisy - noise
    assert loss.item() == pytest.approx((residual**2).mean().item(), abs=1e-15)
    (gradient,) = torch.autograd.grad(loss, model.c)
    assert gradient.item() == pytest.approx((4 * noisy * residual).mean().item(), abs=1e-14)


def test_denoising_loss_parameter_gradient_against_central_differences():
    """The parameter gradient goes through ∇_y E, i.e. it is second order (float64)."""
    model = small_energy(2)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(4, 1, dtype=torch.float64, generator=generator)
    context = landscape_context(x, torch.tensor([0, 2, 5, 9]))
    noisy = torch.randn(4, 1, dtype=torch.float64, generator=generator)
    noise = torch.randn(4, 1, dtype=torch.float64, generator=generator)
    parameter = model.fc2.weight
    (analytic,) = torch.autograd.grad(denoising_loss(model, context, noisy, noise), parameter)
    h = 1e-6
    for index in [(0, 0), (2, 3), (5, 1)]:
        with torch.no_grad():
            parameter[index] += h
        upper = denoising_loss(model, context, noisy, noise).item()
        with torch.no_grad():
            parameter[index] -= 2 * h
        lower = denoising_loss(model, context, noisy, noise).item()
        with torch.no_grad():
            parameter[index] += h
        numeric = (upper - lower) / (2 * h)
        assert analytic[index].item() == pytest.approx(numeric, rel=1e-6, abs=1e-10)


class WavyEnergy(EnergyModel):
    """E(x, y) = cos(3 y): a large gradient step lowers it for some rows and raises it for
    others."""

    candidate_event_dims = 1

    def energy(self, context: Tensor | None, encoded: Tensor) -> Tensor:
        return torch.cos(3 * encoded).sum(-1)


def test_accept_only_lower_step_never_raises_the_energy():
    model = WavyEnergy()
    generator = torch.Generator().manual_seed(5)
    context = landscape_context(torch.zeros(64, 1, dtype=torch.float64), 2)
    candidate = torch.randn(64, 1, dtype=torch.float64, generator=generator)
    loop = float64_loop(ThinkingConfig(step_size=0.7))
    new, cost = accept_lower_step(model, loop, context, candidate, bound=3.0)
    with torch.no_grad():
        before, after = model.energy(context, candidate), model.energy(context, new)
    assert bool((after <= before).all())
    changed = (new != candidate).squeeze(-1)
    assert 0 < int(changed.sum()) < 64, "the large step should be accepted for some rows only"
    assert new.abs().max().item() <= 3.0
    assert (cost.forward, cost.backward) == (2, 1)


def test_loops_have_the_step_sizes_of_the_authors_code():
    schedule = cosine_schedule(10)
    steps = LandscapeSteps(schedule)
    for k in range(10):
        a, lam = schedule.alpha_bar[k].item(), schedule.step_size[k].item()
        assert steps.jump[k].step_size.raw.item() == pytest.approx(math.sqrt(1 - a), rel=1e-6)
        assert steps.first[k].step_size.raw.item() == pytest.approx(lam, rel=1e-6)
        assert steps.later[k].step_size.raw.item() == pytest.approx(
            LATER_STEP_FACTOR * lam, rel=1e-6
        )
        assert steps.bound(k) == pytest.approx(CLIP * math.sqrt(a), rel=1e-6)


def test_sampler_clamps_and_rescales_between_landscapes():
    """With zero energy no step moves the candidate, so only the jump's clamp to ±2 and the
    rescaling act: every row ends at sign(ŷ_init) · 2 √ᾱ_0."""
    model = LandscapeEnergy(1, 1, hidden=6, embedding=4, time=4)
    nn.init.zeros_(model.fc4.weight)
    nn.init.zeros_(model.fc4.bias)
    steps = LandscapeSteps(cosine_schedule(10))
    initial = torch.tensor([[0.3], [-0.01], [1.5]])
    final, cost = annealed_sample(model, steps, torch.zeros(3, 1), initial)
    expected = torch.sign(initial) * CLIP * steps.schedule.alpha_bar[0].sqrt()
    torch.testing.assert_close(final, expected)
    assert (cost.forward, cost.backward) == (110, 60)


def test_objective_loss_is_finite_and_counts_its_passes():
    model = LandscapeEnergy(1, 1, hidden=6, embedding=4, time=4)
    objective = IREDObjective(model, LandscapeSteps(cosine_schedule(10)))
    from ebt.training.batch import Batch

    generator = torch.Generator().manual_seed(6)
    batch = Batch(torch.randn(16, 1, generator=generator), torch.randn(16, 1, generator=generator))
    result = objective.batch_loss(batch, torch.Generator().manual_seed(7))
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert (result.cost.forward, result.cost.backward, result.cost.second_order) == (112, 80, 16)
