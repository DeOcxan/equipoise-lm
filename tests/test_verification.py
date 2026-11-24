"""Best-of-M selection and evaluation under torch.no_grad() (method.md section 5.2)."""

import torch

from ebt.config import InferenceConfig, ThinkingConfig
from support import TanhEnergy, float64_loop


def setup(samples, sequence=True):
    leading = (3, 4) if sequence else (3,)
    model = TanhEnergy(context_dim=2, candidate_dim=3, hidden=6, seed=8).double()
    generator = torch.Generator().manual_seed(9)
    context = torch.randn(*leading, 2, dtype=torch.float64, generator=generator)
    initial = torch.randn(samples, *leading, 3, dtype=torch.float64, generator=generator) * 2
    loop = float64_loop(ThinkingConfig(step_size=0.3))
    return model, context, initial, loop


def test_selected_candidate_has_the_lowest_final_energy_per_position():
    model, context, initial, loop = setup(samples=5)
    config = InferenceConfig(steps=3, samples=5)
    selection = loop.best_of(model, context, initial, config)

    assert selection.candidates.shape == initial.shape
    assert selection.final_energies.shape == (5, 3, 4)
    assert torch.equal(selection.best_index, selection.final_energies.argmin(dim=0))
    for b in range(3):
        for s in range(4):
            chosen = selection.best_index[b, s]
            assert torch.equal(selection.best[b, s], selection.candidates[chosen, b, s])
    best_energy = model.energy(context, selection.best)
    torch.testing.assert_close(best_energy, selection.final_energies.min(dim=0).values)


def test_one_sample_equals_plain_inference():
    model, context, initial, loop = setup(samples=1)
    config = InferenceConfig(steps=3, samples=1, langevin_std=0.2)
    selection = loop.best_of(model, context, initial, config, torch.Generator().manual_seed(10))
    plain = loop.infer(
        model, context, initial[0], config, torch.Generator().manual_seed(10), final_energy=True
    )
    assert torch.equal(selection.best, plain.candidates[-1])
    assert torch.equal(selection.final_energies[0], plain.final_energy)


def test_inference_and_selection_work_inside_no_grad():
    """The failure mode of evaluating inside torch.no_grad(): gradients must still be available."""
    model, context, initial, loop = setup(samples=3)
    config = InferenceConfig(steps=2, samples=3)
    reference_infer = loop.infer(model, context, initial[0], config)
    reference_select = loop.best_of(model, context, initial, config)
    with torch.no_grad():
        trajectory = loop.infer(model, context, initial[0], config, final_energy=True)
        selection = loop.best_of(model, context, initial, config)
    assert torch.equal(trajectory.candidates[-1], reference_infer.candidates[-1])
    assert torch.equal(selection.best, reference_select.best)
    assert not trajectory.candidates[-1].requires_grad
    assert not selection.best.requires_grad
    assert all(parameter.grad is None for parameter in model.parameters())
