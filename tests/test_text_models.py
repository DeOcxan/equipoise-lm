"""Causal text EBT and baseline: the thinking update, layouts (U6), energy contract, step-index
token, positions, causality, initialization, second order and pass counts (method.md sections
3.1, 4.1, 6.3, 11, 14 and 18.3)."""

from dataclasses import replace

import pytest
import torch

from ebt.config import InferenceConfig, ThinkingConfig, TrainingMode
from ebt.core.thinking import ThinkingLoop
from ebt.models.causal_attention import rotary, stream_positions, two_stream_mask
from ebt.models.text_transformer import (
    STEP_TOKENS,
    CausalTransformer,
    TextEnergy,
    TextTransformerConfig,
)

VOCABULARY, LENGTH = 11, 7
CONFIG = TextTransformerConfig(vocabulary=VOCABULARY, dim=64, depth=2, layout="full")


def energy_pair(seed=0):
    """The same float64 energy in the full and in the split layout."""
    torch.manual_seed(seed)
    full = TextEnergy(CONFIG).double()
    split = TextEnergy(replace(CONFIG, layout="split")).double()
    split.load_state_dict(full.state_dict())
    return full, split


CHANGED = 1e-6
"""A float64 energy difference above this is a change, not rounding."""


def inputs(rows=3, seed=1):
    generator = torch.Generator().manual_seed(seed)
    context = torch.randint(0, VOCABULARY, (rows, LENGTH), generator=generator)
    candidate = torch.randn(rows, LENGTH, VOCABULARY, generator=generator, dtype=torch.float64)
    return context, candidate


def energy_of(model, context, candidate):
    """Energy at the thinking variable of `candidate` (logits), as the loop evaluates it."""
    return model.energy(context, model.encode_candidate(model.transform_candidate(candidate)))


def first_and_second_order(model, context, candidate):
    """Energies, candidate gradients, parameter gradients of the energy and parameter
    gradients of a loss on the candidate gradient (create_graph), all float64."""
    candidate = candidate.clone().requires_grad_(True)
    energy = energy_of(model, context, candidate)
    # norm_pred acts after a step and is not part of the energy.
    parameters = [p for name, p in model.named_parameters() if not name.startswith("norm_pred")]
    (candidate_gradient,) = torch.autograd.grad(energy.sum(), candidate, create_graph=True)
    first = torch.autograd.grad(energy.sum(), parameters, retain_graph=True)
    second = torch.autograd.grad(candidate_gradient.pow(2).sum(), parameters)
    return energy.detach(), candidate_gradient.detach(), first, second


def test_u6_layouts_agree_in_energies_and_gradients():
    """U6 both layouts agree in energies and first and second order gradients."""
    full, split = energy_pair()
    context, candidate = inputs()
    energy_full, gradient_full, first_full, second_full = first_and_second_order(
        full, context, candidate
    )
    energy_split, gradient_split, first_split, second_split = first_and_second_order(
        split, context, candidate
    )
    assert energy_full.shape == (3, LENGTH)
    torch.testing.assert_close(energy_split, energy_full, rtol=1e-10, atol=0)
    torch.testing.assert_close(gradient_split, gradient_full, rtol=1e-10, atol=1e-14)
    for a, b in zip(first_split, first_full, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-13)
    for a, b in zip(second_split, second_full, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-13)
    assert all(g.abs().sum() > 0 for g in second_full)


def test_one_thinking_step_is_the_softmax_space_update():
    """Hand-built step: p = softmax(ŷ + η); ŷ' = norm_pred(p - α exp(E / 9) ∇_p E), with the
    gradient taken with respect to p (method.md section 4.1)."""
    torch.manual_seed(0)
    model = TextEnergy(replace(CONFIG, layout="split")).double()
    with torch.no_grad():
        model.norm_pred.weight.uniform_(0.5, 1.5)
    context, candidate = inputs()
    sigma, alpha = 3.0, 5.0
    loop = ThinkingLoop(ThinkingConfig(step_size=alpha, energy_scaled_step=True)).double()
    trajectory = loop.infer(
        model,
        context,
        candidate,
        InferenceConfig(steps=1, langevin_std=sigma),
        torch.Generator().manual_seed(4),
    )

    generator = torch.Generator().manual_seed(4)
    noise = torch.randn(candidate.shape, generator=generator, dtype=torch.float64)
    probabilities = (candidate + sigma * noise).softmax(dim=-1).requires_grad_(True)
    energy = model.energy(context, model.vocab_to_embed(probabilities))
    (gradient,) = torch.autograd.grad(energy.sum(), probabilities)
    stepped = probabilities - alpha * torch.exp(energy / 9).unsqueeze(-1) * gradient
    eps = torch.finfo(stepped.dtype).eps  # nn.RMSNorm with eps=None, as upstream
    assert model.norm_pred.eps is None
    rms = stepped.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    expected = stepped / rms * model.norm_pred.weight

    torch.testing.assert_close(trajectory.candidates[1], expected.detach(), rtol=1e-12, atol=0)
    torch.testing.assert_close(trajectory.energies[0], energy.detach(), rtol=1e-12, atol=0)


def test_mask_lets_prediction_j_see_step_token_prefix_and_itself():
    # Columns: step token, observed 0, 1, 2, predictions 0, 1, 2.
    """The mask lets prediction j see the step token observed 0 to j and itself."""
    mask = two_stream_mask(3, torch.device("cpu"), prefix=1)
    assert mask[0].tolist() == [True, False, False, False, False, False, False]
    assert mask[2].tolist() == [True, True, True, False, False, False, False]
    assert mask[4].tolist() == [True, True, False, False, True, False, False]
    assert mask[6].tolist() == [True, True, True, True, False, False, True]


@pytest.mark.parametrize("layout", ["full", "split"])
def test_energy_at_j_ignores_other_candidates_and_later_tokens(layout):
    """Finite perturbations, float64 (method.md section 3.1).

    Each perturbation changes a single logit per position: adding a constant to every logit of
    a position would leave its softmax unchanged and test nothing. A real change is one above
    CHANGED, far above float64 rounding of these energies (about 1e-15) and far below the
    changes measured (0.009 and more).
    """
    torch.manual_seed(0)
    model = TextEnergy(replace(CONFIG, layout=layout)).double()
    context, candidate = inputs(rows=1)
    j = 3
    energies = energy_of(model, context, candidate)
    reference = energies[0, j]

    changed = candidate.clone()
    changed[0, j + 1 :, 0] += 5.0
    changed[0, :j, 1] -= 5.0
    after = energy_of(model, context, changed)
    after_candidates = after[0, j]
    others = [i for i in range(LENGTH) if i != j]
    assert ((after[0, others] - energies[0, others]).abs() > CHANGED).all()

    later = context.clone()
    later[0, j + 1 :] = (later[0, j + 1 :] + 1) % VOCABULARY
    after_context = energy_of(model, later, candidate)[0, j]

    assert after_candidates.item() == reference.item()
    assert after_context.item() == reference.item()
    own = candidate.clone()
    own[0, j, 0] += 5.0
    assert abs(energy_of(model, context, own)[0, j].item() - reference.item()) > CHANGED


@pytest.mark.parametrize("layout", ["full", "split"])
def test_every_prediction_attends_to_the_step_token(layout):
    torch.manual_seed(0)
    model = TextEnergy(replace(CONFIG, layout=layout)).double()
    context, candidate = inputs(rows=1)
    reference = energy_of(model, context, candidate)
    with torch.no_grad():
        model.step_embedding.weight.add_(1.0)
    assert ((energy_of(model, context, candidate) - reference).abs() > CHANGED).all()


def test_rotary_positions_of_the_step_token_observed_tokens_and_predictions():
    observed, predicted = stream_positions(5, torch.device("cpu"), prefix=STEP_TOKENS)
    assert observed.tolist() == [0, 1, 2, 3, 4, 5]  # step token, then observed i at i + 1
    assert predicted.tolist() == [2, 3, 4, 5, 6]  # prediction j at j + 2
    baseline, _ = stream_positions(5, torch.device("cpu"))
    assert baseline.tolist() == [0, 1, 2, 3, 4]


def test_rotary_scores_depend_on_the_position_difference_only():
    generator = torch.Generator().manual_seed(2)
    query = torch.randn(1, 16, generator=generator, dtype=torch.float64)
    key = torch.randn(1, 16, generator=generator, dtype=torch.float64)

    def score(a, b):
        rotated_query = rotary(query, torch.tensor([a]))
        rotated_key = rotary(key, torch.tensor([b]))
        return (rotated_query * rotated_key).sum().item()

    assert score(5, 2) == pytest.approx(score(9, 6), rel=1e-12)
    assert score(5, 2) != pytest.approx(score(5, 3), rel=1e-6)


def test_baseline_is_causal():
    torch.manual_seed(0)
    model = CausalTransformer(CONFIG).double()
    context, _ = inputs(rows=1)
    j = 2
    reference = model(context)[0, : j + 1]
    later = context.clone()
    later[0, j + 1 :] = (later[0, j + 1 :] + 1) % VOCABULARY
    torch.testing.assert_close(model(later)[0, : j + 1], reference, rtol=0, atol=0)


def test_ebt_has_2d_plus_v_more_parameters_than_the_baseline():
    """Step embedding D, V -> D map V·D, D -> 1 head D and norm_pred V, against the baseline's
    D -> V head V·D."""

    def count(module):
        return sum(parameter.numel() for parameter in module.parameters())

    ebt, baseline = TextEnergy(CONFIG), CausalTransformer(CONFIG)
    assert count(ebt) - count(baseline) == 2 * CONFIG.dim + VOCABULARY


@pytest.mark.parametrize("model_class", [TextEnergy, CausalTransformer])
def test_init_is_xavier_normal_with_separate_qkv_blocks(model_class):
    """Initialization is xavier normal with separate query key value blocks."""
    torch.manual_seed(0)
    dim = 512
    model = model_class(TextTransformerConfig(vocabulary=65, dim=dim, depth=1))
    block = model.backbone.blocks[0]
    for part in block.qkv.weight.chunk(3):
        assert part.std().item() == pytest.approx((2 / (dim + dim)) ** 0.5, rel=0.02)
        assert part.mean().abs().item() < 0.002
    hidden = 4 * dim
    assert block.gate.weight.shape == (hidden, dim) and block.down.weight.shape == (dim, hidden)
    assert block.gate.weight.std().item() == pytest.approx((2 / (dim + hidden)) ** 0.5, rel=0.02)
    assert model.backbone.embedding.weight.std().item() == pytest.approx(0.02, rel=0.1)
    assert block.attention_norm.weight.eq(1).all()
    if model_class is TextEnergy:
        expected = (2 / (65 + dim)) ** 0.5
        assert model.vocab_to_embed.weight.std().item() == pytest.approx(expected, rel=0.02)
        # PyTorch's default N(0, 1), not re-initialized (as upstream).
        assert model.step_embedding.weight.std().item() == pytest.approx(1.0, rel=0.15)


@pytest.mark.parametrize("layout", ["full", "split"])
def test_energy_passes_gradcheck_and_gradgradcheck(layout):
    """First and second derivatives of the energy with respect to the candidate, float64."""
    torch.manual_seed(0)
    config = TextTransformerConfig(vocabulary=5, dim=64, depth=1, layout=layout)
    model = TextEnergy(config).double()
    generator = torch.Generator().manual_seed(3)
    context = torch.randint(0, 5, (2, 3), generator=generator)
    candidate = torch.randn(2, 3, 5, generator=generator, dtype=torch.float64)
    candidate.requires_grad_(True)

    def energy(logits):
        return energy_of(model, context, logits)

    assert torch.autograd.gradcheck(energy, (candidate,))
    assert torch.autograd.gradgradcheck(energy, (candidate,))


def test_pass_counts_of_s2_training_and_inference():
    """S2 training with k steps: k forward, k backward, 1 second-order; inference with 4 steps:
    4 forward, 4 backward and no final energy (method.md section 6.3)."""
    torch.manual_seed(0)
    model = TextEnergy(replace(CONFIG, layout="split"))
    context, candidate = inputs()
    loop = ThinkingLoop(
        ThinkingConfig(mode=TrainingMode.S2, step_size=5.0, steps_min=3, steps_max=3)
    )
    training = loop.train_trajectory(model, context, candidate.float())
    assert (training.cost.forward, training.cost.backward, training.cost.second_order) == (3, 3, 1)
    inference = loop.infer(model, context, candidate.float(), InferenceConfig(steps=4))
    assert (inference.cost.forward, inference.cost.backward, inference.cost.second_order) == (
        4,
        4,
        0,
    )


def test_norm_pred_is_applied_after_every_step():
    torch.manual_seed(0)
    model = TextEnergy(replace(CONFIG, layout="split"))
    context, candidate = inputs()
    loop = ThinkingLoop(ThinkingConfig(mode=TrainingMode.S2, step_size=5.0))
    final = loop.infer(model, context, candidate.float(), InferenceConfig(steps=2)).candidates[-1]
    root_mean_square = final.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(
        root_mean_square, torch.ones_like(root_mean_square), atol=1e-4, rtol=0
    )
