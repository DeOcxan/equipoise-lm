"""Replay buffer (method.md section 9)."""

import torch

from ebt.config import InferenceConfig, ReplayConfig, ThinkingConfig
from ebt.core.analytic import QuadraticEnergy
from ebt.core.replay import ReplayBuffer
from support import float64_loop


def rows(start, count):
    """Rows whose context, target and candidate encode their row number."""
    values = torch.arange(start, start + count, dtype=torch.float64).unsqueeze(-1)
    return values.clone(), values.clone() * 10, values.clone() * 100


def store_fresh(buffer, start, count):
    context, target, candidate = rows(start, count)
    batch = buffer.extend(context, target, torch.zeros_like(candidate))
    buffer.store(batch, torch.cat([candidate, torch.zeros_like(batch.initial[batch.fresh :])]))
    return batch


def contents(buffer):
    """Row numbers of every stored row, read through `extend` with fraction 1."""
    context, target, candidate = rows(0, len(buffer))
    batch = buffer.extend(context, target, candidate)
    return sorted(batch.context[batch.fresh :].squeeze(-1).tolist())


def test_capacity_and_first_in_first_out_order():
    buffer = ReplayBuffer(ReplayConfig(capacity=4, fraction=1.0), torch.Generator().manual_seed(4))
    store_fresh(buffer, 0, 3)
    store_fresh(buffer, 3, 3)
    assert len(buffer) == 4
    assert contents(buffer) == [2.0, 3.0, 4.0, 5.0]


def test_sample_count_is_the_fraction_of_the_batch_limited_by_the_buffer():
    buffer = ReplayBuffer(
        ReplayConfig(capacity=100, fraction=0.5), torch.Generator().manual_seed(0)
    )
    context, target, candidate = rows(0, 5)
    empty = buffer.extend(context, target, candidate)
    assert empty.initial.shape[0] == 5 and empty.fresh == 5

    store_fresh(buffer, 0, 1)
    assert buffer.extend(context, target, candidate).initial.shape[0] == 5 + 1

    store_fresh(buffer, 10, 10)
    batch = buffer.extend(context, target, candidate)
    assert batch.initial.shape[0] == 5 + int(0.5 * 5)


def test_replayed_rows_are_distinct_and_consistent():
    buffer = ReplayBuffer(
        ReplayConfig(capacity=100, fraction=1.0), torch.Generator().manual_seed(1)
    )
    store_fresh(buffer, 0, 8)
    context, target, candidate = rows(50, 8)
    batch = buffer.extend(context, target, candidate)
    replayed_context = batch.context[batch.fresh :].squeeze(-1)
    assert replayed_context.unique().numel() == 8
    torch.testing.assert_close(batch.target[batch.fresh :], batch.context[batch.fresh :] * 10)
    torch.testing.assert_close(batch.initial[batch.fresh :], batch.context[batch.fresh :] * 100)
    torch.testing.assert_close(batch.initial[: batch.fresh], candidate)


def test_only_fresh_rows_are_stored():
    buffer = ReplayBuffer(
        ReplayConfig(capacity=100, fraction=1.0), torch.Generator().manual_seed(2)
    )
    store_fresh(buffer, 0, 4)
    batch = store_fresh(buffer, 4, 4)
    assert batch.initial.shape[0] == 8
    assert len(buffer) == 8
    assert contents(buffer) == list(map(float, range(8)))


def test_replayed_rows_restart_from_their_stored_candidates():
    """A loop step on the extended batch starts each replayed row at its stored candidate."""
    buffer = ReplayBuffer(
        ReplayConfig(capacity=100, fraction=1.0), torch.Generator().manual_seed(3)
    )
    stored_context = torch.zeros(3, 1, dtype=torch.float64)
    stored_target = torch.zeros(3, 1, dtype=torch.float64)
    stored_final = torch.tensor([[2.0], [4.0], [-6.0]], dtype=torch.float64)
    first = buffer.extend(stored_context, stored_target, torch.zeros_like(stored_final))
    buffer.store(first, stored_final)

    fresh_initial = torch.ones(3, 1, dtype=torch.float64)
    batch = buffer.extend(stored_context, stored_target, fresh_initial)
    model = QuadraticEnergy(torch.eye(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    loop = float64_loop(ThinkingConfig(step_size=0.5))
    trajectory = loop.infer(model, batch.context, batch.initial, InferenceConfig(steps=1))

    after = trajectory.candidates[1]
    torch.testing.assert_close(after[: batch.fresh], 0.5 * fresh_initial)
    replayed_start = batch.initial[batch.fresh :]
    assert sorted(replayed_start.squeeze(-1).tolist()) == [-6.0, 2.0, 4.0]
    torch.testing.assert_close(after[batch.fresh :], 0.5 * replayed_start)
