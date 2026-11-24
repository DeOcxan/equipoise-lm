"""The thinking step size α (method.md sections 8.1 and 10)."""

import torch
from torch import Tensor, nn

MIN_STEP_SIZE = 1e-4
"""Lower clamp applied to α before use (method.md section 10)."""

ENERGY_SCALE_TEMPERATURE = 9.0
"""T in the energy-scaled step α · exp(E / T); the only value the upstream scripts use."""


class StepSize(nn.Module):
    """Raw scalar step size, optionally learnable, jittered and energy-scaled.

    Implements method.md sections 8.1 and 10.
    """

    def __init__(self, value: float, learnable: bool, jitter: float, energy_scaled: bool) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(float(value)), requires_grad=learnable)
        self.jitter = jitter
        self.energy_scaled = energy_scaled

    def sample(
        self, energy_shape: torch.Size, training: bool, generator: torch.Generator | None
    ) -> Tensor:
        """One α per energy element, drawn once per call (method.md section 8.1).

        In training with jitter r > 1, each element is uniform in [α / r, α · r]; otherwise
        every element equals the clamped α.
        """
        alpha = self.raw.clamp(min=MIN_STEP_SIZE)
        if not training or self.jitter == 1.0:
            return alpha.expand(energy_shape)
        low = alpha / self.jitter
        high = alpha * self.jitter
        u = torch.rand(energy_shape, generator=generator, dtype=alpha.dtype, device=alpha.device)
        return low + u * (high - low)

    def scale(self, alpha: Tensor, energy: Tensor) -> Tensor:
        """α · exp(E / T) per energy element when energy-scaled, otherwise α (method.md 10)."""
        if not self.energy_scaled:
            return alpha
        return alpha * torch.exp(energy / ENERGY_SCALE_TEMPERATURE)
