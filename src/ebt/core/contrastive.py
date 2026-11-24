"""Contrastive energy losses (method.md sections 16.4 and 16.8)."""

import torch
import torch.nn.functional as F
from torch import Tensor


def contrastive_loss(real_energy: Tensor, fake_energy: Tensor) -> Tensor:
    """Mean of softplus(E_real − E_fake) over all energy elements (method.md section 16).

    E_real is the energy of the true target, E_fake the energy of the model's own candidate.
    softplus(E_real − E_fake) equals the cross-entropy of the logits (−E_real, −E_fake) with
    the real sample as the correct class, which is the form of the upstream loss. It lowers
    the energy of the target and raises the energy of the candidate.
    """
    if real_energy.shape != fake_energy.shape:
        raise ValueError("real and fake energies must have the same shape")
    return F.softplus(real_energy - fake_energy).mean()


def energy_magnitude(real_energy: Tensor, fake_energy: Tensor) -> Tensor:
    """Mean of E_real² + E_fake² over all energy elements (method.md section 16.4).

    The L2 penalty on energy magnitudes of Du and Mordatch (2019), arXiv:1903.08689, section 3.3
    and Algorithm 1. It keeps the energy scale bounded while the contrastive loss separates real
    and fake energies.
    """
    if real_energy.shape != fake_energy.shape:
        raise ValueError("real and fake energies must have the same shape")
    return (real_energy**2 + fake_energy**2).mean()


def info_nce(real_energy: Tensor, counter_energy: Tensor) -> Tensor:
    """InfoNCE loss, mean over rows (method.md section 16.8; Florence et al. 2021, §2).

    `real_energy` has shape (B,) and `counter_energy` shape (B, N). The loss of a row is
    −log(e^{−E_real} / (e^{−E_real} + Σ_j e^{−E_j})), the cross-entropy of the logits −E with the
    true sample as the correct class. With N = 1 it equals `contrastive_loss`.
    """
    if counter_energy.shape[:-1] != real_energy.shape:
        raise ValueError("counter energies must have shape (*real_energy.shape, N)")
    energies = torch.cat([real_energy.unsqueeze(-1), counter_energy], dim=-1)
    return (real_energy + torch.logsumexp(-energies, dim=-1)).mean()
