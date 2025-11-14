"""Exploration policies for stochastic route selection."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def epsilon_greedy(scores: Sequence[float], epsilon: float) -> int:
    """Return candidate index via epsilon-greedy (lower score is better)."""

    if not scores:
        raise ValueError("Scores list empty")
    if random.random() < max(0.0, epsilon):
        return random.randrange(len(scores))
    return int(np.argmin(scores))


def softmax_sample(scores: Sequence[float], temperature: float) -> int:
    """Sample from softmax over negative scores."""

    if temperature <= 0:
        return int(np.argmin(scores))
    scaled = np.array(scores, dtype=np.float64)
    scaled = -scaled / temperature
    probs = np.exp(scaled - scaled.max())
    probs /= probs.sum()
    choice = np.random.choice(len(scores), p=probs)
    return int(choice)


def gumbel_softmax_mixture(scores: Iterable[float], temperature: float) -> torch.Tensor:
    """Return differentiable mixture weights using Gumbel-Softmax."""

    logits = torch.tensor(scores, dtype=torch.float32)
    weights = F.gumbel_softmax(-logits, tau=max(temperature, 1e-3), hard=False)
    return weights


def linear_anneal(step: int, start: float, end: float, total_steps: int) -> float:
    """Linear interpolation schedule."""

    if total_steps <= 0:
        return end
    frac = min(max(step, 0), total_steps) / total_steps
    return float(start + frac * (end - start))


def cosine_anneal(step: int, start: float, end: float, total_steps: int) -> float:
    """Cosine annealing schedule."""

    if total_steps <= 0:
        return end
    frac = min(max(step, 0), total_steps) / total_steps
    cos_val = (1 + math.cos(math.pi * frac)) / 2
    return float(end + (start - end) * cos_val)
