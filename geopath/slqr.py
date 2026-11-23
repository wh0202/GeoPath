# Derived from the GeoPath experiment implementation, built on FT-FSOD.
"""Localization-gap-conditioned stochastic local query regularization."""

import math

import torch
from torch.nn import functional as F


SLQR_MODES = ('adaptive_local', 'resampled_local')


def effective_weight(iteration, total_iterations, weight, warmup_ratio):
    warmup = math.ceil(total_iterations * warmup_ratio)
    return weight * min(1, max(0, iteration / warmup)) if warmup else weight


def sample_directions(quality_gaps, *, seed, gap_scale=0.30,
                      mode='adaptive_local', flip_probability=0.50):
    """Sample detached directions with HED's private CPU RNG.

    Adaptive SLQR uses rho=(1-clamp(gap/gamma, 0, 1))/2. The legacy mode
    retains fixed-probability sampling for reproducibility of older runs.
    """
    if mode not in SLQR_MODES:
        raise ValueError(f'Unknown SLQR mode: {mode}')
    if not math.isfinite(gap_scale) or gap_scale <= 0:
        raise ValueError('SLQR gap_scale must be finite and positive.')
    if mode == 'adaptive_local':
        normalized = (quality_gaps.detach() / gap_scale).clamp(0, 1)
        probabilities = ((1 - normalized) / 2).to(device='cpu', dtype=torch.float32)
    else:
        if not 0 <= flip_probability <= 1:
            raise ValueError('SLQR flip_probability must be in [0, 1].')
        probabilities = flip_probability
    generator = torch.Generator(device='cpu').manual_seed(seed % (2**63 - 1))
    flipped = torch.rand(quality_gaps.numel(), generator=generator) < probabilities
    return torch.where(flipped, -1, 1).to(quality_gaps)


def local_pair_loss(probabilities, ious, *, seed, candidate_iou=0.30,
                    top_k=5, min_gap=0.10, margin=0.20, gap_scale=0.30,
                    mode='adaptive_local', flip_probability=0.50, eps=1e-6):
    """Return a loss sum and normalizer for batch-wide pair averaging.

    In adaptive_local every valid pair has unit weight. resampled_local
    preserves the previous fixed-direction-probability, IoU-weighted loss.
    """
    if mode not in SLQR_MODES:
        raise ValueError(f'Unknown SLQR mode: {mode}')
    scores = (probabilities + eps).log() - (1 - probabilities + eps).log()
    zero = scores.sum() * 0
    if not ious.numel():
        return zero, zero.detach()

    qualities, owners = ious.detach().max(dim=1)
    anchors, candidates, targets = [], [], []
    for gt in range(ious.shape[1]):
        indices = torch.where((owners == gt) & (qualities >= candidate_iou))[0]
        order = qualities[indices].argsort(descending=True, stable=True)
        indices = indices[order[:top_k]]
        if indices.numel() < 2:
            continue
        anchor, others = indices[0], indices[1:]
        others = others[qualities[anchor] - qualities[others] > min_gap]
        if others.numel():
            anchors.append(anchor.expand_as(others))
            candidates.append(others)
            targets.append(torch.full_like(others, gt))
    if not anchors:
        return zero, zero.detach()

    anchor, candidate, target = map(torch.cat, (anchors, candidates, targets))
    quality_gaps = qualities[anchor] - qualities[candidate]
    direction = sample_directions(
        quality_gaps, seed=seed, gap_scale=gap_scale, mode=mode,
        flip_probability=flip_probability).to(scores)
    score_gap = scores[anchor, target] - scores[candidate, target]
    weights = (torch.ones_like(quality_gaps) if mode == 'adaptive_local'
               else (quality_gaps / gap_scale).clamp(max=1))
    return (weights * F.softplus(margin - direction * score_gap)).sum(), weights.sum()
