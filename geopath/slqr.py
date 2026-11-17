# Derived from the GeoPath experiment implementation, built on FT-FSOD.
"""Local resampled stochastic query regularization."""

import math

import torch
from torch.nn import functional as F


def effective_weight(iteration, total_iterations, weight, warmup_ratio):
    warmup = math.ceil(total_iterations * warmup_ratio)
    return weight * min(1, max(0, iteration / warmup)) if warmup else weight


def local_pair_loss(probabilities, ious, *, seed, candidate_iou=0.30,
                    top_k=5, min_gap=0.10, margin=0.20, gap_scale=0.30,
                    flip_probability=0.50, eps=1e-6):
    """Return the weighted loss sum and weight sum for one image."""
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
    generator = torch.Generator(device='cpu').manual_seed(seed % (2**63 - 1))
    flipped = torch.rand(anchor.numel(), generator=generator) < flip_probability
    direction = torch.where(flipped, -1, 1).to(scores)
    score_gap = scores[anchor, target] - scores[candidate, target]
    weights = ((qualities[anchor] - qualities[candidate]) / gap_scale).clamp(max=1)
    return (weights * F.softplus(margin - direction * score_gap)).sum(), weights.sum()
