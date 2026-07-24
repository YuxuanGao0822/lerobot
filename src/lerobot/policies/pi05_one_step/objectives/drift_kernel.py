#!/usr/bin/env python

# Copyright 2026 Xingdong Zuo.
# Copyright 2026 DriftingVLA contributors.
# Licensed under the Apache License, Version 2.0.
#
# This module is a shared, reorganized copy of the Apache-2.0
# `lerobot_policy_pi05_drift` kernel. See docs/driftingvla/THIRD_PARTY_NOTICES.md.

"""One canonical grouped drifting kernel for Chunk, Step-wise, and Per-Dim."""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def masked_mse(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Per-example MSE over valid action timesteps only."""
    mask = valid.to(dtype=prediction.dtype)[:, :, None]
    squared = (prediction - target).square() * mask
    denominator = (mask.sum(dim=(1, 2)) * prediction.shape[-1]).clamp(min=1.0)
    return squared.sum(dim=(1, 2)) / denominator


def validate_temperatures(temperatures) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in temperatures)
    except TypeError as exc:
        raise ValueError("temperatures must be a non-empty iterable of positive floats") from exc
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"temperatures must be finite and positive; got {values}")
    return values


def pairwise_l2(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Dot-product L2 matching the DBP reference instead of `torch.cdist`."""
    xy = torch.einsum("...nd,...md->...nm", x, y)
    xx = torch.einsum("...nd,...nd->...n", x, x)
    yy = torch.einsum("...md,...md->...m", y, y)
    squared = xx.unsqueeze(-1) + yy.unsqueeze(-2) - 2 * xy
    return torch.sqrt(torch.clamp(squared, min=eps))


def _drift_goal(
    generated: Tensor,
    positives: Tensor,
    valid: Tensor,
    feature_valid: Tensor,
    temperatures: tuple[float, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    siblings, feature_dim = generated.shape[2], generated.shape[3]
    feature_valid = feature_valid.to(dtype=generated.dtype) * valid[:, :, None].to(generated.dtype)
    generated = generated * feature_valid[:, :, None, :]
    positives = positives * feature_valid[:, :, None, :]
    candidates = torch.cat([generated, positives], dim=2)
    candidate_count = candidates.shape[2]
    valid_unit = valid[:, :, None, None]

    distance = pairwise_l2(generated, candidates)
    valid_feature_count = feature_valid.sum(dim=-1)
    pair_count = valid.sum(dim=1) * (siblings * candidate_count)
    scale = (distance * valid_unit).sum(dim=(1, 2, 3)) / pair_count.clamp(min=1.0)
    scale = torch.where(pair_count > 0, scale, torch.ones_like(scale))
    effective_feature_dim = (
        valid_feature_count * valid.to(generated.dtype)
    ).sum(dim=1) / valid.sum(dim=1).clamp(min=1).to(generated.dtype)
    effective_feature_dim = torch.where(
        valid.sum(dim=1) > 0,
        effective_feature_dim,
        generated.new_full(effective_feature_dim.shape, float(feature_dim)),
    )
    input_scale = torch.clamp(scale / effective_feature_dim.sqrt(), min=1e-3)

    generated_scaled = generated / input_scale[:, None, None, None]
    candidates_scaled = candidates / input_scale[:, None, None, None]
    normalized_distance = distance / scale.clamp(min=1e-3)[:, None, None, None]

    self_mask = F.pad(
        torch.eye(siblings, device=distance.device, dtype=distance.dtype),
        (0, candidate_count - siblings),
    )
    normalized_distance = normalized_distance + self_mask * 100.0

    temperature_view = normalized_distance.new_tensor(temperatures).view(-1, 1, 1, 1, 1)
    logits = -normalized_distance.unsqueeze(0) / temperature_view
    affinity = torch.sqrt(
        torch.clamp(logits.softmax(dim=-1) * logits.softmax(dim=-2), min=1e-6)
    )
    affinity_sibling, affinity_positive = affinity[..., :siblings], affinity[..., siblings:]
    coefficient = torch.cat(
        [
            -affinity_sibling * affinity_positive.sum(dim=-1, keepdim=True),
            affinity_positive * affinity_sibling.sum(dim=-1, keepdim=True),
        ],
        dim=-1,
    )

    force = torch.einsum("kqbgy,qbys->kqbgs", coefficient, candidates_scaled)
    force = force - coefficient.sum(dim=-1, keepdim=True) * generated_scaled
    feature_mask = feature_valid[:, :, None, :]
    unit_count = valid_feature_count.sum(dim=1) * siblings
    energy = (force.square() * feature_mask).sum(dim=(2, 3, 4)) / unit_count.clamp(min=1.0)
    force_rms = torch.sqrt(torch.clamp(energy, min=1e-8))
    drift = (force / force_rms[:, :, None, None, None]).sum(dim=0)
    return generated_scaled + drift, input_scale, scale, energy


def drift_loss_grouped(
    generated: Tensor,
    positives: Tensor,
    valid: Tensor | None = None,
    temperatures=(0.02, 0.05, 0.2),
    *,
    feature_valid: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute DBP loss for `[Q,B,G,S]` groups in at least fp32.

    Invalid `[Q,B]` units are absent from group statistics and have exactly
    zero loss and zero generated-sample gradient. ``feature_valid`` removes
    individual coordinates from pairwise distances, scale estimation, force
    normalization, and the final residual; it is not merely an input-zeroing
    convenience.
    """
    temperatures = validate_temperatures(temperatures)
    if generated.ndim != 4:
        raise ValueError(f"generated must have shape [Q,B,G,S], got {tuple(generated.shape)}")
    if positives.ndim != 4:
        raise ValueError(f"positives must have shape [Q,B,P,S], got {tuple(positives.shape)}")
    if positives.shape[:2] != generated.shape[:2] or positives.shape[-1] != generated.shape[-1]:
        raise ValueError(
            "positives must share Q, B, and S with generated; "
            f"got generated={tuple(generated.shape)}, positives={tuple(positives.shape)}"
        )
    if generated.shape[2] < 2:
        raise ValueError("DBP sibling repulsion requires at least two generated samples (G >= 2).")
    if positives.shape[2] < 1:
        raise ValueError("At least one positive demonstration is required (P >= 1).")
    if positives.device != generated.device:
        raise ValueError("generated and positives must be on the same device.")
    if valid is not None and valid.shape != generated.shape[:2]:
        raise ValueError(
            f"valid must have shape [Q,B]={tuple(generated.shape[:2])}, got {tuple(valid.shape)}"
        )

    compute_dtype = torch.float64 if generated.dtype == torch.float64 else torch.float32
    generated = generated.to(compute_dtype)
    positives = positives.to(compute_dtype)
    valid_float = (
        generated.new_ones(generated.shape[:2])
        if valid is None
        else valid.to(device=generated.device, dtype=generated.dtype)
    )
    if feature_valid is None:
        feature_mask = generated.new_ones(generated.shape)
        feature_mask = feature_mask[:, :, 0, :]
    else:
        if feature_valid.shape != generated.shape[:2] + generated.shape[-1:]:
            raise ValueError(
                "feature_valid must have shape [Q,B,S] matching grouped inputs; "
                f"got {tuple(feature_valid.shape)} for {tuple(generated.shape)}"
            )
        feature_mask = feature_valid.to(device=generated.device, dtype=generated.dtype)
    feature_mask = feature_mask * valid_float[:, :, None]

    with torch.no_grad(), torch.autocast(device_type=generated.device.type, enabled=False):
        goal, input_scale, scale, energy = _drift_goal(
            generated.detach(), positives, valid_float, feature_mask, temperatures
        )

    generated_scaled = generated / input_scale[:, None, None, None]
    residual = (generated_scaled - goal) * feature_mask[:, :, None, :]
    denominator = (
        feature_mask.sum(dim=-1) * generated.shape[2]
    ).clamp(min=1.0)
    loss = residual.square().sum(dim=(-1, -2)) / denominator
    info = {"scale": scale}
    for temperature, value in zip(temperatures, energy, strict=True):
        info[f"loss_{temperature}"] = value
    return loss, info
