#!/usr/bin/env python

# Copyright 2026 DriftingVLA contributors.
# Licensed under the Apache License, Version 2.0.

"""DBP grouping transforms; the drifting kernel is invariant across modes."""

from typing import Literal

from torch import Tensor


Grouping = Literal["chunk", "stepwise", "perdim"]


def group_drifting_tensors(
    generated: Tensor,
    demonstration: Tensor,
    valid_timestep: Tensor,
    grouping: Grouping,
    *,
    return_feature_mask: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
    """Map `[B,G,H,D]`, `[B,H,D]`, `[B,H]` to grouped kernel inputs.

    Returns generated `[Q,B,G,S]`, positives `[Q,B,1,S]`, and valid
    groups `[Q,B]`. When ``return_feature_mask`` is true, a fourth tensor
    `[Q,B,S]` identifies valid feature coordinates inside each group. This
    distinction matters for Chunk and Per-Dim groups with a partially padded
    action horizon: zeroing a coordinate is not the same as removing it from
    distance and force statistics.
    """
    if generated.ndim != 4 or demonstration.ndim != 3 or valid_timestep.ndim != 2:
        raise ValueError("Expected generated [B,G,H,D], demonstration [B,H,D], valid [B,H].")
    bsize, _, horizon, action_dim = generated.shape
    if demonstration.shape != (bsize, horizon, action_dim):
        raise ValueError("Demonstration shape must match generated B,H,D dimensions.")
    if valid_timestep.shape != (bsize, horizon):
        raise ValueError("valid_timestep must have shape [B,H].")

    if grouping == "stepwise":
        result = (
            generated.permute(2, 0, 1, 3),
            demonstration.transpose(0, 1).unsqueeze(2),
            valid_timestep.transpose(0, 1),
        )
        if return_feature_mask:
            feature_mask = valid_timestep.transpose(0, 1).unsqueeze(-1).expand(-1, -1, action_dim)
            return (*result, feature_mask)
        return result

    mask = valid_timestep.to(generated.dtype)
    generated = generated * mask[:, None, :, None]
    demonstration = demonstration * mask[:, :, None]
    valid_observation = valid_timestep.any(dim=1)

    if grouping == "chunk":
        result = (
            generated.flatten(2).unsqueeze(0),
            demonstration.flatten(1)[None, :, None, :],
            valid_observation.unsqueeze(0),
        )
        if return_feature_mask:
            feature_mask = (
                valid_timestep[:, :, None]
                .expand(-1, -1, action_dim)
                .reshape(bsize, horizon * action_dim)
                .unsqueeze(0)
            )
            return (*result, feature_mask)
        return result
    if grouping == "perdim":
        result = (
            generated.permute(3, 0, 1, 2),
            demonstration.permute(2, 0, 1).unsqueeze(2),
            valid_observation.unsqueeze(0).expand(action_dim, bsize),
        )
        if return_feature_mask:
            feature_mask = valid_timestep.transpose(0, 1).unsqueeze(0).expand(action_dim, -1, -1)
            return (*result, feature_mask)
        return result
    raise ValueError(f"Unknown drifting grouping: {grouping!r}")
