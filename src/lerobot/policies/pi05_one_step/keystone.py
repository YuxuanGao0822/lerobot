#!/usr/bin/env python

# Copyright 2026 Xingdong Zuo. All rights reserved.
# Copyright 2026 DriftingVLA contributors.
# Licensed under the Apache License, Version 2.0.

"""Optional KeyStone-style selection over sampled action chunks.

This is a modularized adaptation of the selector in the authoritative
``lerobot_policy_pi05_drift`` reference.  The caller must remove padded action
dimensions before flattening candidates; this module deliberately knows
nothing about LeRobot feature metadata.
"""

import torch
from torch import Tensor


_EPS = 1e-8


@torch.no_grad()
def cluster_medoid_select(
    candidates: Tensor,
    num_clusters: int = 2,
    unimodal_tau: float = 0.3,
) -> tuple[Tensor, dict[str, list]]:
    """Select one candidate index per observation from ``[B,K,F]`` tensors."""
    if candidates.ndim != 3:
        raise ValueError(f"candidates must be [B,K,F], got {tuple(candidates.shape)}")
    batch_size, samples, features = candidates.shape
    if samples < 1 or features < 1:
        raise ValueError("candidates must contain at least one sample and one feature")
    if num_clusters < 1:
        raise ValueError("num_clusters must be positive")
    if unimodal_tau <= 0:
        raise ValueError("unimodal_tau must be positive")

    device = candidates.device
    if samples == 1:
        indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        return indices, {
            "spread": [0.0] * batch_size,
            "unimodal": [True] * batch_size,
            "cluster_size": [1] * batch_size,
        }

    values = candidates.float()
    distances = torch.cdist(values, values)
    upper_i, upper_j = torch.triu_indices(samples, samples, offset=1, device=device)
    indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    spread_out: list[float] = []
    unimodal_out: list[bool] = []
    size_out: list[int] = []

    for batch_index in range(batch_size):
        distance = distances[batch_index]
        global_medoid = int(distance.sum(dim=1).argmin())
        median_pairwise = distance[upper_i, upper_j].median()
        spread = float(
            (values[batch_index].mean(dim=0) - values[batch_index, global_medoid]).norm()
            / (median_pairwise + _EPS)
        )
        spread_out.append(spread)
        if spread < unimodal_tau:
            indices[batch_index] = global_medoid
            unimodal_out.append(True)
            size_out.append(samples)
            continue

        unimodal_out.append(False)
        assignment = _kmeans_assign(values[batch_index], min(num_clusters, samples))
        counts = torch.bincount(assignment, minlength=int(assignment.max()) + 1)
        members = (assignment == int(counts.argmax())).nonzero(as_tuple=True)[0]
        within = distance[members][:, members].sum(dim=1)
        indices[batch_index] = members[int(within.argmin())]
        size_out.append(int(counts.max()))

    return indices, {
        "spread": spread_out,
        "unimodal": unimodal_out,
        "cluster_size": size_out,
    }


def _kmeans_assign(values: Tensor, num_clusters: int) -> Tensor:
    """Deterministic small-batch Lloyd iterations over ``[K,F]`` candidates."""
    centroids = values[:num_clusters].clone()
    assignment: Tensor | None = None
    for _ in range(10):
        updated = torch.cdist(values, centroids).argmin(dim=1)
        if assignment is not None and torch.equal(updated, assignment):
            break
        assignment = updated
        for cluster in range(num_clusters):
            member_mask = assignment == cluster
            if member_mask.any():
                centroids[cluster] = values[member_mask].mean(dim=0)
    return torch.cdist(values, centroids).argmin(dim=1)
