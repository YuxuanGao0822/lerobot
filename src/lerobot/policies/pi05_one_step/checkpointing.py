#!/usr/bin/env python

# Copyright 2026 DriftingVLA contributors.
# Licensed under the Apache License, Version 2.0.

"""Pure checkpoint-key provenance helpers (no tensor/model loading)."""

from collections.abc import Mapping
from typing import TypeVar


Value = TypeVar("Value")

FRESH_EXPERT_PREFIXES = (
    "model.paligemma_with_expert.gemma_expert.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.time_mlp_in.",
    "model.time_mlp_out.",
)


def method_new_prefixes(method: str) -> tuple[str, ...]:
    return ()


def skipped_checkpoint_prefixes(method: str, fresh_action_expert: bool) -> tuple[str, ...]:
    prefixes = method_new_prefixes(method)
    if fresh_action_expert:
        prefixes += FRESH_EXPERT_PREFIXES
    return prefixes


def filter_pretrained_state(
    state: Mapping[str, Value],
    *,
    method: str,
    fresh_action_expert: bool,
) -> tuple[dict[str, Value], tuple[str, ...]]:
    prefixes = skipped_checkpoint_prefixes(method, fresh_action_expert)
    return ({key: value for key, value in state.items() if not key.startswith(prefixes)}, prefixes)
