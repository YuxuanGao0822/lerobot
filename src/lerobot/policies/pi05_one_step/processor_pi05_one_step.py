#!/usr/bin/env python

# Copyright 2026 DriftingVLA contributors.
# Licensed under the Apache License, Version 2.0.

"""The one-step policies deliberately reuse the stock π0.5 processor."""

from typing import Any

from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

from .configuration_pi05_one_step import PI05OneStepConfig


def make_pi05_one_step_pre_post_processors(
    config: PI05OneStepConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
):
    return make_pi05_pre_post_processors(config, dataset_stats)

