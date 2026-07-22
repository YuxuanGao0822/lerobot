from typing import Any

import torch

from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_pi0_drift import PI0DriftConfig


def make_pi0_drift_pre_post_processors(
    config: PI0DriftConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """PI0-Drift is processor-compatible with the released PI0 checkpoints."""
    return make_pi0_pre_post_processors(config, dataset_stats)
