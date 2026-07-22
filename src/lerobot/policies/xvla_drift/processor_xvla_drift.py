from typing import Any

import torch

from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_xvla_drift import XVLADriftConfig


def make_xvla_drift_pre_post_processors(
    config: XVLADriftConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_xvla_pre_post_processors(config, dataset_stats)
