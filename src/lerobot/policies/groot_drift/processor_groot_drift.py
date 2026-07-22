from typing import Any

import torch

from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_groot_drift import GrootDriftConfig


def make_groot_drift_pre_post_processors(
    config: GrootDriftConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_groot_pre_post_processors(config, dataset_stats, dataset_meta)
