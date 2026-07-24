#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Copyright 2026 DriftingVLA contributors.
# Licensed under the Apache License, Version 2.0.

"""Configuration for architecture-controlled π0.5 one-step objectives.

Stock π0.5 flow and its naive one-Euler-step evaluation intentionally remain
``policy.type=pi05``. This config covers only methods that change training.
"""

import math
from dataclasses import dataclass
from typing import Literal

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


OneStepMethod = Literal[
    "dbp_chunk",
    "dbp_stepwise",
    "drifting_perdim",
]


@PreTrainedConfig.register_subclass("pi05_one_step")
@dataclass
class PI05OneStepConfig(PI05Config):
    """A shared π0.5 architecture with an explicit objective selector."""

    method: OneStepMethod = "drifting_perdim"
    objective: str = "dbp"
    inference_nfe: int = 1

    # Checkpoint provenance. ``fresh_action_expert`` is consumed after the
    # partial load; ``init_label`` remains serialized in trained checkpoints.
    fresh_action_expert: bool = False
    action_expert_init: Literal["pretrained", "fresh"] | None = None
    init_label: str | None = None
    teacher_checkpoint: str | None = None
    train_full_model: bool = True

    # Shared DBP kernel.
    drifting_gen_per_label: int = 8
    drifting_temperatures: tuple[float, ...] = (0.02, 0.05, 0.2)
    drifting_grouping: Literal["chunk", "stepwise", "perdim"] = "perdim"
    mask_padded_timesteps: bool = True

    # Optional KeyStone-style test-time selection. K=1 is the formal main
    # protocol; K>1 is an inference-only compute/robustness ablation.
    test_time_samples: int = 1
    test_time_clusters: int = 2
    test_time_unimodal_tau: float = 0.3

    def __post_init__(self):
        super().__post_init__()
        supported = {"dbp_chunk", "dbp_stepwise", "drifting_perdim"}
        if self.method not in supported:
            raise ValueError(f"Unsupported one-step method {self.method!r}; expected one of {sorted(supported)}")
        if self.inference_nfe != 1:
            raise ValueError("All pi05_one_step methods have inference_nfe=1 by definition.")
        if self.rtc_config is not None:
            raise ValueError("RTC requires an iterative flow trajectory and is incompatible with pi05_one_step.")
        if self.test_time_samples < 1:
            raise ValueError("test_time_samples must be >= 1.")
        if self.test_time_clusters < 1:
            raise ValueError("test_time_clusters must be >= 1.")
        if not math.isfinite(self.test_time_unimodal_tau) or self.test_time_unimodal_tau <= 0:
            raise ValueError("test_time_unimodal_tau must be finite and positive.")
        if self.test_time_samples > 1 and self.test_time_clusters < 2:
            raise ValueError("test_time_clusters must be >= 2 when test_time_samples > 1.")

        if self.action_expert_init is None:
            self.action_expert_init = "fresh" if self.fresh_action_expert else "pretrained"
        elif self.fresh_action_expert and self.action_expert_init != "fresh":
            raise ValueError(
                "fresh_action_expert=True conflicts with "
                f"action_expert_init={self.action_expert_init!r}."
            )
        elif (
            self.action_expert_init == "fresh"
            and not self.fresh_action_expert
            and not self.init_label
        ):
            raise ValueError(
                "action_expert_init='fresh' with fresh_action_expert=False is valid only "
                "for an already initialized checkpoint carrying init_label provenance."
            )

        if self.method.startswith("dbp_") or self.method == "drifting_perdim":
            expected = {
                "dbp_chunk": "chunk",
                "dbp_stepwise": "stepwise",
                "drifting_perdim": "perdim",
            }[self.method]
            if self.drifting_grouping != expected:
                raise ValueError(
                    f"method={self.method!r} requires drifting_grouping={expected!r}; "
                    f"got {self.drifting_grouping!r}."
                )
            if self.drifting_gen_per_label < 2:
                raise ValueError("DBP objectives require drifting_gen_per_label >= 2.")
            if not self.drifting_temperatures:
                raise ValueError("drifting_temperatures must be non-empty.")
            converted = tuple(float(value) for value in self.drifting_temperatures)
            if any(not math.isfinite(value) or value <= 0 for value in converted):
                raise ValueError("drifting_temperatures must contain finite positive values.")
            self.drifting_temperatures = converted

        method_objectives = {
            "dbp_chunk": "dbp",
            "dbp_stepwise": "dbp",
            "drifting_perdim": "dbp",
        }
        self.objective = method_objectives[self.method]
        self.train_full_model = not self.train_expert_only
