# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from lerobot.policies.rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla_drift")
@dataclass
class SmolVLADriftConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    # Drifting loss objective. False keeps the standard flow-matching path.
    use_drifting_loss: bool = True
    # Number of generated action samples per observation. Must be >= 2 so samples
    # can repel each other.
    drifting_gen_per_label: int = 8
    # Kernel temperatures used by the canonical Drift field.
    drifting_temperatures: tuple[float, ...] = (0.02, 0.05, 0.2)
    # True: one drift_loss call per chunk timestep on [n_valid, G, real_dim]
    # slices, each with its own data-dependent scale (DBPO `per_timestep_loss`).
    # False (default): a single call on the flattened chunk
    # [B, G, chunk*real_dim] with one global scale per observation;
    # padded (b, t) rows are zero-masked in both gen and pos.
    drifting_per_timestep_loss: bool = False
    # Per-action-dim matching: groups = the real action dims; each dim's
    # time-series over the chunk (padded steps zero-masked, like flat-chunk) is
    # matched across the G samples. Mutually exclusive with per-timestep.
    drifting_perdim_loss: bool = True

    # KeyStone test-time self-consistency selection (drift path only). With
    # `test_time_samples` K > 1, inference draws K one-step candidate chunks and
    # returns the guarded cluster-medoid (see `keystone_util.py`). K=1 = off.
    test_time_samples: int = 1
    test_time_clusters: int = 2
    test_time_unimodal_tau: float = 0.3

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.use_drifting_loss:
            if self.drifting_gen_per_label < 2:
                raise ValueError(
                    "`drifting_gen_per_label` (G) must be >= 2; G=1 degenerates the drift objective "
                    "(no sibling samples to repel from)."
                )
            try:
                drifting_temperatures = tuple(float(t) for t in self.drifting_temperatures)
            except TypeError as exc:
                raise ValueError("`drifting_temperatures` must be a non-empty iterable of floats.") from exc
            if len(drifting_temperatures) == 0:
                raise ValueError("`drifting_temperatures` must contain at least one positive value.")
            for temperature in drifting_temperatures:
                if not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError(
                        "`drifting_temperatures` must contain only finite positive values; "
                        f"got {self.drifting_temperatures}."
                    )
            self.drifting_temperatures = drifting_temperatures
            if self.drifting_perdim_loss and self.drifting_per_timestep_loss:
                raise ValueError(
                    "`drifting_perdim_loss=True` is mutually exclusive with "
                    "`drifting_per_timestep_loss=True`; set per-timestep loss to False."
                )
            if not self.use_cache:
                raise ValueError(
                    "`use_cache=False` is unsupported with `use_drifting_loss=True`: the Drift generator "
                    "builds one observation prefix KV cache and expands it across generated samples."
                )
            if self.rtc_config is not None:
                raise ValueError(
                    "`rtc_config` is incompatible with `use_drifting_loss=True`: Real-Time Chunking "
                    "hooks into the multi-step flow-matching integrator, and the Drift sampler is a "
                    "single forward pass with no integrator to hook into."
                )
            # NOTE: `num_steps` is a flow-matching integrator setting and is never
            # read on the drift path -- `sample_actions` dispatches to the one-step
            # drift sampler before the Euler loop. It is deliberately left untouched.

        # KeyStone test-time selection validation.
        if self.test_time_samples < 1:
            raise ValueError(f"`test_time_samples` must be >= 1, got {self.test_time_samples}.")
        if self.test_time_samples > 1:
            if not self.use_drifting_loss:
                raise ValueError(
                    "`test_time_samples` > 1 (KeyStone) requires `use_drifting_loss=True`: candidate "
                    "selection hooks into the one-step drift sampler, not the flow-matching integrator."
                )
            if self.rtc_config is not None:
                raise ValueError(
                    "`test_time_samples` > 1 is incompatible with `rtc_config`: RTC guides the "
                    "denoising trajectory, KeyStone selects among independent one-step candidates."
                )
            if self.test_time_clusters < 2:
                raise ValueError(
                    f"`test_time_clusters` must be >= 2 when selection is on, got {self.test_time_clusters}."
                )
            if not self.test_time_unimodal_tau > 0:
                raise ValueError(
                    f"`test_time_unimodal_tau` must be > 0, got {self.test_time_unimodal_tau}."
                )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
