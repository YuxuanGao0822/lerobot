#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# Copyright 2026 DriftingVLA contributors.
# Licensed under the Apache License, Version 2.0.

"""Unified π0.5 implementation for one-step training objectives.

This module inherits the stock π0.5 backbone and adds only interval
conditioning, objective construction, direct/one-step samplers, and strict
checkpoint provenance. It does not modify the stock π0.5 implementation or
the compatibility `pi05_drift` policy.

STATICALLY IMPLEMENTED. REMOTE EXECUTION REQUIRED.
"""

from __future__ import annotations

import builtins
import logging
from pathlib import Path
from typing import Unpack

import torch
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.common.vla_utils import (
    clone_past_key_values,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)
from lerobot.policies.pi05.modeling_pi05 import ActionSelectKwargs, PI05Policy, PI05Pytorch
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _transformers_available, require_package

if _transformers_available:
    from transformers.cache_utils import DynamicCache
else:
    DynamicCache = None

from .configuration_pi05_one_step import PI05OneStepConfig
from .checkpointing import filter_pretrained_state
from .keystone import cluster_medoid_select
from .objectives import drift_loss_grouped, group_drifting_tensors, masked_mse


_DRIFT_METHODS = {"dbp_chunk", "dbp_stepwise", "drifting_perdim"}


class PI05OneStepPytorch(PI05Pytorch):
    """Stock π0.5 action expert with modular one-step objectives."""

    def __init__(self, config: PI05OneStepConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        self.last_test_time_info: dict[str, list] | None = None

    def embed_suffix_interval(self, actions: Tensor, time: Tensor, target_time: Tensor | None):
        """Reuse π0.5 suffix tokens and add method-specific interval conditioning."""
        if target_time is not None:
            raise ValueError("target_time conditioning is not part of the DBP/DriftingVLA objectives.")
        suffix, pad, attention, condition = super().embed_suffix(actions, time)
        return suffix, pad, attention, condition

    def build_prefix_cache(self, images, image_masks, tokens, token_masks):
        """Encode the VLM prefix once while preserving optional full-model gradients."""
        prefix, prefix_pad, prefix_attention = self.embed_prefix(
            images, image_masks, tokens, token_masks
        )
        attention_2d = make_att_2d_masks(prefix_pad, prefix_attention)
        positions = torch.cumsum(prefix_pad, dim=1) - 1
        attention_4d = prepare_attention_masks_4d(attention_2d)

        language_model = self.paligemma_with_expert.paligemma.model.language_model
        language_model.config._attn_implementation = "eager"  # noqa: SLF001
        checkpointing_was_on = getattr(language_model, "gradient_checkpointing", False)
        if checkpointing_was_on:
            language_model.gradient_checkpointing = False
        try:
            _, cache = self.paligemma_with_expert.forward(
                attention_mask=attention_4d,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[prefix, None],
                use_cache=True,
            )
        finally:
            if checkpointing_was_on:
                language_model.gradient_checkpointing = True
        if cache is None:
            raise RuntimeError("π0.5 prefix prefill returned no KV cache.")
        return prefix_pad, cache

    @staticmethod
    def expand_prefix_cache(prefix_pad: Tensor, cache, repeats: int):
        if repeats == 1:
            return prefix_pad, cache
        if DynamicCache is None:
            require_package("transformers", extra="pi")
        expanded_pad = prefix_pad.repeat_interleave(repeats, dim=0)
        expanded_cache = DynamicCache(
            tuple(
                (
                    keys.repeat_interleave(repeats, dim=0),
                    values.repeat_interleave(repeats, dim=0),
                    sliding_window,
                )
                for keys, values, sliding_window in cache
            )
        )
        return expanded_pad, expanded_cache

    def decode_cached(
        self,
        action_state: Tensor,
        time: Tensor,
        target_time: Tensor | None,
        prefix_pad: Tensor,
        cache,
    ) -> Tensor:
        """Evaluate one action field/direct head against a prefilled VLM cache."""
        suffix, suffix_pad, suffix_attention, condition = self.embed_suffix_interval(
            action_state, time, target_time
        )
        suffix_length = suffix_pad.shape[1]
        batch_size, prefix_length = prefix_pad.shape
        prefix_attention = prefix_pad[:, None, :].expand(batch_size, suffix_length, prefix_length)
        suffix_attention = make_att_2d_masks(suffix_pad, suffix_attention)
        full_attention = torch.cat([prefix_attention, suffix_attention], dim=2)
        positions = prefix_pad.sum(dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1

        expert = self.paligemma_with_expert.gemma_expert.model
        expert.config._attn_implementation = "eager"  # noqa: SLF001
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=prepare_attention_masks_4d(full_attention),
            position_ids=positions,
            past_key_values=clone_past_key_values(cache),
            inputs_embeds=[None, suffix],
            use_cache=False,
            adarms_cond=[None, condition],
        )
        action_tokens = outputs[1][:, -self.config.chunk_size :].float()
        return self.action_out_proj(action_tokens)

    def sample_direct_n(self, prefix_pad: Tensor, cache, repeats: int) -> Tensor:
        """Return direct outputs `[B,G,H,Dmax]` from one expanded expert call."""
        batch_size = prefix_pad.shape[0]
        expanded_pad, expanded_cache = self.expand_prefix_cache(prefix_pad, cache, repeats)
        noise = self.sample_noise(
            (batch_size * repeats, self.config.chunk_size, self.config.max_action_dim),
            prefix_pad.device,
        )
        fixed_time = torch.ones(batch_size * repeats, device=noise.device, dtype=torch.float32)
        output = self.decode_cached(noise, fixed_time, None, expanded_pad, expanded_cache)
        return output.view(batch_size, repeats, self.config.chunk_size, self.config.max_action_dim)

    @torch.no_grad()
    def select_test_time_chunk(self, candidates: Tensor) -> Tensor:
        """Select from ``[B,K,H,Dmax]`` using only real action dimensions."""
        if candidates.ndim != 4:
            raise ValueError(f"candidates must be [B,K,H,Dmax], got {tuple(candidates.shape)}")
        if ACTION not in self.config.output_features:
            raise ValueError("KeyStone requires output_features[ACTION] to identify real action dimensions.")
        real_action_dim = self.config.output_features[ACTION].shape[0]
        if not 0 < real_action_dim <= candidates.shape[-1]:
            raise ValueError(
                f"Invalid real action dimension {real_action_dim} for candidates {tuple(candidates.shape)}"
            )
        flattened = candidates[..., :real_action_dim].flatten(2)
        selected, info = cluster_medoid_select(
            flattened,
            num_clusters=self.config.test_time_clusters,
            unimodal_tau=self.config.test_time_unimodal_tau,
        )
        self.last_test_time_info = info
        batch_indices = torch.arange(candidates.shape[0], device=candidates.device)
        return candidates[batch_indices, selected]

    def forward_drift(
        self,
        images,
        image_masks,
        tokens,
        token_masks,
        actions: Tensor,
        valid: Tensor,
        real_action_dim: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        prefix_pad, cache = self.build_prefix_cache(images, image_masks, tokens, token_masks)
        generated = self.sample_direct_n(prefix_pad, cache, self.config.drifting_gen_per_label)
        generated = generated[..., :real_action_dim]
        demonstration = actions[..., :real_action_dim]
        grouped, positives, valid_group, feature_valid = group_drifting_tensors(
            generated,
            demonstration,
            valid,
            self.config.drifting_grouping,
            return_feature_mask=True,
        )
        per_group, kernel_info = drift_loss_grouped(
            grouped,
            positives,
            valid_group,
            self.config.drifting_temperatures,
            feature_valid=feature_valid,
        )
        per_observation = per_group.sum(dim=0) / valid_group.sum(dim=0).clamp(min=1)
        info: dict[str, Tensor] = {}
        nonempty = valid_group.any(dim=1).to(per_group.dtype)
        denominator = nonempty.sum().clamp(min=1)
        for key, value in kernel_info.items():
            info[key] = (value * nonempty).sum() / denominator
        info["centroid_mse"] = masked_mse(
            generated.mean(dim=1), demonstration, valid
        ).mean()
        info["sample_variance"] = (
            generated.var(dim=1, unbiased=False) * valid[:, :, None]
        ).sum() / (valid.sum() * real_action_dim).clamp(min=1)
        return per_observation, info

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        image_masks,
        tokens,
        token_masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """One inference dispatch; `num_steps` is deliberately ignored."""
        del num_steps, kwargs
        prefix_pad, cache = self.build_prefix_cache(images, image_masks, tokens, token_masks)
        batch_size = tokens.shape[0]
        if noise is None and self.config.test_time_samples > 1:
            candidates = self.sample_direct_n(prefix_pad, cache, self.config.test_time_samples)
            if not torch.isfinite(candidates).all():
                raise FloatingPointError("KeyStone candidate sampling produced non-finite values.")
            return self.select_test_time_chunk(candidates)
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.chunk_size, self.config.max_action_dim),
                tokens.device,
            )
        ones = torch.ones(batch_size, device=noise.device, dtype=torch.float32)

        if self.config.method in _DRIFT_METHODS:
            # Direct action read: fixed time is architectural conditioning only.
            actions = self.decode_cached(noise, ones, None, prefix_pad, cache)
            if not torch.isfinite(actions).all():
                raise FloatingPointError("One-step action sampling produced non-finite values.")
            return actions
        raise RuntimeError(f"Unhandled one-step method {self.config.method!r}")


class PI05OneStepPolicy(PI05Policy):
    """LeRobot policy wrapper for all training-modified one-step baselines."""

    config_class = PI05OneStepConfig
    name = "pi05_one_step"

    def __init__(self, config: PI05OneStepConfig, **kwargs):
        del kwargs
        require_package("transformers", extra="pi")
        # Avoid PI05Policy.__init__, which would instantiate the stock core.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = PI05OneStepPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Strict warm-start/partial-load path with serialized provenance."""
        if config is None:
            raise ValueError(
                "PI05OneStepPolicy requires an explicit PI05OneStepConfig when loading a stock π0.5 checkpoint."
            )
        if not isinstance(config, PI05OneStepConfig):
            raise TypeError(f"Expected PI05OneStepConfig, got {type(config).__name__}.")
        model = cls(config, **kwargs)

        try:
            from safetensors.torch import load_file
            from transformers.utils import cached_file

            resolved = cached_file(
                pretrained_name_or_path,
                "model.safetensors",
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            original = load_file(resolved)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model.safetensors from {pretrained_name_or_path!r}; "
                "refusing to continue with random weights."
            ) from exc

        fixed = model._fix_pytorch_state_dict_keys(original, model.config)
        remapped = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}

        filtered, skipped_prefixes = filter_pretrained_state(
            remapped,
            method=config.method,
            fresh_action_expert=config.fresh_action_expert,
        )
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        invalid_missing = [key for key in missing if not key.startswith(skipped_prefixes)]
        if invalid_missing or unexpected:
            raise RuntimeError(
                "π0.5 one-step checkpoint mismatch: "
                f"unexpected missing={invalid_missing[:5]}, unexpected={list(unexpected)[:5]}."
            )
        if strict and not skipped_prefixes and (missing or unexpected):
            raise RuntimeError(f"Strict checkpoint load failed: missing={missing}, unexpected={unexpected}")

        if config.fresh_action_expert:
            model.config.fresh_action_expert = False
            model.config.action_expert_init = "fresh"
            if not model.config.init_label:
                model.config.init_label = "pi05_vlm_fresh_action_expert"
        elif not model.config.init_label:
            model.config.action_expert_init = "pretrained"
            model.config.init_label = "pi05_pretrained"
        model.config.teacher_checkpoint = None
        logging.info(
            "Loaded π0.5 one-step checkpoint source=%s loaded_keys=%d skipped_prefixes=%s "
            "missing_keys=%d unexpected_keys=%d action_expert_init=%s",
            pretrained_name_or_path,
            len(filtered),
            list(skipped_prefixes),
            len(missing),
            len(unexpected),
            model.config.action_expert_init,
        )
        return model

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        images, image_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        token_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        real_action_dim = self.config.output_features[ACTION].shape[0]
        action_is_pad = batch.get("action_is_pad")
        valid = (
            torch.ones(actions.shape[:2], dtype=torch.bool, device=actions.device)
            if action_is_pad is None or not self.config.mask_padded_timesteps
            else ~action_is_pad.to(device=actions.device)
        )

        if self.config.method in _DRIFT_METHODS:
            per_observation, info = self.model.forward_drift(
                images, image_masks, tokens, token_masks, actions, valid, real_action_dim
            )
        else:
            raise RuntimeError(f"Unhandled one-step method {self.config.method!r}")

        loss = per_observation if reduction == "none" else per_observation.mean()
        loss_dict = {key: value.detach().item() for key, value in info.items()}
        loss_dict["loss"] = per_observation.detach().mean().item()
        loss_dict["method"] = self.config.method
        return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, object]:
        projections = "action_in_proj|action_out_proj|time_mlp_in|time_mlp_out"
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({projections}))"
        return {"target_modules": target_modules, "modules_to_save": []}
