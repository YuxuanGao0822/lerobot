from __future__ import annotations

import torch
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import XVLAModel, XVLAPolicy
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from .configuration_xvla_drift import XVLADriftConfig
from .drifting_util import drift_loss_grouped
from .keystone_util import cluster_medoid_select


class XVLADriftModel(XVLAModel):
    """X-VLA action transformer trained as a one-step clean-action generator."""

    def __init__(self, config, florence_config, proprio_dim):
        super().__init__(config, florence_config, proprio_dim)
        self.last_test_time_info = None

    def _sample_drift_candidates(
        self,
        input_ids,
        image_input,
        image_mask,
        domain_id,
        proprio,
        samples,
        noise=None,
    ):
        target_dtype = self._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)
        encoded = self.forward_vlm(input_ids, image_input, image_mask)
        batch_size = input_ids.shape[0]

        expanded_encoded = {
            key: value.repeat_interleave(samples, dim=0) for key, value in encoded.items()
        }
        expanded_domain = domain_id.repeat_interleave(samples, dim=0)
        expanded_proprio = proprio.repeat_interleave(samples, dim=0)
        if noise is None:
            source = torch.randn(
                batch_size * samples,
                self.chunk_size,
                self.dim_action,
                device=proprio.device,
                dtype=target_dtype,
            )
        else:
            if samples != 1:
                raise ValueError("Explicit X-VLA-Drift noise is only supported for one candidate.")
            source = noise.to(device=proprio.device, dtype=target_dtype)
        fixed_time = torch.ones(batch_size * samples, device=proprio.device, dtype=target_dtype)
        proprio_model, source_model = self.action_space.preprocess(expanded_proprio, source)
        generated = self.transformer(
            domain_id=expanded_domain,
            action_with_noise=source_model,
            t=fixed_time,
            proprio=proprio_model,
            **expanded_encoded,
        )
        generated = self.action_space.postprocess(generated)
        return generated.view(batch_size, samples, self.chunk_size, self.dim_action)

    def forward_drifting(
        self,
        input_ids,
        image_input,
        image_mask,
        domain_id,
        proprio,
        action,
        actions_is_pad=None,
        reduction="mean",
    ):
        target_dtype = self._get_target_dtype()
        action = action.to(dtype=target_dtype)
        batch_size = action.shape[0]
        generated = self._sample_drift_candidates(
            input_ids,
            image_input,
            image_mask,
            domain_id,
            proprio,
            self.config.drifting_gen_per_label,
        )
        valid = (
            torch.ones(batch_size, self.chunk_size, dtype=torch.bool, device=action.device)
            if actions_is_pad is None
            else ~actions_is_pad
        )
        if self.config.drifting_perdim_loss:
            mask = valid.float()
            gen_grouped = (generated * mask[:, None, :, None]).permute(3, 0, 1, 2)
            data_grouped = (action * mask[:, :, None]).permute(2, 0, 1).unsqueeze(2)
            valid_grouped = valid.any(dim=1).unsqueeze(0).expand(self.dim_action, batch_size)
        elif self.config.drifting_per_timestep_loss:
            gen_grouped = generated.permute(2, 0, 1, 3)
            data_grouped = action.transpose(0, 1).unsqueeze(2)
            valid_grouped = valid.transpose(0, 1)
        else:
            mask = valid.float()
            gen_grouped = (generated * mask[:, None, :, None]).flatten(2).unsqueeze(0)
            data_grouped = (action * mask[:, :, None]).flatten(1)[None, :, None, :]
            valid_grouped = valid.any(dim=1).unsqueeze(0)
        per_unit, drift_info = drift_loss_grouped(
            gen_grouped, data_grouped, valid_grouped, self.config.drifting_temperatures
        )
        mean_loss = per_unit.sum() / valid_grouped.sum().clamp(min=1)
        per_sample = per_unit.sum(dim=0) / valid_grouped.sum(dim=0).clamp(min=1)
        with torch.no_grad():
            used = valid_grouped.any(dim=1).to(per_unit.dtype)
            count = used.sum().clamp(min=1)
            names = ["scale"] + [f"loss_{value}" for value in self.config.drifting_temperatures]
            values = [(drift_info[name] * used).sum() / count for name in names]
            valid_mask = valid.unsqueeze(-1)
            denom = (valid.sum() * self.dim_action).clamp(min=1)
            values.extend(
                [
                    ((generated.mean(dim=1) - action).square() * valid_mask).sum() / denom,
                    (generated.var(dim=1, unbiased=False) * valid_mask).sum() / denom,
                ]
            )
            names.extend(["centroid_mse", "sample_variance"])
            info = dict(zip(names, torch.stack(values).tolist(), strict=True))
        return (per_sample if reduction == "none" else mean_loss), info

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids,
        image_input,
        image_mask,
        domain_id,
        proprio,
        steps,
    ):
        del steps
        self.eval()
        samples = self.config.test_time_samples
        candidates = self._sample_drift_candidates(
            input_ids, image_input, image_mask, domain_id, proprio, samples
        )
        if samples == 1:
            return candidates[:, 0]
        indices, info = cluster_medoid_select(
            candidates.flatten(2),
            num_clusters=self.config.test_time_clusters,
            unimodal_tau=self.config.test_time_unimodal_tau,
        )
        self.last_test_time_info = info
        return candidates[torch.arange(candidates.shape[0], device=candidates.device), indices]


class XVLADriftPolicy(XVLAPolicy):
    config_class = XVLADriftConfig
    name = "xvla_drift"

    def __init__(self, config: XVLADriftConfig, **kwargs):
        require_package("transformers", extra="xvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        florence_config = config.get_florence_config()
        proprio_dim = config.max_state_dim if config.use_proprio else 0
        self.model = XVLADriftModel(config, florence_config, proprio_dim)
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ):
        """Seed Florence architecture metadata from a released X-VLA checkpoint.

        LeRobot's training factory intentionally passes the new Drift config to
        preserve dataset-derived input/output features. Released X-VLA configs,
        however, are the only source of the large nested Florence config. Copy
        that architecture payload before model construction while keeping the
        requested ``xvla_drift`` type and all post-training feature overrides.
        """
        config_kwargs = {
            key: kwargs[key]
            for key in (
                "force_download",
                "resume_download",
                "proxies",
                "token",
                "cache_dir",
                "local_files_only",
                "revision",
            )
            if key in kwargs
        }
        if config is None:
            base_config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **config_kwargs)
            if not isinstance(base_config, XVLAConfig):
                raise ValueError("X-VLA-Drift must be initialized from an X-VLA checkpoint.")
            config_payload = base_config.to_dict()
            config_payload.pop("type", None)
            config = XVLADriftConfig(**config_payload)
        elif isinstance(config, XVLADriftConfig) and not config.florence_config:
            base_config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **config_kwargs)
            if not isinstance(base_config, XVLAConfig):
                raise ValueError("X-VLA-Drift must be initialized from an X-VLA checkpoint.")
            config.florence_config = dict(base_config.florence_config)
            config._florence_config_obj = None
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            **kwargs,
        )

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        inputs = self._build_model_inputs(batch)
        targets = self._prepare_action_targets(batch)
        loss, drift_info = self.model.forward_drifting(
            action=targets,
            actions_is_pad=batch.get("action_is_pad"),
            reduction=reduction,
            **inputs,
        )
        log_dict = {f"drift_{key}": value for key, value in drift_info.items()}
        log_dict["loss"] = loss.mean().item() if reduction == "none" else loss.item()
        return loss, log_dict
