from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION

from .configuration_groot_drift import GrootDriftConfig
from .drifting_util import drift_loss_grouped
from .keystone_util import cluster_medoid_select


class GrootDriftPolicy(GrootPolicy):
    """Independent GR00T N1.7 Drift policy with unchanged checkpoint keys."""

    name = "groot_drift"
    config_class = GrootDriftConfig

    def __init__(self, config: GrootDriftConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.last_test_time_info = None

    def _drift_context(self, inputs):
        backbone_inputs, action_inputs = self._groot_model.prepare_input(inputs)
        backbone_output = self._groot_model.backbone(backbone_inputs)
        head = self._groot_model.action_head
        head.set_frozen_modules_to_eval_mode()
        backbone_output = head.process_backbone_output(backbone_output)
        state = action_inputs.state
        if state.shape[1] != head.config.state_history_length:
            raise ValueError("state history length does not match GR00T N1.7 config.")
        state = state.view(state.shape[0], 1, -1)
        state_features = head.state_encoder(state, action_inputs.embodiment_id)
        if head.training and head.state_dropout_prob > 0:
            drop = torch.rand(state_features.shape[0], device=state_features.device) < head.state_dropout_prob
            state_features = state_features * (1 - drop[:, None, None].to(state_features.dtype))
        return head, backbone_output, action_inputs, state_features

    @staticmethod
    def _repeat_backbone_context(backbone_output, samples):
        return {
            "vl_embeds": backbone_output.backbone_features.repeat_interleave(samples, dim=0),
            "attention_mask": backbone_output.backbone_attention_mask.repeat_interleave(samples, dim=0),
            "image_mask": backbone_output.image_mask.repeat_interleave(samples, dim=0),
        }

    def _sample_drift_candidates(self, head, backbone_output, action_inputs, state_features, samples):
        context = self._repeat_backbone_context(backbone_output, samples)
        batch_size = state_features.shape[0]
        embodiment_id = action_inputs.embodiment_id.repeat_interleave(samples, dim=0)
        expanded_state = state_features.repeat_interleave(samples, dim=0)
        source = torch.randn(
            batch_size * samples,
            head.config.action_horizon,
            head.action_dim,
            device=state_features.device,
            dtype=state_features.dtype,
        )
        # GR00T parameterizes noise at t=0 (opposite to the PI convention).
        # Drift direct-reads the decoder at that single source endpoint.
        timesteps = torch.zeros(batch_size * samples, dtype=torch.long, device=state_features.device)
        action_features = head.action_encoder(source, timesteps, embodiment_id)
        if head.config.add_pos_embed:
            positions = torch.arange(action_features.shape[1], device=state_features.device)
            action_features = action_features + head.position_embedding(positions).unsqueeze(0)
        state_action = torch.cat((expanded_state, action_features), dim=1)
        if head.config.use_alternate_vl_dit:
            model_output, _ = head.model(
                hidden_states=state_action,
                encoder_hidden_states=context["vl_embeds"],
                encoder_attention_mask=context["attention_mask"],
                timestep=timesteps,
                return_all_hidden_states=True,
                image_mask=context["image_mask"],
                backbone_attention_mask=context["attention_mask"],
            )
        else:
            model_output, _ = head.model(
                hidden_states=state_action,
                encoder_hidden_states=context["vl_embeds"],
                encoder_attention_mask=context["attention_mask"],
                timestep=timesteps,
                return_all_hidden_states=True,
            )
        decoded = head.action_decoder(model_output, embodiment_id)
        generated = decoded[:, -head.config.action_horizon :]
        return generated.view(batch_size, samples, head.config.action_horizon, head.action_dim)

    def _reduce_drift(self, generated, demonstrations, valid, real_dim, reduction):
        batch_size = demonstrations.shape[0]
        generated = generated[..., :real_dim]
        demonstrations = demonstrations[..., :real_dim]
        if self.config.drifting_perdim_loss:
            mask = valid.float()
            gen_grouped = (generated * mask[:, None, :, None]).permute(3, 0, 1, 2)
            data_grouped = (demonstrations * mask[:, :, None]).permute(2, 0, 1).unsqueeze(2)
            valid_grouped = valid.any(dim=1).unsqueeze(0).expand(real_dim, batch_size)
        elif self.config.drifting_per_timestep_loss:
            gen_grouped = generated.permute(2, 0, 1, 3)
            data_grouped = demonstrations.transpose(0, 1).unsqueeze(2)
            valid_grouped = valid.transpose(0, 1)
        else:
            mask = valid.float()
            gen_grouped = (generated * mask[:, None, :, None]).flatten(2).unsqueeze(0)
            data_grouped = (demonstrations * mask[:, :, None]).flatten(1)[None, :, None, :]
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
            denom = (valid.sum() * real_dim).clamp(min=1)
            values.extend(
                [
                    ((generated.mean(dim=1) - demonstrations).square() * valid_mask).sum() / denom,
                    (generated.var(dim=1, unbiased=False) * valid_mask).sum() / denom,
                ]
            )
            names.extend(["centroid_mse", "sample_variance"])
            info = dict(zip(names, torch.stack(values).tolist(), strict=True))
        return (per_sample if reduction == "none" else mean_loss), info

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        inputs = self._filter_groot_inputs(batch, include_action=True)
        device = get_device_from_parameters(self)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            head, backbone_output, action_inputs, state_features = self._drift_context(inputs)
            generated = self._sample_drift_candidates(
                head,
                backbone_output,
                action_inputs,
                state_features,
                self.config.drifting_gen_per_label,
            )
            real_dim = self.config.output_features[ACTION].shape[0]
            action_mask = action_inputs.action_mask[..., :real_dim]
            valid = action_mask.to(torch.bool).any(dim=-1)
            loss, info = self._reduce_drift(
                generated, action_inputs.action, valid, real_dim, reduction
            )
        log_dict = {f"drift_{key}": value for key, value in info.items()}
        log_dict["loss"] = loss.mean().item() if reduction == "none" else loss.item()
        return loss, log_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: object) -> Tensor:
        if kwargs.get("prev_chunk_left_over") is not None:
            raise ValueError("GR00T-Drift has no iterative trajectory for RTC overlap guidance.")
        self.eval()
        inputs = self._filter_groot_inputs(batch, include_action=False)
        device = get_device_from_parameters(self)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            head, backbone_output, action_inputs, state_features = self._drift_context(inputs)
            candidates = self._sample_drift_candidates(
                head,
                backbone_output,
                action_inputs,
                state_features,
                self.config.test_time_samples,
            )
        real_dim = self.config.output_features[ACTION].shape[0]
        if self.config.test_time_samples == 1:
            actions = candidates[:, 0]
        else:
            indices, info = cluster_medoid_select(
                candidates[..., :real_dim].flatten(2),
                num_clusters=self.config.test_time_clusters,
                unimodal_tau=self.config.test_time_unimodal_tau,
            )
            self.last_test_time_info = info
            actions = candidates[torch.arange(candidates.shape[0], device=candidates.device), indices]
        prediction_horizon = self._resolve_prediction_horizon(actions)
        return actions[:, :prediction_horizon, :real_dim]
