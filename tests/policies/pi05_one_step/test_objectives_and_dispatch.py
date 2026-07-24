"""REMOTE VERIFICATION REQUIRED: authored but not executed locally."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05_one_step.modeling_pi05_one_step import PI05OneStepPytorch
from lerobot.policies.pi05_one_step.objectives import masked_mse


@pytest.mark.parametrize("method", ["dbp_chunk", "dbp_stepwise", "drifting_perdim"])
def test_direct_methods_ignore_num_steps_and_decode_once(method):
    core = PI05OneStepPytorch.__new__(PI05OneStepPytorch)
    nn.Module.__init__(core)
    core.config = SimpleNamespace(method=method, chunk_size=4, max_action_dim=5)
    calls = []

    def build_prefix_cache(self, images, image_masks, tokens, token_masks):
        return torch.ones(tokens.shape[0], 2, dtype=torch.bool), object()

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def decode_cached(self, action_state, time, target_time, prefix_pad, cache):
        calls.append((time.clone(), target_time))
        return torch.full_like(action_state, 7.0)

    core.build_prefix_cache = MethodType(build_prefix_cache, core)
    core.sample_noise = MethodType(sample_noise, core)
    core.decode_cached = MethodType(decode_cached, core)
    tokens = torch.ones(2, 3, dtype=torch.long)
    output = core.sample_actions([], [], tokens, torch.ones_like(tokens), num_steps=99)
    assert output.shape == (2, 4, 5)
    assert output.eq(7).all()
    assert len(calls) == 1
    assert calls[0][0].eq(1).all()
    assert calls[0][1] is None


def test_stock_pi05_defaults_remain_multistep_flow():
    config = PI05Config()
    assert config.num_inference_steps == 10
    assert not hasattr(config, "method")


def test_masked_mse_averages_only_valid_timesteps_and_dimensions():
    prediction = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, False]])
    loss = masked_mse(prediction, target, valid)
    torch.testing.assert_close(loss, torch.tensor([5.0]))
