"""REMOTE VERIFICATION REQUIRED: authored but not executed locally."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.pi05_one_step.keystone import cluster_medoid_select
from lerobot.policies.pi05_one_step.modeling_pi05_one_step import PI05OneStepPytorch
from lerobot.utils.constants import ACTION


def test_cluster_medoid_k1_is_identity():
    candidates = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    indices, info = cluster_medoid_select(candidates)
    torch.testing.assert_close(indices, torch.zeros(2, dtype=torch.long))
    assert info["unimodal"] == [True, True]
    assert info["cluster_size"] == [1, 1]


def test_selector_ignores_padded_action_dimensions():
    core = PI05OneStepPytorch.__new__(PI05OneStepPytorch)
    nn.Module.__init__(core)
    core.config = SimpleNamespace(
        output_features={ACTION: SimpleNamespace(shape=(2,))},
        test_time_clusters=2,
        test_time_unimodal_tau=0.3,
    )
    real = torch.tensor([[[[0.0, 0.0]], [[0.1, 0.0]], [[0.2, 0.0]]]])
    first_padding = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]])
    second_padding = torch.tensor([[[[1e6, -1e6]], [[-9e6, 4e6]], [[7e6, 8e6]]]])
    first = core.select_test_time_chunk(torch.cat([real, first_padding], dim=-1))
    second = core.select_test_time_chunk(torch.cat([real, second_padding], dim=-1))
    torch.testing.assert_close(first[..., :2], second[..., :2])
    torch.testing.assert_close(first[..., :2], torch.tensor([[[0.1, 0.0]]]))


def test_k_greater_than_one_uses_one_batched_candidate_call():
    core = PI05OneStepPytorch.__new__(PI05OneStepPytorch)
    nn.Module.__init__(core)
    core.config = SimpleNamespace(
        method="drifting_perdim",
        chunk_size=3,
        max_action_dim=4,
        test_time_samples=4,
    )
    calls = []

    def build_prefix_cache(self, images, image_masks, tokens, token_masks):
        return torch.ones(tokens.shape[0], 2, dtype=torch.bool), object()

    def sample_direct_n(self, prefix_pad, cache, repeats):
        calls.append(repeats)
        return torch.zeros(prefix_pad.shape[0], repeats, 3, 4)

    def select_test_time_chunk(self, candidates):
        return candidates[:, 0]

    core.build_prefix_cache = MethodType(build_prefix_cache, core)
    core.sample_direct_n = MethodType(sample_direct_n, core)
    core.select_test_time_chunk = MethodType(select_test_time_chunk, core)
    tokens = torch.ones(2, 3, dtype=torch.long)
    output = core.sample_actions([], [], tokens, torch.ones_like(tokens))
    assert output.shape == (2, 3, 4)
    assert calls == [4]


@pytest.mark.parametrize(
    ("samples", "clusters", "tau"),
    [(0, 2, 0.3), (4, 0, 0.3), (4, 2, 0.0)],
)
def test_invalid_selector_arguments(samples, clusters, tau):
    if samples == 0:
        with pytest.raises(ValueError, match="at least one sample"):
            cluster_medoid_select(torch.empty(1, 0, 2), clusters, tau)
    else:
        with pytest.raises(ValueError):
            cluster_medoid_select(torch.zeros(1, samples, 2), clusters, tau)
