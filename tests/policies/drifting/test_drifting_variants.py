import importlib

import pytest
import torch

import lerobot.policies  # noqa: F401 -- imports built-in config registrations
from lerobot.configs import PreTrainedConfig


DRIFT_VARIANTS = (
    ("pi0_drift", "lerobot.policies.pi0_drift.drifting_util"),
    ("pi05_drift", "lerobot.policies.pi05_drift.drifting_util"),
    ("smolvla_drift", "lerobot.policies.smolvla_drift.drifting_util"),
    ("xvla_drift", "lerobot.policies.xvla_drift.drifting_util"),
    ("groot_drift", "lerobot.policies.groot_drift.drifting_util"),
)


@pytest.mark.parametrize(("policy_type", "module_name"), DRIFT_VARIANTS)
def test_drift_variant_is_registered_and_loss_has_gradient(policy_type, module_name):
    assert policy_type in PreTrainedConfig.get_known_choices()

    drift_loss_grouped = importlib.import_module(module_name).drift_loss_grouped
    generated = torch.randn(3, 2, 4, 5, requires_grad=True)
    demonstrations = torch.randn(3, 2, 1, 5)
    valid = torch.tensor([[True, True], [True, False], [False, False]])

    loss, info = drift_loss_grouped(generated, demonstrations, valid)

    assert loss.shape == (3, 2)
    assert torch.isfinite(loss).all()
    assert loss[2].eq(0).all()
    assert set(info) == {"scale", "loss_0.02", "loss_0.05", "loss_0.2"}

    loss.sum().backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()
    assert generated.grad[2].eq(0).all()
