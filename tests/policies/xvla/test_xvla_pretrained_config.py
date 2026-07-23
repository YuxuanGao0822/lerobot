#!/usr/bin/env python

import pytest

pytest.importorskip("transformers")

from lerobot.configs import PreTrainedConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.policies.xvla.modeling_xvla import _hydrate_xvla_architecture_config


def test_hydrate_xvla_architecture_preserves_runtime_overrides(monkeypatch) -> None:
    released_florence_config = {
        "vision_config": {"model_type": "florence_vision", "embed_dim": 1024},
        "text_config": {"model_type": "bart", "d_model": 1024},
    }
    released_config = XVLAConfig(florence_config=released_florence_config, domain_id=0)
    runtime_config = XVLAConfig(
        florence_config={},
        domain_id=3,
        dtype="bfloat16",
        action_mode="auto",
    )

    monkeypatch.setattr(
        PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: released_config,
    )

    resolved = _hydrate_xvla_architecture_config(runtime_config, "lerobot/xvla-base")

    assert resolved is runtime_config
    assert resolved.florence_config == released_florence_config
    assert resolved.domain_id == 3
    assert resolved.dtype == "bfloat16"
    assert resolved.action_mode == "auto"


def test_hydrate_xvla_architecture_rejects_non_xvla_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        PreTrainedConfig,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(ValueError, match="X-VLA checkpoint"):
        _hydrate_xvla_architecture_config(None, "not-an-xvla-checkpoint")
