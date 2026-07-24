"""REMOTE VERIFICATION REQUIRED: pure-key tests, not executed locally."""

import pytest

from lerobot.policies.pi05_one_step.checkpointing import filter_pretrained_state
from lerobot.policies.pi05_one_step.configuration_pi05_one_step import PI05OneStepConfig


def _synthetic_state():
    return {
        "model.paligemma_with_expert.paligemma.language.weight": "vlm",
        "model.paligemma_with_expert.gemma_expert.layer.weight": "expert",
        "model.action_in_proj.weight": "input",
        "model.action_out_proj.weight": "output",
        "model.time_mlp_in.weight": "time-in",
        "model.time_mlp_out.weight": "time-out",
    }


def test_fresh_expert_keeps_vlm_and_filters_all_expert_keys():
    filtered, prefixes = filter_pretrained_state(
        _synthetic_state(), method="drifting_perdim", fresh_action_expert=True
    )
    assert filtered == {"model.paligemma_with_expert.paligemma.language.weight": "vlm"}
    assert len(prefixes) == 5


def test_config_serializes_method_provenance_fields():
    config = PI05OneStepConfig(
        method="dbp_chunk",
        drifting_grouping="chunk",
        fresh_action_expert=True,
    )
    assert config.method == "dbp_chunk"
    assert config.objective == "dbp"
    assert config.inference_nfe == 1
    assert config.action_expert_init == "fresh"
    assert config.mask_padded_timesteps is True
    assert config.test_time_samples == 1
    assert hasattr(config, "init_label")
    assert hasattr(config, "teacher_checkpoint")


def test_trained_fresh_expert_checkpoint_does_not_filter_expert_again():
    """The immutable provenance label is separate from the one-time load switch."""
    config = PI05OneStepConfig(
        method="drifting_perdim",
        drifting_grouping="perdim",
        fresh_action_expert=False,
        action_expert_init="fresh",
        init_label="pi05_vlm_fresh_action_expert",
    )
    filtered, prefixes = filter_pretrained_state(
        _synthetic_state(),
        method=config.method,
        fresh_action_expert=config.fresh_action_expert,
    )
    assert filtered == _synthetic_state()
    assert prefixes == ()
    assert config.action_expert_init == "fresh"


def test_fresh_provenance_cannot_be_claimed_without_reset_or_checkpoint_label():
    with pytest.raises(ValueError, match="init_label"):
        PI05OneStepConfig(
            method="drifting_perdim",
            drifting_grouping="perdim",
            fresh_action_expert=False,
            action_expert_init="fresh",
        )


def test_keystone_config_rejects_invalid_candidate_cluster_contract():
    with pytest.raises(ValueError, match="test_time_clusters"):
        PI05OneStepConfig(
            method="drifting_perdim",
            drifting_grouping="perdim",
            test_time_samples=4,
            test_time_clusters=1,
        )
