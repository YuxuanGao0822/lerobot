"""REMOTE VERIFICATION REQUIRED: implemented but not executed locally."""

import pytest
import torch

from lerobot.policies.pi05_one_step.objectives import drift_loss_grouped, group_drifting_tensors


def _inputs():
    generated = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    demonstration = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    return generated, demonstration, valid


def test_grouping_shapes_and_matching_demo_mask_transform():
    generated, demonstration, valid = _inputs()
    expected = {
        "chunk": ((1, 2, 3, 20), (1, 2, 1, 20), (1, 2)),
        "stepwise": ((4, 2, 3, 5), (4, 2, 1, 5), (4, 2)),
        "perdim": ((5, 2, 3, 4), (5, 2, 1, 4), (5, 2)),
    }
    for grouping, shapes in expected.items():
        gen_group, demo_group, valid_group = group_drifting_tensors(
            generated, demonstration, valid, grouping
        )
        assert tuple(gen_group.shape) == shapes[0]
        assert tuple(demo_group.shape) == shapes[1]
        assert tuple(valid_group.shape) == shapes[2]

    _, _, _, feature_mask = group_drifting_tensors(
        generated, demonstration, valid, "chunk", return_feature_mask=True
    )
    assert tuple(feature_mask.shape) == (1, 2, 20)
    assert feature_mask[0, 0, 10:].eq(0).all()

    chunk_gen, chunk_demo, _ = group_drifting_tensors(generated, demonstration, valid, "chunk")
    assert chunk_gen[0, 0, :, 10:].eq(0).all()
    assert chunk_demo[0, 0, :, 10:].eq(0).all()


def test_padded_timesteps_do_not_change_chunk_or_perdim_loss_and_have_zero_gradient():
    generated, demonstration, valid = _inputs()
    for grouping in ("chunk", "perdim"):
        live = generated.clone().requires_grad_(True)
        grouped, positives, grouped_valid, feature_mask = group_drifting_tensors(
            live, demonstration, valid, grouping, return_feature_mask=True
        )
        loss, _ = drift_loss_grouped(
            grouped, positives, grouped_valid, feature_valid=feature_mask
        )
        baseline = loss.sum()
        baseline.backward()
        assert live.grad[0, :, 2:].eq(0).all()

        changed_generated = generated.clone()
        changed_demo = demonstration.clone()
        changed_generated[0, :, 2:] = 1e5
        changed_demo[0, 2:] = -1e5
        grouped_2, positives_2, valid_2, feature_mask_2 = group_drifting_tensors(
            changed_generated,
            changed_demo,
            valid,
            grouping,
            return_feature_mask=True,
        )
        changed, _ = drift_loss_grouped(
            grouped_2, positives_2, valid_2, feature_valid=feature_mask_2
        )
        torch.testing.assert_close(changed.sum(), baseline.detach(), rtol=1e-5, atol=1e-6)


def test_padded_action_dimensions_are_removed_before_grouping_contract():
    generated, demonstration, valid = _inputs()
    real_dim = 3
    grouped_a = group_drifting_tensors(
        generated[..., :real_dim], demonstration[..., :real_dim], valid, "perdim"
    )
    generated[..., real_dim:] = 1e6
    demonstration[..., real_dim:] = -1e6
    grouped_b = group_drifting_tensors(
        generated[..., :real_dim], demonstration[..., :real_dim], valid, "perdim"
    )
    for left, right in zip(grouped_a, grouped_b, strict=True):
        torch.testing.assert_close(left, right)


def test_masked_horizon_matches_physically_truncated_horizon():
    """Padding must be absent from kernel geometry, not represented as zero features."""
    torch.manual_seed(2028)
    generated = torch.randn(1, 3, 4, 2)
    demonstration = torch.randn(1, 4, 2)
    valid = torch.tensor([[True, True, False, False]])

    for grouping in ("chunk", "perdim"):
        grouped, positives, grouped_valid, feature_mask = group_drifting_tensors(
            generated,
            demonstration,
            valid,
            grouping,
            return_feature_mask=True,
        )
        masked_loss, masked_info = drift_loss_grouped(
            grouped,
            positives,
            grouped_valid,
            feature_valid=feature_mask,
        )

        truncated = group_drifting_tensors(
            generated[:, :, :2],
            demonstration[:, :2],
            torch.ones(1, 2, dtype=torch.bool),
            grouping,
            return_feature_mask=True,
        )
        truncated_loss, truncated_info = drift_loss_grouped(
            truncated[0],
            truncated[1],
            truncated[2],
            feature_valid=truncated[3],
        )
        torch.testing.assert_close(masked_loss, truncated_loss, rtol=2e-5, atol=2e-6)
        for key in masked_info:
            torch.testing.assert_close(masked_info[key], truncated_info[key], rtol=2e-5, atol=2e-6)


def test_kernel_rejects_ambiguous_shapes_and_degenerate_siblings():
    generated = torch.randn(2, 3, 2, 5)
    positives = torch.randn(2, 3, 1, 5)
    valid = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="G >= 2"):
        drift_loss_grouped(generated[:, :, :1], positives, valid)
    with pytest.raises(ValueError, match="share Q, B, and S"):
        drift_loss_grouped(generated, positives[..., :4], valid)
    with pytest.raises(ValueError, match="valid must have shape"):
        drift_loss_grouped(generated, positives, valid[:, :2])
    with pytest.raises(ValueError, match="feature_valid must have shape"):
        drift_loss_grouped(
            generated,
            positives,
            valid,
            feature_valid=torch.ones(2, 3, 4, dtype=torch.bool),
        )
