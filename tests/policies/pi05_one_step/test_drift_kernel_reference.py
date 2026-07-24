"""REMOTE VERIFICATION REQUIRED: numerical reference tests, not executed locally."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from lerobot.policies.pi05_one_step.objectives.drift_kernel import drift_loss_grouped
from lerobot.policies.pi05_one_step.objectives.grouping import group_drifting_tensors


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reference module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_pi05_reference_root(root: str) -> Path:
    """Accept either the standalone package root or its source package directory."""
    candidate = Path(root)
    direct = candidate / "drifting_util.py"
    nested = candidate / "src" / "lerobot_policy_pi05_drift" / "drifting_util.py"
    if direct.exists():
        return candidate
    if nested.exists():
        return nested.parent
    raise FileNotFoundError(
        "Expected drifting_util.py at the reference root or "
        "src/lerobot_policy_pi05_drift/drifting_util.py"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grouped_kernel_matches_authoritative_pi05_reference(dtype):
    root = os.environ.get("DRIFTINGVLA_PI05_DRIFT_REFERENCE_ROOT")
    if not root:
        pytest.fail("Set DRIFTINGVLA_PI05_DRIFT_REFERENCE_ROOT for the required remote check.")
    reference_root = _resolve_pi05_reference_root(root)
    reference = _load_module(reference_root / "drifting_util.py", "pi05_drift_reference")

    torch.manual_seed(2026)
    generated = torch.randn(4, 2, 3, 5, dtype=dtype, requires_grad=True)
    positives = torch.randn(4, 2, 1, 5, dtype=dtype)
    valid = torch.tensor(
        [[True, True], [True, False], [False, True], [False, False]], dtype=torch.bool
    )
    candidate_loss, candidate_info = drift_loss_grouped(generated, positives, valid)
    reference_generated = generated.detach().clone().requires_grad_(True)
    reference_loss, reference_info = reference.drift_loss_grouped(
        reference_generated, positives, valid
    )
    tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(candidate_loss, reference_loss, **tolerance)
    for key in candidate_info:
        torch.testing.assert_close(candidate_info[key], reference_info[key], **tolerance)

    candidate_loss.sum().backward()
    reference_loss.sum().backward()
    torch.testing.assert_close(generated.grad, reference_generated.grad, **tolerance)
    assert generated.grad[3].eq(0).all()


def test_chunk_and_stepwise_match_diffusion_policy_reference():
    root = os.environ.get("DRIFTINGVLA_DBP_REFERENCE_ROOT")
    if not root:
        pytest.fail("Set DRIFTINGVLA_DBP_REFERENCE_ROOT for the required remote check.")
    reference = _load_module(
        Path(root) / "diffusion_policy/model/drifting/drifting_util.py",
        "diffusion_policy_drift_reference",
    )
    torch.manual_seed(2027)
    generated = torch.randn(2, 4, 3, 5)
    demonstration = torch.randn(2, 3, 5)
    valid = torch.ones(2, 3, dtype=torch.bool)

    candidate_chunk_generated = generated.clone().requires_grad_(True)
    chunk, positive_chunk, valid_chunk = group_drifting_tensors(
        candidate_chunk_generated, demonstration, valid, "chunk"
    )
    chunk_loss, chunk_info = drift_loss_grouped(chunk, positive_chunk, valid_chunk)
    reference_chunk_generated = generated.clone().requires_grad_(True)
    reference_chunk, reference_chunk_info = reference.drift_loss(
        reference_chunk_generated.flatten(2),
        demonstration.flatten(1).unsqueeze(1),
        R_list=(0.02, 0.05, 0.2),
    )
    torch.testing.assert_close(chunk_loss.squeeze(0), reference_chunk, rtol=2e-5, atol=2e-6)
    for key in chunk_info:
        torch.testing.assert_close(
            chunk_info[key].squeeze(0), reference_chunk_info[key], rtol=2e-5, atol=2e-6
        )
    chunk_loss.sum().backward()
    reference_chunk.sum().backward()
    torch.testing.assert_close(
        candidate_chunk_generated.grad,
        reference_chunk_generated.grad,
        rtol=2e-5,
        atol=2e-6,
    )

    candidate_step_generated = generated.clone().requires_grad_(True)
    step, positive_step, valid_step = group_drifting_tensors(
        candidate_step_generated, demonstration, valid, "stepwise"
    )
    step_loss, step_info = drift_loss_grouped(step, positive_step, valid_step)
    reference_step_generated = generated.clone().requires_grad_(True)
    reference_step_losses = []
    for timestep in range(3):
        reference_step, reference_step_info = reference.drift_loss(
            reference_step_generated[:, :, timestep],
            demonstration[:, timestep].unsqueeze(1),
            R_list=(0.02, 0.05, 0.2),
        )
        reference_step_losses.append(reference_step)
        torch.testing.assert_close(step_loss[timestep], reference_step, rtol=2e-5, atol=2e-6)
        for key in step_info:
            torch.testing.assert_close(
                step_info[key][timestep], reference_step_info[key], rtol=2e-5, atol=2e-6
            )
    step_loss.sum().backward()
    torch.stack(reference_step_losses).sum().backward()
    torch.testing.assert_close(
        candidate_step_generated.grad,
        reference_step_generated.grad,
        rtol=2e-5,
        atol=2e-6,
    )
