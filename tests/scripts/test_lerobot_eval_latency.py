"""REMOTE VERIFICATION REQUIRED: latency accounting tests, not executed locally."""

from types import SimpleNamespace

from lerobot.scripts.lerobot_eval import (
    _configured_policy_nfe,
    _configured_test_time_samples,
    _exclude_chunk_warmup,
    _rate_hz_from_timing,
    _timing_summary_ms,
)


def test_chunk_warmup_excludes_samples_through_first_generation():
    action = [0.10, 0.01, 0.02, 0.20, 0.01]
    pipeline = [0.11, 0.02, 0.03, 0.21, 0.02]
    generated = [True, False, False, True, False]
    measured_action, measured_pipeline, measured_generated, excluded = _exclude_chunk_warmup(
        action, pipeline, generated, warmup_chunk_generations=1
    )
    assert measured_action == action[1:]
    assert measured_pipeline == pipeline[1:]
    assert measured_generated == generated[1:]
    assert excluded == 1


def test_timing_summary_uses_milliseconds_and_exact_percentiles():
    summary = _timing_summary_ms([0.001, 0.002, 0.003])
    assert summary["mean_ms"] == 2.0
    assert summary["p50_ms"] == 2.0
    assert summary["p95_ms"] == 2.9
    assert _rate_hz_from_timing(summary) == 500.0


def test_declared_nfe_prefers_native_one_step_provenance():
    direct = SimpleNamespace(config=SimpleNamespace(inference_nfe=1, num_inference_steps=10))
    flow = SimpleNamespace(config=SimpleNamespace(num_inference_steps=10))
    unknown = SimpleNamespace(config=SimpleNamespace())
    assert _configured_policy_nfe(direct) == 1
    assert _configured_policy_nfe(flow) == 10
    assert _configured_policy_nfe(unknown) is None


def test_test_time_sample_provenance_defaults_to_one():
    keystone = SimpleNamespace(config=SimpleNamespace(test_time_samples=8))
    ordinary = SimpleNamespace(config=SimpleNamespace())
    assert _configured_test_time_samples(keystone) == 8
    assert _configured_test_time_samples(ordinary) == 1
