# Remote execution plan

**All commands below are REMOTE SERVER ONLY.** They may load checkpoints/datasets,
use GPU, start simulators, or write substantial output. They have not been executed
locally.

## Phase 0: environment report

```bash
# REMOTE SERVER ONLY: report environment; may inspect installed packages.
python - <<'PY'
import sys, torch
print(sys.version)
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
git rev-parse HEAD
git status --short
```

Return Python/PyTorch/CUDA versions, GPU list, commit and dirty status.

## Phase 1: smoke matrix

```bash
# REMOTE SERVER ONLY: uses GPU and loads pi05_base.
MODE=smoke NUM_PROCESSES=1 GPU_IDS=0 BATCH_SIZE=1 DRIFTING_G=2 \
  bash experiments/driftingvla/unified_pi05/run_train.sh pi05_dbp_chunk libero 1000
MODE=smoke NUM_PROCESSES=1 GPU_IDS=0 BATCH_SIZE=1 DRIFTING_G=2 \
  bash experiments/driftingvla/unified_pi05/run_train.sh pi05_dbp_stepwise libero 1000
MODE=smoke NUM_PROCESSES=1 GPU_IDS=0 BATCH_SIZE=1 DRIFTING_G=2 \
  bash experiments/driftingvla/unified_pi05/run_train.sh pi05_driftingvla libero 1000
```

Expected evidence: finite loss, expected tensor shapes, saved smoke checkpoint and
no missing-key/random-weight fallback. Actual success is pending remote output.

## Phase 1b: authored test suite

```bash
# REMOTE SERVER ONLY: executes authored tests and may initialize PyTorch.
export DRIFTINGVLA_PI05_DRIFT_REFERENCE_ROOT=/path/to/lerobot_policy_pi05_drift-main
export DRIFTINGVLA_DBP_REFERENCE_ROOT=/path/to/drift-based-policy
pytest -q \
  tests/policies/pi05_one_step \
  tests/envs/test_robotwin.py \
  tests/scripts/test_lerobot_eval_latency.py
```

Return the complete test summary, failing node IDs, package versions and traceback.
These tests are authored locally but are intentionally not executed in this workspace.
They include KeyStone K=1 identity, K>1 batched dispatch, and invariance to padded
action columns.

## Phase 2: baseline training

Run stock π0.5 and each one-step method with the declared three seeds. Use 8-card
DDP only after the single-device smoke run is clean. Preserve checkpoints at 30k,
40k and 50k. Do not evaluate RoboTwin on a server without its simulator dependencies.

```bash
# REMOTE SERVER ONLY: inspect the exact 24-run matrix first; no project execution in dry-run mode.
MODE=dry-run bash experiments/driftingvla/unified_pi05/launch_train_matrix.sh

# REMOTE SERVER ONLY: sequential 8-GPU training, model/data loading and checkpoint writes.
MODE=train bash experiments/driftingvla/unified_pi05/launch_train_matrix.sh
```

## Phase 3: LIBERO evaluation

The paired launcher assigns four suites for one checkpoint to GPUs 0–3 and four
suites for a second checkpoint to GPUs 4–7. Verify that each evaluator has an isolated
CUDA device and output directory. Record success, episodes, action calls, chunk
regenerations and synchronized latency percentiles.

```bash
# REMOTE SERVER ONLY: two checkpoints × four suites on GPUs 0--7.
EPISODES=50 EVAL_BATCH_SIZE=10 LATENCY_WARMUP_CHUNKS=1 \
  bash experiments/driftingvla/unified_pi05/eval_libero_pair.sh \
  /path/to/checkpoint_A /path/to/checkpoint_B /path/to/output_pair

# REMOTE SERVER ONLY: naive one-step uses the unchanged flow checkpoints.
NFE_OVERRIDE=1 EPISODES=50 EVAL_BATCH_SIZE=10 LATENCY_WARMUP_CHUNKS=1 \
  bash experiments/driftingvla/unified_pi05/eval_libero_pair.sh \
  /path/to/flow_checkpoint_A /path/to/flow_checkpoint_B /path/to/naive_output_pair
```

Use the 50k checkpoints for the primary table. Evaluate 30k/40k separately for the
learning curve; never choose a method-specific checkpoint from these test results.

## Phase 4: RoboTwin evaluation

Run on the separate server that has the RoboTwin simulator and rendering stack. Export
the exact dataset overlay and task condition. Return environment versions and task
configuration alongside results.

```bash
# REMOTE SERVER ONLY: simulator/GPU required. Easy=demo_clean, Hard=demo_randomized.
LATENCY_WARMUP_CHUNKS=1 \
  bash experiments/driftingvla/unified_pi05/eval_robotwin.sh \
  /path/to/checkpoint "task_a,task_b" demo_clean /path/to/easy_output
LATENCY_WARMUP_CHUNKS=1 \
  bash experiments/driftingvla/unified_pi05/eval_robotwin.sh \
  /path/to/checkpoint "task_a,task_b" demo_randomized /path/to/hard_output
```

## Phase 5: profiling

Use the instrumented evaluation output plus a short warm-up exclusion. Report mean,
p50 and p95 for action generation and policy pipeline latency. If CUDA events or
device synchronization are unavailable, label the measurement CPU-wall-clock and do
not compare it directly with synchronized GPU timings. The launchers default to one
cold action-chunk generation excluded per task (`LATENCY_WARMUP_CHUNKS=1`). The final
`eval_info.json` must retain per-task, per-suite, and overall timing summaries together
with `configured_nfe`, expected expert-forward count, policy-select calls, chunk
generations, policy-side rate, allocated/reserved peak memory, and the warm-up count.
For optional KeyStone runs it must additionally record `test_time_samples` and
`candidate_equivalent_nfe_per_chunk`; all primary runs use `test_time_samples=1`.
Peak memory includes the resident model and is aggregated by the maximum across tasks;
run profiling with `max_parallel_tasks=1`. `configured_nfe` is declarative provenance,
not a runtime profiler measurement.

## Remote result bundle

Return one archive or directory containing the command/environment manifest, logs,
checkpoint config and file listing, per-seed/per-checkpoint metrics, latency JSON/CSV,
failure traces/videos where applicable, and commit/diff information.
