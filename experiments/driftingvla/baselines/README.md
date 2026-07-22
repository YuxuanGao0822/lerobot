# DriftingVLA baseline post-training matrix

This directory defines the non-drifting reference runs for the DriftingVLA
study. All commands are intended for the remote 8 x A800 80 GB server. Do not
run training, evaluation, model loading, or dataset loading in the local
workspace.

## Matrix and experimental unit

The benchmark names refer to the unified training datasets, not to one separate
training run per simulator task:

| Benchmark | Dataset | Scope |
|---|---|---|
| LIBERO | `lerobot/libero` | the four LIBERO suites in the unified dataset |
| RoboTwin 2.0 | `lerobot/robotwin_unified` | the unified 50-task dataset |

The five baselines are `pi0`, `pi05`, `smolvla`, `xvla`, and `groot`. Every
model/benchmark pair uses seeds `1000`, `1001`, and `1002`; every seed saves
steps 20k, 40k, and 60k. Therefore:

- one model/benchmark pair: 3 runs and 9 checkpoints;
- full matrix: 30 runs and 90 checkpoints.

All runs use per-process batch 4 on 8 DDP workers (global batch 32). This holds
the number of training examples per optimizer step constant across model
families. Do not increase the smaller models' batch size in the paper runs;
that would change the data budget represented by a 20k/40k/60k checkpoint.

## Post-training contracts

- Pi0 and Pi0.5: full post-training from `lerobot/pi0_base` and
  `lerobot/pi05_base`, BF16 computation, gradient checkpointing.
- SmolVLA: released expert-only post-training partition (vision encoder frozen,
  expert and state projection trainable) from `lerobot/smolvla_base`.
- X-VLA: full recommended adaptation from `lerobot/xvla-base`, BF16, automatic
  action padding, domain 3 for LIBERO and domain 6 for RoboTwin. The policy is
  constructed with each current dataset's visual feature names, so RoboTwin's
  three camera keys do not need a rename map.
- GR00T N1.7: `nvidia/GR00T-N1.7-3B`, frozen LLM/vision encoders and trainable
  projector, diffusion action model, and VL action-head layers. LIBERO uses
  `libero_sim`; RoboTwin uses `new_embodiment`.

Each future Drift run must match its baseline's initialization, trainable
parameter partition, global batch, optimizer schedule, data, seed, and training
steps. The action-generation objective should be the controlled difference.

## Required preflight

1. Revoke any Hugging Face token pasted into chat or logs and create a new
   read-only token. Export it only in the remote shell; never add it to this
   repository.
2. Check free disk space. Full checkpoints retain model, optimizer, scheduler,
   and RNG state. Full-tuned 4B models can make the 90-checkpoint matrix exceed
   2 TB; reserve at least 2.5-3 TB before starting.
3. The reported remote kernel is 4.19, below Accelerate's recommended 5.5.
   Complete an 8-GPU DDP smoke run before committing to a 60k run. A hang here
   is an infrastructure issue; changing code or retrying the entire matrix is
   not an adequate substitute for a supported host kernel.
4. Confirm that Pi0.5 quantile statistics exist in both dataset metadata files.
   If they do not, stop and fix/augment the remote dataset metadata before
   training; do not silently change Pi0.5 normalization only for one benchmark.

## Commands

All examples below are **REMOTE SERVER ONLY**.

Print one command without loading a model or dataset:

```bash
MODE=dry-run bash experiments/driftingvla/baselines/run_one.sh pi05 libero 1000
```

Run one 2-step, 8-GPU DDP smoke test. It loads the selected model and dataset and
saves a smoke checkpoint outside the final training directory:

```bash
MODE=smoke bash experiments/driftingvla/baselines/run_one.sh pi05 libero 1000
```

Run all three 60k seeds sequentially for one model/benchmark pair:

```bash
MODE=train WANDB_ENABLE=true \
  bash experiments/driftingvla/baselines/launch_group.sh pi05 libero
```

Resume one interrupted seed from `checkpoints/last` with the same eight-worker
world size:

```bash
bash experiments/driftingvla/baselines/resume_one.sh pi05 libero 1000
```

Audit the complete matrix without loading models or datasets:

```bash
bash experiments/driftingvla/baselines/audit_checkpoints.sh
```

## Recommended launch order

Do not submit all 30 long jobs at once. For each model, first smoke both datasets
and inspect the logged parameter count, peak memory, first-step loss, checkpoint
tree, and DDP synchronization. A practical order is:

1. SmolVLA on LIBERO, then RoboTwin;
2. Pi0.5 on LIBERO, then RoboTwin;
3. Pi0 on LIBERO, then RoboTwin;
4. X-VLA on LIBERO, then RoboTwin;
5. GR00T N1.7 on LIBERO, then RoboTwin.

The first long run for each model should be only seed 1000. Evaluate its 20k,
40k, and 60k checkpoints before spending compute on seeds 1001 and 1002. If all
three checkpoints are uniformly poor, diagnose the model/dataset contract rather
than multiplying a broken configuration by three seeds.

## What to return after a smoke run

Return the smallest useful diagnostic bundle:

- the complete launcher log from config print through the first two losses and
  checkpoint save;
- `nvidia-smi` or `gpustat` captured while the training step is active (not after
  the process exits);
- the output of `find <run_dir>/checkpoints -maxdepth 3 -type f -printf '%p %s\n'`;
- for failures: full traceback, model/benchmark/seed, exact command, and the
  first tensor-shape mismatch or missing-key message.

All configurations and scripts in this directory are statically reviewed and
pending remote execution validation.
