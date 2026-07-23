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
| RoboTwin 2.0 | `lerobot/robotwin_unified` at commit `1287871839fae2296bc27b88a5457c3e1eba8e1f` | the unified 50-task dataset |

The five baselines are `pi0`, `pi05`, `smolvla`, `xvla`, and `groot`. Every
model/benchmark pair uses seeds `1000`, `1001`, and `1002`; every seed trains to
50k and saves exactly steps 30k, 40k, and 50k. Therefore:

- one model/benchmark pair: 3 runs and 9 checkpoints;
- full matrix: 30 runs and 90 checkpoints.

All runs use per-process batch 4 on 8 DDP workers (global batch 32). This holds
the number of training examples per optimizer step constant across model
families. Do not increase the smaller models' batch size in the paper runs;
that would change the data budget represented by a 30k/40k/50k checkpoint.
The launcher also sets `ddp_broadcast_buffers=false`: these Transformer VLAs do
not use mutable BatchNorm-style running statistics, while broadcasting immutable
positional/rotary buffers before every forward adds an unnecessary NCCL
collective. `ddp_find_unused_parameters=true` remains enabled for policies with
conditional computation.

The baseline launcher defaults to `num_workers=0`. PyTorch warns that NCCL DDP
combined with DataLoader workers created through the Linux `fork` start method
can deadlock; the failed Pi0.5 run used four workers per rank and stalled after
4,586 otherwise healthy updates. Its logged data time was about 0.003 seconds,
so removing worker processes should have negligible throughput impact. A future
worker-enabled configuration must explicitly use a reviewed `spawn` or
`forkserver` DataLoader context and be validated separately.

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

The public RoboTwin dataset currently exposes its v3 schema on `main` without a
`v3.0` Hub tag. LeRobot otherwise tries to resolve its default `v3.0` revision
and fails before downloading `meta/info.json`. The launcher therefore pins the
snapshot commit above instead of relying on mutable `main`.

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
   Complete an 8-GPU DDP smoke run before committing to a 50k run. A hang here
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

For a long-NCCL preflight, keep the real 50k schedule but write a temporary
recovery checkpoint at step 5k. Stop after observing step 5.1k, then resume with
the formal 30k/40k/50k checkpoint list:

```bash
MODE=train SAVE_FREQ_OVERRIDE=5000 OUTPUT_ROOT=outputs/baselines_nccl_preflight \
  bash experiments/driftingvla/baselines/run_one.sh pi05 libero 1000

CHECKPOINT_STEPS_OVERRIDE='[30000,40000,50000]' \
  OUTPUT_ROOT=outputs/baselines_nccl_preflight \
  bash experiments/driftingvla/baselines/resume_one.sh pi05 libero 1000
```

Run all three 50k seeds sequentially for one model/benchmark pair:

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

The first long run for each model should be only seed 1000. Evaluate its 30k,
40k, and 50k checkpoints before spending compute on seeds 1001 and 1002. If all
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
