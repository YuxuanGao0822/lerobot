# Reproduction protocol

This document is a protocol, not an execution record. Every command is intended for
the remote server only.

## Environment and installation (remote server only)

Use the repository's pinned environment or container. Do not install dependencies in
the local workspace. On the remote host, record the output of the environment report
in `REMOTE_EXECUTION_PLAN.md` before installing any missing optional extras. The
required capabilities are the LeRobot training/evaluation extras, PyTorch/Transformers
for π0.5, LIBERO for simulation, and the RoboTwin/SAPIEN/MPLib/CuRobo stack on the
separate RoboTwin evaluation server.

## Dataset preparation

Use the exact dataset revision in the run manifest. For RoboTwin, compute and review
the q01/q99 quantile overlay once on the remote server and export
`ROBOTWIN_DATASET_ROOT`; do not recompute statistics independently for each method.
Record the overlay path, hash, real action dimension and task condition (`demo_clean`
or `demo_randomized`).

## Fixed factors

- Start from the released `pi05_base` checkpoint for post-training methods.
- Keep dataset revision, normalization overlay, camera preprocessing, language
  formatting, action horizon and seed manifest fixed within each benchmark.
- Report seeds individually and as mean ± standard deviation or confidence interval.
- Save checkpoints at 30000, 40000 and 50000 steps for long runs.
- Use 50000 steps as the primary endpoint; use 30000/40000 only for learning-curve
  and training-efficiency analysis. Do not select a method-specific best checkpoint
  from test success.
- Keep the legacy `pi05_drift` package unchanged while comparing the new package.

## Required run manifest

Each run must record:

```text
method, benchmark, suite/task, seed, git_commit, checkpoint_source,
checkpoint_step, dataset_repo, dataset_revision, dataset_root,
action_dim_real, chunk_size, drifting_grouping, drifting_gen_per_label,
temperatures, fresh_action_expert, action_expert_init, init_label,
mask_padded_timesteps, train_expert_only, dtype,
test_time_samples, test_time_clusters, test_time_unimodal_tau,
gradient_checkpointing, batch_size, optimizer, lr, total_steps,
evaluation_episodes, device_ids, wall_clock_start, wall_clock_end
```

## Remote-only commands

The launchers print commands in dry-run mode. To execute a smoke run:

```bash
# REMOTE SERVER ONLY: loads the model/dataset and uses GPU.
MODE=smoke NUM_PROCESSES=1 GPU_IDS=0 BATCH_SIZE=1 DRIFTING_G=2 \
  bash experiments/driftingvla/unified_pi05/run_train.sh pi05_dbp_chunk libero 1000
```

To execute the full training protocol:

```bash
# REMOTE SERVER ONLY: trains with GPU and writes checkpoints/logs.
MODE=train RUN_STEPS=50000 SAVE_FREQ_OVERRIDE=10000 \
  bash experiments/driftingvla/unified_pi05/run_train.sh pi05_driftingvla libero 1000
```

The same launcher accepts `robotwin` after the reviewed
`ROBOTWIN_DATASET_ROOT` is exported. Do not run these commands locally.

To print the complete controlled matrix without loading a model or dataset on the
remote host, and then to run it sequentially after inspection:

```bash
# REMOTE SERVER ONLY: dry-run prints 4 trained methods × 2 benchmarks × 3 seeds.
MODE=dry-run bash experiments/driftingvla/unified_pi05/launch_train_matrix.sh

# REMOTE SERVER ONLY: uses all configured GPUs and writes 30k/40k/50k checkpoints.
MODE=train bash experiments/driftingvla/unified_pi05/launch_train_matrix.sh
```

`pi05_flow_naive_1step` is deliberately absent because it reuses each trained flow
checkpoint with `NFE_OVERRIDE=1` during evaluation. MeanFlow-VLA and SnapFlow are
author-managed external reproductions and are not dispatched by this matrix.

The primary DriftingVLA tables must use `test_time_samples=1`. KeyStone-style
self-consistency is an optional inference-only ablation with `K in {1,4,8}`. For
`K>1`, report both the batched action-expert call count and
`candidate_equivalent_nfe_per_chunk`; do not describe K parallel candidates as having
the same compute as K=1 merely because they share a single batched module invocation.

## Checkpoint conversion and loading

Stock `pi05_base` is loaded through the strict `PI05OneStepPolicy.from_pretrained`
path for DBP methods. The loader must report the source checkpoint, the number of
loaded keys, skipped fresh-expert prefixes, missing keys and unexpected keys. A
checkpoint that falls back to random weights is invalid. After a trained checkpoint is
saved, reload it with the serialized policy config and verify that the `init_label`
and immutable `action_expert_init` provenance do not trigger a second fresh reset.
The one-time `fresh_action_expert` switch is expected to be serialized as false after
initialization, while `action_expert_init=fresh` remains. Return the checkpoint file
listing and loader summary with the run bundle.

## Acceptance criteria

Do not proceed to the full matrix until the remote smoke run confirms:

1. dataset sample contains images, language, state, action and `action_is_pad`;
2. expected padded shape is `[B,50,32]` and real dimension is reported;
3. one-step sampler produces `[B,50,D_real]` after postprocessing;
4. finite loss and gradient norms are logged;
5. checkpoint contains serialized policy and training configuration;
6. save/reload produces the same output structure;
7. no unexpected missing or random-initialization fallback is reported.

Failure output to return for diagnosis: complete traceback, command, policy config,
first batch feature keys/shapes, checkpoint file listing, and the first 100 log lines.
