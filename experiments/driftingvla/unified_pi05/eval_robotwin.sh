#!/usr/bin/env bash

# REMOTE SERVER ONLY, on a server with RoboTwin/SAPIEN/MPLib/CuRobo installed.
set -euo pipefail

[[ $# -ge 3 ]] || { echo "Usage: $0 CKPT TASKS demo_clean|demo_randomized [OUTPUT_DIR]" >&2; exit 2; }
checkpoint=$1
tasks=$2
condition=$3
output_dir=${4:-outputs/driftingvla_eval/robotwin/${condition}}
case "$condition" in demo_clean|demo_randomized) ;; *) exit 2 ;; esac
policy_overrides=()
[[ -z "${NFE_OVERRIDE:-}" ]] || policy_overrides+=("--policy.num_inference_steps=${NFE_OVERRIDE}")
[[ -z "${TEST_TIME_SAMPLES:-}" ]] || policy_overrides+=("--policy.test_time_samples=${TEST_TIME_SAMPLES}")
[[ -z "${TEST_TIME_CLUSTERS:-}" ]] || policy_overrides+=("--policy.test_time_clusters=${TEST_TIME_CLUSTERS}")
[[ -z "${TEST_TIME_UNIMODAL_TAU:-}" ]] || policy_overrides+=("--policy.test_time_unimodal_tau=${TEST_TIME_UNIMODAL_TAU}")

lerobot-eval \
  "--policy.path=${checkpoint}/pretrained_model" \
  --env.type=robotwin \
  "--env.task=${tasks}" \
  "--env.task_config=${condition}" \
  "--eval.n_episodes=${EPISODES:-50}" \
  "--eval.batch_size=${EVAL_BATCH_SIZE:-1}" \
  "--eval.latency_warmup_chunk_generations=${LATENCY_WARMUP_CHUNKS:-1}" \
  --eval.use_async_envs=false \
  --eval.recording=false \
  "${policy_overrides[@]}" \
  "--output_dir=${output_dir}"
