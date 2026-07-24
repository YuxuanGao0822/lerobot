#!/usr/bin/env bash

# REMOTE SERVER ONLY. Evaluate two checkpoints concurrently: four LIBERO suites
# on GPUs 0-3 for checkpoint A and GPUs 4-7 for checkpoint B.
set -euo pipefail

[[ $# -ge 2 ]] || { echo "Usage: $0 CKPT_A CKPT_B [OUTPUT_ROOT]" >&2; exit 2; }
ckpt_a=$1
ckpt_b=$2
output_root=${3:-outputs/driftingvla_eval/libero}
episodes=${EPISODES:-50}
batch_size=${EVAL_BATCH_SIZE:-10}
suites=(libero_spatial libero_object libero_goal libero_10)
policy_overrides=()
[[ -z "${NFE_OVERRIDE:-}" ]] || policy_overrides+=("--policy.num_inference_steps=${NFE_OVERRIDE}")
[[ -z "${TEST_TIME_SAMPLES:-}" ]] || policy_overrides+=("--policy.test_time_samples=${TEST_TIME_SAMPLES}")
[[ -z "${TEST_TIME_CLUSTERS:-}" ]] || policy_overrides+=("--policy.test_time_clusters=${TEST_TIME_CLUSTERS}")
[[ -z "${TEST_TIME_UNIMODAL_TAU:-}" ]] || policy_overrides+=("--policy.test_time_unimodal_tau=${TEST_TIME_UNIMODAL_TAU}")

launch_checkpoint() {
  local checkpoint=$1
  local first_gpu=$2
  local label=$3
  local pids=()
  for index in 0 1 2 3; do
    local suite=${suites[$index]}
    local gpu=$((first_gpu + index))
    local out="${output_root}/${label}/${suite}"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" lerobot-eval \
      "--policy.path=${checkpoint}/pretrained_model" \
      --env.type=libero \
      "--env.task=${suite}" \
      "--eval.n_episodes=${episodes}" \
      "--eval.batch_size=${batch_size}" \
      "--eval.latency_warmup_chunk_generations=${LATENCY_WARMUP_CHUNKS:-1}" \
      --eval.use_async_envs=true \
      --eval.recording=false \
      "${policy_overrides[@]}" \
      "--output_dir=${out}" \
      >"${out}/eval.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

launch_checkpoint "$ckpt_a" 0 checkpoint_a &
pid_a=$!
launch_checkpoint "$ckpt_b" 4 checkpoint_b &
pid_b=$!
wait "$pid_a"
wait "$pid_b"
