#!/usr/bin/env bash

# Resume one interrupted baseline run from checkpoints/last using the same DDP
# world size. The saved train config supplies model, dataset, batch size, and the
# 50k target; CLI values below only select the checkpoint and output location.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash experiments/driftingvla/baselines/resume_one.sh MODEL BENCHMARK SEED" >&2
  exit 2
fi

model=$1
benchmark=$2
seed=$3
num_processes=${NUM_PROCESSES:-8}
gpu_ids=${GPU_IDS:-0,1,2,3,4,5,6,7}
output_root=${OUTPUT_ROOT:-outputs/baselines}
run_dir="${output_root}/train/${benchmark}/${model}/seed_${seed}"
checkpoint_dir="${run_dir}/checkpoints/last"
config_path="${checkpoint_dir}/pretrained_model/train_config.json"
log_dir="${output_root}/logs/resume/${benchmark}/${model}"
log_file="${log_dir}/seed_${seed}.log"

if [[ ! -f "$config_path" ]]; then
  echo "Resume config not found: $config_path" >&2
  exit 1
fi

train_entry=$(command -v lerobot-train || true)
if [[ -z "$train_entry" ]]; then
  train_entry=lerobot-train
fi

cmd=(
  accelerate launch
  --multi_gpu
  "--num_processes=${num_processes}"
  "$train_entry"
  "--config_path=${config_path}"
  "--resume=true"
  "--output_dir=${run_dir}"
)

if [[ -n "${SAVE_FREQ_OVERRIDE:-}" ]]; then
  cmd+=("--save_freq=${SAVE_FREQ_OVERRIDE}")
fi
if [[ -n "${CHECKPOINT_STEPS_OVERRIDE:-}" ]]; then
  cmd+=("--checkpoint_steps=${CHECKPOINT_STEPS_OVERRIDE}")
fi

printf 'Resume command:'
printf ' %q' env "CUDA_VISIBLE_DEVICES=${gpu_ids}" "${cmd[@]}"
printf '\n'

if [[ "${MODE:-resume}" == "dry-run" ]]; then
  exit 0
fi

mkdir -p "$log_dir"
CUDA_VISIBLE_DEVICES="$gpu_ids" "${cmd[@]}" 2>&1 | tee -a "$log_file"
