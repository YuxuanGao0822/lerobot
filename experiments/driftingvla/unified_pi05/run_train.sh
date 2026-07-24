#!/usr/bin/env bash

# REMOTE SERVER ONLY. This launcher loads models/datasets unless MODE=dry-run.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

usage() {
  echo "Usage: MODE=dry-run|smoke|train bash $0 METHOD BENCHMARK SEED" >&2
  echo "METHOD: pi05_flow | pi05_dbp_chunk | pi05_dbp_stepwise | pi05_driftingvla" >&2
  echo "BENCHMARK: libero | robotwin" >&2
}

[[ $# -eq 3 ]] || { usage; exit 2; }
method=$1
benchmark=$2
seed=$3
mode=${MODE:-dry-run}

case "$method" in
  pi05_flow|pi05_dbp_chunk|pi05_dbp_stepwise|pi05_driftingvla) ;;
  pi05_flow_naive_1step)
    echo "pi05_flow_naive_1step has no training run; evaluate a pi05_flow checkpoint with NFE=1." >&2
    exit 2
    ;;
  *) usage; exit 2 ;;
esac
case "$benchmark" in libero|robotwin) ;; *) usage; exit 2 ;; esac
case "$mode" in dry-run|smoke|train) ;; *) usage; exit 2 ;; esac

num_processes=${NUM_PROCESSES:-8}
gpu_ids=${GPU_IDS:-0,1,2,3,4,5,6,7}
batch_size=${BATCH_SIZE:-1}
output_root=${OUTPUT_ROOT:-outputs/driftingvla_unified}
wandb_enable=${WANDB_ENABLE:-false}
[[ "$num_processes" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_PROCESSES must be a positive integer" >&2; exit 2; }

if [[ "$mode" == train ]]; then
  steps=${RUN_STEPS:-50000}
  save_freq=${SAVE_FREQ_OVERRIDE:-50000}
  checkpoint_steps=${CHECKPOINT_STEPS_OVERRIDE:-'[50000]'}
  log_freq=100
else
  steps=${RUN_STEPS:-2}
  save_freq=${SAVE_FREQ_OVERRIDE:-$steps}
  checkpoint_steps="[${steps}]"
  log_freq=1
fi

dataset_args=(
  "--dataset.repo_id=lerobot/libero"
  "--dataset.video_backend=${VIDEO_BACKEND:-pyav}"
)
if [[ "$benchmark" == robotwin ]]; then
  robotwin_root=${ROBOTWIN_DATASET_ROOT:-}
  if [[ "$mode" != dry-run && -z "$robotwin_root" ]]; then
    echo "ROBOTWIN_DATASET_ROOT must point to the reviewed q01/q99 overlay." >&2
    exit 2
  fi
  dataset_args=(
    "--dataset.repo_id=lerobot/robotwin_unified"
    "--dataset.revision=1287871839fae2296bc27b88a5457c3e1eba8e1f"
    "--dataset.video_backend=${VIDEO_BACKEND:-pyav}"
  )
  [[ -z "$robotwin_root" ]] || dataset_args+=("--dataset.root=${robotwin_root}")
fi

policy_args=(
  "--policy.type=pi05"
  "--policy.pretrained_path=lerobot/pi05_base"
  "--policy.dtype=bfloat16"
  "--policy.gradient_checkpointing=true"
  "--policy.compile_model=false"
  "--policy.train_expert_only=false"
  "--policy.push_to_hub=false"
)

if [[ "$method" != pi05_flow ]]; then
  one_step_method=${method#pi05_}
  case "$one_step_method" in
    driftingvla) one_step_method=drifting_perdim; grouping=perdim ;;
    dbp_chunk) grouping=chunk ;;
    dbp_stepwise) grouping=stepwise ;;
  esac
  policy_args=(
    "--policy.type=pi05_one_step"
    "--policy.method=${one_step_method}"
    "--policy.pretrained_path=lerobot/pi05_base"
    "--policy.dtype=bfloat16"
    "--policy.gradient_checkpointing=true"
    "--policy.compile_model=false"
    "--policy.drifting_grouping=${grouping}"
    "--policy.test_time_samples=1"
    "--policy.push_to_hub=false"
  )
  case "$one_step_method" in
    dbp_chunk|dbp_stepwise|drifting_perdim)
      policy_args+=(
        "--policy.fresh_action_expert=${FRESH_ACTION_EXPERT:-true}"
        "--policy.train_expert_only=false"
        "--policy.drifting_gen_per_label=${DRIFTING_G:-8}"
        "--policy.drifting_temperatures=[0.02,0.05,0.2]"
      )
      ;;
  esac
fi

run_dir="${output_root}/${mode}/${benchmark}/${method}/seed_${seed}"
log_file="${output_root}/logs/${mode}/${benchmark}/${method}/seed_${seed}.log"
accelerate_cmd=(accelerate launch)
if (( num_processes > 1 )); then
  accelerate_cmd+=(--multi_gpu)
fi
accelerate_cmd+=("--num_processes=${num_processes}")

cmd=(
  "${accelerate_cmd[@]}" -m lerobot.scripts.lerobot_train
  "--batch_size=${batch_size}"
  "--steps=${steps}"
  "--save_checkpoint=true"
  "--save_freq=${save_freq}"
  "--checkpoint_steps=${checkpoint_steps}"
  "--save_checkpoint_to_hub=false"
  "--output_dir=${run_dir}"
  "--job_name=${method}_${benchmark}_seed${seed}"
  "--seed=${seed}"
  "--num_workers=0"
  "--persistent_workers=false"
  "--ddp_broadcast_buffers=false"
  "--ddp_find_unused_parameters=true"
  "--env_eval_freq=0"
  "--eval_steps=0"
  "--log_freq=${log_freq}"
  "--wandb.enable=${wandb_enable}"
  "${policy_args[@]}"
  "${dataset_args[@]}"
)

printf 'REMOTE command:'
printf ' %q' env "CUDA_VISIBLE_DEVICES=${gpu_ids}" "${cmd[@]}"
printf '\n'
[[ "$mode" == dry-run ]] && exit 0
[[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite $run_dir" >&2; exit 1; }
mkdir -p "$(dirname "$log_file")"
CUDA_VISIBLE_DEVICES="$gpu_ids" "${cmd[@]}" 2>&1 | tee "$log_file"
