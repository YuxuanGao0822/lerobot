#!/usr/bin/env bash

# Launch one non-drifting VLA post-training run. This file is intended for the
# remote training server only; it is safe to inspect or invoke with MODE=dry-run
# without loading a model or dataset.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash experiments/driftingvla/baselines/run_one.sh MODEL BENCHMARK SEED

MODEL:      pi0 | pi05 | smolvla | xvla | groot
BENCHMARK:  libero | robotwin
SEED:       integer, normally 1000, 1001, or 1002

Environment variables:
  MODE             dry-run (default), smoke, or train
  NUM_PROCESSES    number of DDP workers (default: 8)
  GPU_IDS          CUDA device list (default: 0,1,2,3,4,5,6,7)
  BATCH_SIZE       per-process batch size (default: 4; global batch 32 on 8 GPUs)
  OUTPUT_ROOT      output root (default: outputs/baselines)
  RUN_STEPS        override 2-step smoke or 50k train target (diagnostics only)
  SAVE_FREQ_OVERRIDE override smoke/train checkpoint frequency (diagnostics only)
  WANDB_ENABLE     true or false (default: false)
  NUM_WORKERS      workers per process (default: 0 for NCCL fork safety)
  PREFETCH_FACTOR  dataloader prefetch factor (default: 2)
  HF_HOME          optional Hugging Face cache location
  ROBOTWIN_DATASET_ROOT optional prepared RoboTwin dataset root with q01/q99 stats

Examples:
  MODE=dry-run bash experiments/driftingvla/baselines/run_one.sh pi05 libero 1000
  MODE=smoke bash experiments/driftingvla/baselines/run_one.sh pi05 libero 1000
  MODE=train WANDB_ENABLE=true bash experiments/driftingvla/baselines/run_one.sh pi05 libero 1000
EOF
}

if [[ $# -ne 3 ]]; then
  usage >&2
  exit 2
fi

model=$1
benchmark=$2
seed=$3

case "$model" in
  pi0|pi05|smolvla|xvla|groot) ;;
  *) echo "Unsupported MODEL: $model" >&2; usage >&2; exit 2 ;;
esac

case "$benchmark" in
  libero|robotwin) ;;
  *) echo "Unsupported BENCHMARK: $benchmark" >&2; usage >&2; exit 2 ;;
esac

if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer: $seed" >&2
  exit 2
fi

mode=${MODE:-dry-run}
case "$mode" in
  dry-run|smoke|train) ;;
  *) echo "MODE must be dry-run, smoke, or train: $mode" >&2; exit 2 ;;
esac

num_processes=${NUM_PROCESSES:-8}
gpu_ids=${GPU_IDS:-0,1,2,3,4,5,6,7}
batch_size=${BATCH_SIZE:-4}
output_root=${OUTPUT_ROOT:-outputs/baselines}
wandb_enable=${WANDB_ENABLE:-false}
num_workers=${NUM_WORKERS:-0}
prefetch_factor=${PREFETCH_FACTOR:-2}
ddp_find_unused_parameters=true
if [[ "$model" == "xvla" ]]; then
  # X-VLA's full training graph uses every trainable parameter on each batch.
  # Its 8-GPU smoke run confirmed that unused-parameter discovery only adds an
  # extra autograd-graph traversal and can be disabled safely.
  ddp_find_unused_parameters=false
fi

if [[ "$mode" == "train" ]]; then
  steps=50000
  save_freq=10000
  log_freq=100
else
  steps=2
  save_freq=2
  log_freq=1
fi
steps=${RUN_STEPS:-$steps}
save_freq=${SAVE_FREQ_OVERRIDE:-$save_freq}

if [[ "$mode" == "train" && -z "${RUN_STEPS:-}" && -z "${SAVE_FREQ_OVERRIDE:-}" ]]; then
  checkpoint_steps='[30000,40000,50000]'
elif [[ "$mode" == "smoke" ]]; then
  checkpoint_steps="[${steps}]"
else
  # Diagnostic runs use periodic saves so a temporary recovery point can be
  # written without changing the formal 30k/40k/50k artifact contract.
  checkpoint_steps='[]'
fi

dataset_repo=lerobot/libero
dataset_args=()
rename_args=()
if [[ "$benchmark" == "robotwin" ]]; then
  dataset_repo=lerobot/robotwin_unified
  # The public RoboTwin dataset has no LeRobot `v3.0` tag. Pin the current Hub
  # commit explicitly so metadata resolution bypasses tag lookup and all paper
  # runs consume exactly the same 79.5 GB snapshot.
  dataset_args=(
    "--dataset.revision=1287871839fae2296bc27b88a5457c3e1eba8e1f"
  )
  hf_lerobot_home=${HF_LEROBOT_HOME:-${HF_HOME:-${HOME}/.cache/huggingface}/lerobot}
  robotwin_dataset_root=${ROBOTWIN_DATASET_ROOT:-${hf_lerobot_home}/derived/robotwin_unified_quantiles_1287871839fa}
  if [[ "$mode" != "dry-run" ]]; then
    if [[ ! -f "${robotwin_dataset_root}/meta/stats.json" ]]; then
      echo "RoboTwin quantile overlay has no meta/stats.json: ${robotwin_dataset_root}" >&2
      echo "Run prepare_robotwin_quantile_overlay.py first or set ROBOTWIN_DATASET_ROOT." >&2
      exit 2
    fi
    if [[ ! -f "${robotwin_dataset_root}/robotwin_quantile_overlay.json" ]]; then
      echo "RoboTwin root is not a reviewed quantile overlay: ${robotwin_dataset_root}" >&2
      echo "Refusing to train with missing or unverified q01/q99 metadata." >&2
      exit 2
    fi
  fi
  dataset_args+=("--dataset.root=${robotwin_dataset_root}")

  if [[ "$model" == "xvla" ]]; then
    # RoboTwin's dataset follows physical camera names, while the released
    # X-VLA processor declares three generic visual slots. Keep the mapping
    # local to this model/dataset pair so every other baseline retains its
    # native feature contract.
    rename_args=(
      '--rename_map={"observation.images.cam_high":"observation.images.image","observation.images.cam_left_wrist":"observation.images.image2","observation.images.cam_right_wrist":"observation.images.image3"}'
    )
  fi
fi

run_dir="${output_root}/${mode}/${benchmark}/${model}/seed_${seed}"
log_dir="${output_root}/logs/${mode}/${benchmark}/${model}"
log_file="${log_dir}/seed_${seed}.log"

common_args=(
  "--dataset.repo_id=${dataset_repo}"
  "--policy.push_to_hub=false"
  "--batch_size=${batch_size}"
  "--steps=${steps}"
  "--save_checkpoint=true"
  "--save_freq=${save_freq}"
  "--checkpoint_steps=${checkpoint_steps}"
  "--save_checkpoint_to_hub=false"
  "--output_dir=${run_dir}"
  "--job_name=${model}_${benchmark}_seed${seed}"
  "--seed=${seed}"
  "--ddp_broadcast_buffers=false"
  "--ddp_find_unused_parameters=${ddp_find_unused_parameters}"
  "--env_eval_freq=0"
  "--eval_steps=0"
  "--log_freq=${log_freq}"
  "--num_workers=${num_workers}"
  "--prefetch_factor=${prefetch_factor}"
  "--persistent_workers=false"
  "--wandb.enable=${wandb_enable}"
  "--wandb.disable_artifact=true"
)

policy_args=()
case "$model" in
  pi0)
    policy_args=(
      "--policy.type=pi0"
      "--policy.pretrained_path=lerobot/pi0_base"
      "--policy.dtype=bfloat16"
      "--policy.gradient_checkpointing=true"
      "--policy.compile_model=false"
      "--policy.freeze_vision_encoder=false"
      "--policy.train_expert_only=false"
    )
    ;;
  pi05)
    policy_args=(
      "--policy.type=pi05"
      "--policy.pretrained_path=lerobot/pi05_base"
      "--policy.dtype=bfloat16"
      "--policy.gradient_checkpointing=true"
      "--policy.compile_model=false"
      "--policy.freeze_vision_encoder=false"
      "--policy.train_expert_only=false"
    )
    ;;
  smolvla)
    # Keep SmolVLA's released post-training recipe explicit. Its Drift partner
    # must use exactly the same frozen/trainable partition for a fair ablation.
    policy_args=(
      "--policy.type=smolvla"
      "--policy.pretrained_path=lerobot/smolvla_base"
      "--policy.compile_model=false"
      "--policy.freeze_vision_encoder=true"
      "--policy.train_expert_only=true"
      "--policy.train_state_proj=true"
    )
    ;;
  xvla)
    domain_id=3
    if [[ "$benchmark" == "robotwin" ]]; then
      domain_id=6
    fi
    policy_args=(
      # X-VLA's nested Florence architecture is stored in the checkpoint
      # config. `policy.path` loads it first and then applies these CLI
      # post-training overrides; constructing `policy.type=xvla` from defaults
      # would leave `florence_config` empty.
      "--policy.path=lerobot/xvla-base"
      "--policy.dtype=bfloat16"
      "--policy.action_mode=auto"
      "--policy.domain_id=${domain_id}"
      "--policy.freeze_vision_encoder=false"
      "--policy.freeze_language_encoder=false"
      "--policy.train_policy_transformer=true"
      "--policy.train_soft_prompts=true"
    )
    ;;
  groot)
    embodiment_tag=new_embodiment
    if [[ "$benchmark" == "libero" ]]; then
      embodiment_tag=libero_sim
    fi
    policy_args=(
      "--policy.type=groot"
      "--policy.base_model_path=nvidia/GR00T-N1.7-3B"
      "--policy.embodiment_tag=${embodiment_tag}"
      "--policy.use_bf16=true"
      "--policy.model_params_fp32=true"
      "--policy.tune_llm=false"
      "--policy.tune_visual=false"
      "--policy.tune_projector=true"
      "--policy.tune_diffusion_model=true"
      "--policy.tune_vlln=true"
      "--policy.max_steps=${steps}"
    )
    ;;
esac

train_entry=$(command -v lerobot-train || true)
if [[ -z "$train_entry" ]]; then
  train_entry=lerobot-train
fi

cmd=(
  accelerate launch
  --multi_gpu
  "--num_processes=${num_processes}"
  "$train_entry"
  "${common_args[@]}"
  "${policy_args[@]}"
  "${dataset_args[@]}"
  "${rename_args[@]}"
)

printf 'Run: model=%s benchmark=%s seed=%s mode=%s\n' "$model" "$benchmark" "$seed" "$mode"
printf 'Dataset: %s\n' "$dataset_repo"
printf 'DDP: %s processes; per-process batch=%s; global batch=%s\n' \
  "$num_processes" "$batch_size" "$((num_processes * batch_size))"
printf 'Output: %s\n' "$run_dir"
printf 'Command:'
printf ' %q' env "CUDA_VISIBLE_DEVICES=${gpu_ids}" "${cmd[@]}"
printf '\n'

if [[ "$mode" == "dry-run" ]]; then
  exit 0
fi

if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing run directory: $run_dir" >&2
  echo "Resume it with resume_one.sh or choose a different OUTPUT_ROOT." >&2
  exit 1
fi

mkdir -p "$log_dir"
CUDA_VISIBLE_DEVICES="$gpu_ids" "${cmd[@]}" 2>&1 | tee "$log_file"
