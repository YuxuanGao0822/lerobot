#!/usr/bin/env bash

# Launch the complete baseline matrix sequentially. Each child run consumes all
# configured GPUs, so this script never overlaps training jobs. Completed formal
# runs are skipped; incomplete run directories stop the matrix and must be
# explicitly resumed or archived by the user.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mode=${MODE:-dry-run}
output_root=${OUTPUT_ROOT:-outputs/baselines}
models=${MODELS:-"pi05 pi0 smolvla xvla groot"}
benchmarks=${BENCHMARKS:-"libero robotwin"}
seeds=${SEEDS:-"1000 1001 1002"}
skip_complete=${SKIP_COMPLETE:-true}

case "$mode" in
  dry-run|smoke|train) ;;
  *) echo "MODE must be dry-run, smoke, or train: $mode" >&2; exit 2 ;;
esac

case "$skip_complete" in
  true|false) ;;
  *) echo "SKIP_COMPLETE must be true or false: $skip_complete" >&2; exit 2 ;;
esac

if [[ "$mode" == "train" && ( -n "${RUN_STEPS:-}" || -n "${SAVE_FREQ_OVERRIDE:-}" ) ]]; then
  echo "Formal matrix runs forbid RUN_STEPS and SAVE_FREQ_OVERRIDE." >&2
  echo "Use run_one.sh for diagnostic-length training." >&2
  exit 2
fi

checkpoint_complete() {
  local checkpoint=$1
  local required

  for required in \
    "${checkpoint}/pretrained_model/config.json" \
    "${checkpoint}/pretrained_model/model.safetensors" \
    "${checkpoint}/pretrained_model/policy_preprocessor.json" \
    "${checkpoint}/pretrained_model/policy_postprocessor.json" \
    "${checkpoint}/pretrained_model/train_config.json" \
    "${checkpoint}/training_state/optimizer_state.safetensors" \
    "${checkpoint}/training_state/training_step.json"; do
    [[ -f "$required" ]] || return 1
  done
}

formal_run_complete() {
  local run_dir=$1
  local step
  local checkpoint

  for step in 030000 040000 050000; do
    checkpoint="${run_dir}/checkpoints/${step}"
    checkpoint_complete "$checkpoint" || return 1
  done
}

planned=0
launched=0
skipped=0
read -r -a model_list <<< "$models"
read -r -a benchmark_list <<< "$benchmarks"
read -r -a seed_list <<< "$seeds"
total=$((${#model_list[@]} * ${#benchmark_list[@]} * ${#seed_list[@]}))

for model in $models; do
  for benchmark in $benchmarks; do
    for seed in $seeds; do
      planned=$((planned + 1))
      run_dir="${output_root}/${mode}/${benchmark}/${model}/seed_${seed}"

      if [[ "$skip_complete" == "true" && "$mode" != "dry-run" ]]; then
        run_complete=false
        if [[ "$mode" == "train" ]] && formal_run_complete "$run_dir"; then
          run_complete=true
        elif [[ "$mode" == "smoke" ]] && checkpoint_complete "${run_dir}/checkpoints/000002"; then
          run_complete=true
        fi

        if [[ "$run_complete" == "true" ]]; then
          printf 'SKIP complete: model=%s benchmark=%s seed=%s\n' "$model" "$benchmark" "$seed"
          skipped=$((skipped + 1))
          continue
        fi
        if [[ -e "$run_dir" ]]; then
          echo "Incomplete $mode run already exists: $run_dir" >&2
          if [[ "$mode" == "train" ]]; then
            echo "Inspect it, then resume with:" >&2
            echo "  OUTPUT_ROOT=$output_root bash ${script_dir}/resume_one.sh $model $benchmark $seed" >&2
          fi
          echo "Or archive it before restarting this matrix." >&2
          exit 1
        fi
      fi

      printf 'LAUNCH %s/%s: model=%s benchmark=%s seed=%s mode=%s\n' \
        "$planned" "$total" "$model" "$benchmark" "$seed" "$mode"
      MODE="$mode" OUTPUT_ROOT="$output_root" \
        bash "${script_dir}/run_one.sh" "$model" "$benchmark" "$seed"
      launched=$((launched + 1))
    done
  done
done

printf 'Matrix finished: planned=%s launched=%s skipped_complete=%s\n' \
  "$planned" "$launched" "$skipped"
