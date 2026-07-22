#!/usr/bin/env bash

# Read-only audit for the expected 5 x 2 x 3 x 3 baseline checkpoint matrix.

set -euo pipefail

output_root=${OUTPUT_ROOT:-outputs/baselines}
models=(pi0 pi05 smolvla xvla groot)
benchmarks=(libero robotwin)
seeds=(1000 1001 1002)
steps=(020000 040000 060000)
missing=0
found=0

for model in "${models[@]}"; do
  for benchmark in "${benchmarks[@]}"; do
    for seed in "${seeds[@]}"; do
      for step in "${steps[@]}"; do
        checkpoint="${output_root}/train/${benchmark}/${model}/seed_${seed}/checkpoints/${step}"
        required=(
          "${checkpoint}/pretrained_model/config.json"
          "${checkpoint}/pretrained_model/model.safetensors"
          "${checkpoint}/pretrained_model/train_config.json"
          "${checkpoint}/training_state/optimizer_state.safetensors"
          "${checkpoint}/training_state/training_step.json"
        )
        ok=true
        for path in "${required[@]}"; do
          if [[ ! -f "$path" ]]; then
            printf 'MISSING\t%s\n' "$path"
            ok=false
            missing=$((missing + 1))
          fi
        done
        if [[ "$ok" == true ]]; then
          size=$(du -sh "$checkpoint" | awk '{print $1}')
          printf 'OK\t%s\t%s\n' "$checkpoint" "$size"
          found=$((found + 1))
        fi
      done
    done
  done
done

printf 'Summary: complete_checkpoints=%s/90 missing_required_files=%s\n' "$found" "$missing"
if ((missing > 0)); then
  exit 1
fi

