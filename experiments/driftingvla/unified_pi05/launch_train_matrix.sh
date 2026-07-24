#!/usr/bin/env bash

# REMOTE SERVER ONLY. Sequentially dispatch the controlled π0.5 training matrix.
# Each run uses all GPUs configured by run_train.sh; runs are deliberately not
# launched concurrently on the same device set.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${MODE:-dry-run}
methods=${METHODS:-"pi05_flow pi05_dbp_chunk pi05_dbp_stepwise pi05_driftingvla"}
benchmarks=${BENCHMARKS:-"libero robotwin"}
seeds=${SEEDS:-"1000 1001 1002"}

case "$mode" in dry-run|smoke|train) ;; *) echo "MODE must be dry-run, smoke, or train" >&2; exit 2 ;; esac

planned=0
for benchmark in $benchmarks; do
  for method in $methods; do
    for seed in $seeds; do
      planned=$((planned + 1))
      printf '\n[%d] method=%s benchmark=%s seed=%s mode=%s\n' \
        "$planned" "$method" "$benchmark" "$seed" "$mode"
      MODE="$mode" bash "$script_dir/run_train.sh" "$method" "$benchmark" "$seed"
    done
  done
done

printf '\nMatrix dispatch finished: planned=%d mode=%s\n' "$planned" "$mode"
printf '%s\n' "Naive π0.5 one-step is evaluation-only and therefore has no training row."
