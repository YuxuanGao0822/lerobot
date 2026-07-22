#!/usr/bin/env bash

# Run the three prescribed seeds sequentially for one model/benchmark pair.
# Each child run occupies all configured GPUs, so seeds are intentionally not
# launched concurrently.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash experiments/driftingvla/baselines/launch_group.sh MODEL BENCHMARK" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model=$1
benchmark=$2
seeds=${SEEDS:-"1000 1001 1002"}

for seed in $seeds; do
  bash "${script_dir}/run_one.sh" "$model" "$benchmark" "$seed"
done

