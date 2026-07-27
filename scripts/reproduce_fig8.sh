#!/usr/bin/env bash
# Reproduce the BO trajectories in Figure 8 of the paper.
#
# Usage: bash scripts/reproduce_fig8.sh [METHOD] [BENCHMARK] [REP]
#
# Defaults loop over all five methods and all six benchmarks for one
# repetition; pass arguments to run a single configuration.  The bencher
# service must be running on localhost:50051.
#
# Iteration counts and initial-design sizes match the paper's evaluation
# protocol: n_init = floor(3 * sqrt(d)), n_iter according to the paper's
# per-benchmark budgets (see the figure captions).

set -euo pipefail

METHODS=${1:-"msr mle_scaled mle_ln2 mle_ln2_raasp dsp"}
BENCHMARKS=${2:-"mopta08 lasso-dna svm mujoco-ant mujoco-humanoid rover"}
REP=${3:-1}

declare -A N_ITER=(
  [mopta08]=1000
  [lasso-dna]=1000
  [svm]=1000
  [mujoco-ant]=1000
  [mujoco-humanoid]=400
  [rover]=1000
)

for method in $METHODS; do
  for bench in $BENCHMARKS; do
    n_iter=${N_ITER[$bench]}
    out="runs/fig8/${bench}_${method}_${REP}"
    if [ -f "${out}/best_so_far.npy" ]; then
      echo "[skip] ${out} already exists"
      continue
    fi
    echo "[run]  ${method} on ${bench} (n_iter=${n_iter}, rep=${REP})"
    python -m understanding_hdbo.run \
        --objective bencher --benchmark "$bench" \
        --method "$method" --n-iter "$n_iter" \
        --seed "$REP" --out "$out"
  done
done
