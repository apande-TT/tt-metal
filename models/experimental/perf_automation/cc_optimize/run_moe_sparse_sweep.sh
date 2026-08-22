#!/bin/bash
# SEPARATE pre-pass for a Mixture-of-Experts model: sweep the sparse expert GEMM
# (cores x in0_block_w x out_subblock_w x obw_mult) and write a CSV of every
# PCC-passing config, fastest first. Run this BEFORE the optimize loop. It does
# not invoke or modify the optimize tool -- it is a standalone step whose CSV you
# inspect, or seed the run's matmul buckets from.
#
# Read GUIDELINES/13_MOE_SPARSE_EXPERTS.md first. Sections 1-5 explain the traps:
# score per tile against the DEVICE osw=1 output, never a torch reference, and do
# not trust a delta the model cannot reproduce in place.
#
# Usage:
#   run_moe_sparse_sweep.sh gate_up  <K> <N> [experts] [active]
#   run_moe_sparse_sweep.sh down     <K> <N> [experts] [active]
# Example (gpt-oss-20b, 32 experts top-4, hidden 2880):
#   run_moe_sparse_sweep.sh gate_up 2880 5760 32 4
#   run_moe_sparse_sweep.sh down    2880 2880 32 4
#
# Optional env: MOE_SWEEP_CSV (output path), MOE_SWEEP_ITERS (timed reps, def 3),
#   MOE_SWEEP_GRID ("9 5" to pin one grid), MOE_SWEEP_PCC (min pcc), MOE_SWEEP_EXTRA.
set -euo pipefail

PROJ="${1:?usage: run_moe_sparse_sweep.sh <gate_up|down> <K> <N> [experts] [active]}"
K="${2:?missing K (reduction depth, e.g. 2880)}"
N="${3:?missing N (output width, e.g. 5760 for a fused [gate|up])}"
EXPERTS="${4:-32}"
ACTIVE="${5:-4}"

# down takes the ACTIVATION as the sparse operand; gate_up takes the WEIGHT.
case "$PROJ" in
  gate_up) SPARSE_INPUT="b" ;;
  down)    SPARSE_INPUT="a" ;;
  *) echo "proj must be gate_up or down, got '$PROJ'" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PKG="$REPO_ROOT/models/experimental/perf_automation"
cd "$REPO_ROOT"

ARGS=(--proj "$PROJ" --K "$K" --N "$N" --experts "$EXPERTS" --active "$ACTIVE" --sparse-input "$SPARSE_INPUT")
ARGS+=(--iters "${MOE_SWEEP_ITERS:-3}")
ARGS+=(--csv "${MOE_SWEEP_CSV:-moe_sparse_sweep_${PROJ}.csv}")
[ -n "${MOE_SWEEP_GRID:-}" ] && ARGS+=(--grid ${MOE_SWEEP_GRID})
[ -n "${MOE_SWEEP_PCC:-}" ] && ARGS+=(--pcc "$MOE_SWEEP_PCC")
[ -n "${MOE_SWEEP_EXTRA:-}" ] && ARGS+=(${MOE_SWEEP_EXTRA})

echo "[moe-sparse-sweep] $PROJ  K=$K N=$N experts=$EXPERTS active=$ACTIVE sparse-input=$SPARSE_INPUT"
PYTHONPATH="$REPO_ROOT:$PKG" \
  python "$PKG/cc_optimize/moe_sparse_matmul_sweep.py" "${ARGS[@]}"
echo "[moe-sparse-sweep] done -- validate the winner IN THE MODEL before keeping it (13 section 1)."
