#!/usr/bin/env bash
# Phase 8 — traced full stack: TT LM + traced DiT + TT VAE (CFG off when traced).
set -uo pipefail
REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole
export ACESTEP_PIPELINE_DIR="${ACESTEP_PIPELINE_DIR:-/local/ttuser/gtobar/acestep_pipeline}"
export ACESTEP_RUN_PHASE8="${ACESTEP_RUN_PHASE8:-1}"
PY="$REPO/python_env/bin/python"
LOG=/tmp/acestep_phase8.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

REF="${ACESTEP_PHASE8_REF:-/tmp/ref_kaazoom_25s.wav}"
if [[ ! -f "$REF" ]]; then
  echo "ERROR: reference WAV not found: $REF"
  exit 2
fi

echo "=== Phase 8 traced full-stack start $(date -Is) ==="

flock /tmp/tt_ace_device.lock "$PY" -m pytest \
  models/tt_dit/tests/models/acestep/test_e2e_traced_full_stack_acestep.py \
  -s -v --timeout=7200

echo "=== Phase 8 traced full-stack done $(date -Is) ==="
