#!/usr/bin/env bash
# Phase 7C — device e2e + quality gate for TT LM planner (fast gate: 8s / 4 steps).
# Production listen signoff: ACESTEP_RUN_PHASE7C_PROD=1 bash docs/acestep-az-phase7c-run.sh
set -uo pipefail
REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole
export ACESTEP_PIPELINE_DIR="${ACESTEP_PIPELINE_DIR:-/local/ttuser/gtobar/acestep_pipeline}"
export ACESTEP_RUN_PHASE7C="${ACESTEP_RUN_PHASE7C:-1}"
PY="$REPO/python_env/bin/python"
LOG=/tmp/acestep_phase7c.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

REF="${ACESTEP_PHASE7_REF:-/tmp/ref_kaazoom_25s.wav}"
if [[ ! -f "$REF" ]]; then
  echo "ERROR: reference WAV not found: $REF"
  exit 2
fi

echo "=== Phase 7C device gate start $(date -Is) ==="
echo "ref=$REF ACESTEP_RUN_PHASE7C=$ACESTEP_RUN_PHASE7C"

flock /tmp/tt_ace_device.lock "$PY" -m pytest \
  models/tt_dit/tests/models/acestep/test_e2e_lm_planner_acestep.py \
  -s -v --timeout=7200

echo "=== Phase 7C device gate done $(date -Is) ==="
