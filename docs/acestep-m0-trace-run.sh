#!/usr/bin/env bash
# Phase 6 — trace + 2-CQ enablement and perf baseline.
# Run only after Phases 1–5 functional gates pass, or in dedicated m0-trace tmux window.
set -uo pipefail
REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole
PY="$REPO/python_env/bin/python"
LOG=/tmp/acestep_m0_trace.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=== Phase 6 trace enablement start $(date -Is) ==="
FAIL=0

run_dev() {
  echo "--- $1 ---"
  flock /tmp/tt_ace_device.lock "$PY" -m pytest "${@:2}" || FAIL=1
}

run_dev "1/3 traced latents e2e" \
  models/demos/hf_eager/acestep_v15_base/tests/e2e/test_e2e_generate_audio_traced.py \
  -s -v --timeout=900

run_dev "2/3 traced latents perf (4 steps)" \
  models/tt_dit/tests/models/acestep/test_e2e_perf_traced_acestep.py \
  -k 1x1 -s -v --timeout=3600

run_dev "3/3 traced music perf (4 steps)" \
  models/tt_dit/tests/models/acestep/test_e2e_music_perf_traced_acestep.py \
  -k 1x1 -s -v --timeout=3600

if [[ "$FAIL" -eq 0 ]]; then
  echo "=== Phase 6 trace PASS $(date -Is) ==="
  exit 0
else
  echo "=== Phase 6 trace FAIL $(date -Is) ==="
  exit 1
fi
