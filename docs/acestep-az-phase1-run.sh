#!/usr/bin/env bash
# Phase 1 G0 gate — functional baseline, no trace dependency.
# Traced perf moved to Phase 6 (docs/acestep-m0-trace-run.sh).
set -uo pipefail
REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole
PY="$REPO/python_env/bin/python"
LOG=/tmp/acestep_phase1.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=== Phase 1 G0 start $(date -Is) (no trace) ==="
FAIL=0

run_host() {
  echo "--- $1 ---"
  "$PY" -m pytest "${@:2}" || FAIL=1
}

run_dev() {
  echo "--- $1 ---"
  flock /tmp/tt_ace_device.lock "$PY" -m pytest "${@:2}" || FAIL=1
}

run_host "1/3 host vae golden (no device)" \
  models/demos/hf_eager/acestep_v15_base/tests/e2e/test_e2e_generate_audio_with_vae_host.py \
  -k golden_latents -s -v --timeout=300

run_dev "2/3 eager e2e latents (non-traced)" \
  models/demos/hf_eager/acestep_v15_base/tests/e2e/test_e2e_generate_audio.py \
  -s -v --timeout=900

run_dev "3/3 tt latents + host vae" \
  models/demos/hf_eager/acestep_v15_base/tests/e2e/test_e2e_generate_audio_with_vae_host.py \
  -k tt_latents_host_vae -s -v --timeout=900

if [[ "$FAIL" -eq 0 ]]; then
  echo "=== Phase 1 G0 PASS $(date -Is) ==="
  echo "Next: Phase 2A/2B in parallel; traced perf deferred to Phase 6."
  exit 0
else
  echo "=== Phase 1 G0 FAIL $(date -Is) ==="
  exit 1
fi
