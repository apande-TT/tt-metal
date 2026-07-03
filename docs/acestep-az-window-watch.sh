#!/usr/bin/env bash
# Tail agent log in a tmux window. Usage: acestep-az-window-watch.sh <2a|2b|3|4|5|device|m0|phase1>
set -euo pipefail
REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole

case "${1:-help}" in
  2a|phase2a) LOG=/tmp/acestep_agent_2a.log; TITLE="Phase 2A agent — live text" ;;
  2b|phase2b) LOG=/tmp/acestep_agent_2b.log; TITLE="Phase 2B agent — ref audio" ;;
  3|phase3)     LOG=/tmp/acestep_agent_3.log; TITLE="Phase 3 agent — CFG/APG prep" ;;
  4|phase4|vae) LOG=/tmp/acestep_agent_4.log; TITLE="Phase 4 agent — TT VAE decoder" ;;
  5|phase5)     LOG=/tmp/acestep_agent_5.log; TITLE="Phase 5 agent — CLI skeleton" ;;
  device)       LOG=/tmp/acestep_agent_device.log; TITLE="Device window — SERIAL pytest only" ;;
  m0|trace)     LOG=/tmp/acestep_agent_m0.log; TITLE="Phase 6 trace — DEFERRED" ;;
  phase1|1)     LOG=/tmp/acestep_phase1.log; TITLE="Phase 1 G0 — baseline log" ;;
  *)
    echo "Usage: $0 {2a|2b|3|4|5|device|m0|phase1}"
    exit 1
    ;;
esac

: > "$LOG"
{
  echo "=== $TITLE ==="
  echo "Started: $(date -Is)"
  echo "Log: $LOG"
  echo ""
} >> "$LOG"

clear
echo "=== $TITLE ==="
echo "Log: $LOG"
echo "Press Ctrl-C to stop tail (agent keeps running in Cursor)."
echo ""
tail -f "$LOG"
