#!/usr/bin/env bash
# Attach log tail watchers to acestep-az tmux windows (safe parallel agents only).
set -euo pipefail
SESSION=acestep-az
REPO=/local/ttuser/dvartanians/ace/tt-metal
WATCH="$REPO/docs/acestep-az-window-watch.sh"
SETUP='export REPO=/local/ttuser/dvartanians/ace/tt-metal; cd "$REPO"; export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole'

start_watch() {
  local win="$1" arg="$2"
  tmux send-keys -t "$SESSION:$win" C-c 2>/dev/null || true
  tmux send-keys -t "$SESSION:$win" "$SETUP" C-m
  tmux send-keys -t "$SESSION:$win" "bash $WATCH $arg" C-m
}

# Parallel-safe windows — tail agent logs
start_watch phase2a-text 2a
start_watch phase2b-ref 2b
start_watch phase4-vae 4
start_watch phase3-sampler 3
start_watch phase5-demo 5

# Idle / serial windows — status only
start_watch phase1 phase1
start_watch device device
start_watch m0-trace m0

echo "Tmux windows wired to agent logs. Use: az go 2a && tail continues there"
