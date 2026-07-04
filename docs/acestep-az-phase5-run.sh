#!/usr/bin/env bash
# Phase 5 — full TT A→Z demo signoff (TT text + TT DiT + TT VAE, production sampler).
# Host-only: reference WAV encode (timbre). Tokenizer path B+D (no LM planner).
set -uo pipefail
REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole
PY="$REPO/python_env/bin/python"
LOG=/tmp/acestep_phase5.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

REF="${ACESTEP_PHASE5_REF:-/tmp/ref_kaazoom_25s.wav}"
OUT="${ACESTEP_PHASE5_OUT:-/tmp/az_phase5_signoff.wav}"
PROMPT="${ACESTEP_PHASE5_PROMPT:-smooth jazz pop, female lead vocal, warm piano, soft drums, lounge, 90 bpm}"

if [[ ! -f "$REF" ]]; then
  echo "ERROR: reference WAV not found: $REF"
  echo "Set ACESTEP_PHASE5_REF to a timbre reference WAV."
  exit 2
fi

read -r -d '' LYRICS <<'EOF' || true
[verse]
City lights are fading slow
Warm piano starts to glow
Soft drums keep the time so low
In this lounge where feelings flow
[chorus]
Stay with me tonight
Under neon light
Smooth jazz in the air
Like we haven't got a care
EOF

if [[ -n "${ACESTEP_PHASE5_LYRICS:-}" ]]; then
  LYRICS="$ACESTEP_PHASE5_LYRICS"
fi

echo "=== Phase 5 full TT A→Z start $(date -Is) ==="
echo "ref=$REF out=$OUT"

flock /tmp/tt_ace_device.lock "$PY" -m models.tt_dit.pipelines.acestep.demo_acestep_az \
  --prompt "$PROMPT" \
  --lyrics "$LYRICS" \
  --reference "$REF" \
  --output "$OUT" \
  --audio-duration 30 \
  --infer-steps 30 \
  --guidance-scale 7.0 \
  --shift 3.0 \
  --seed 42 \
  --use-tt-vae \
  --use-tt-text-encode \
  --no-traced

if [[ -f "$OUT" ]]; then
  echo "=== Phase 5 SIGNOFF WAV: $OUT ==="
  echo "Listen: aplay $OUT"
  echo "=== Phase 5 PASS $(date -Is) ==="
  exit 0
fi

echo "=== Phase 5 FAIL — no output WAV $(date -Is) ==="
exit 1
