#!/usr/bin/env bash
# Phase 7 — full production stack: TT LM planner + TT text + TT DiT + TT VAE.
# Host: reference WAV encode (timbre) + LM quantizer. LM replaces Call B tokenizer.
set -uo pipefail
REPO=/local/ttuser/dvartanians/ace/tt-metal
cd "$REPO"
export TT_METAL_HOME="$REPO" PYTHONPATH="$REPO" ARCH_NAME=blackhole
export ACESTEP_PIPELINE_DIR="${ACESTEP_PIPELINE_DIR:-/local/ttuser/gtobar/acestep_pipeline}"
PY="$REPO/python_env/bin/python"
LOG=/tmp/acestep_phase7.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

REF="${ACESTEP_PHASE7_REF:-/tmp/ref_kaazoom_25s.wav}"
OUT="${ACESTEP_PHASE7_OUT:-/tmp/az_lm_tt.wav}"
PROMPT="${ACESTEP_PHASE7_PROMPT:-smooth jazz pop, female lead vocal, warm piano, soft drums, lounge, 90 bpm}"
LM_MODEL="${ACESTEP_PHASE7_LM_MODEL:-1.7B}"

if [[ ! -f "$REF" ]]; then
  echo "ERROR: reference WAV not found: $REF"
  echo "Set ACESTEP_PHASE7_REF to a timbre reference WAV."
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

if [[ -n "${ACESTEP_PHASE7_LYRICS:-}" ]]; then
  LYRICS="$ACESTEP_PHASE7_LYRICS"
fi

echo "=== Phase 7 full production demo start $(date -Is) ==="
echo "ref=$REF out=$OUT lm_model=$LM_MODEL"

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
  --use-lm-planner \
  --use-tt-lm-planner \
  --lm-model "$LM_MODEL" \
  --use-tt-vae \
  --use-tt-text-encode \
  --no-traced

if [[ -f "$OUT" ]]; then
  echo "=== Phase 7 SIGNOFF WAV: $OUT ==="
  echo "Listen: aplay $OUT"
  echo "=== Phase 7 PASS $(date -Is) ==="
  exit 0
fi

echo "=== Phase 7 FAIL — no output WAV $(date -Is) ==="
exit 1
