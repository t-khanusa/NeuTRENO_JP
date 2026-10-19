#!/usr/bin/env bash
# Grid search for the Input-Anchored ViT (P-ONLY controller).
#
# The single effective knob is the retention rate g = anchor_k_p (authority=1).
# g -> 0 : pure ViT (collapses).   g -> 1 : strongly anchored to the input.
# We also vary the structural knobs anchor_tokens {patch, all} and readout {cls, mean}.
#
# Runs are SEQUENTIAL (one GPU). Each logs to wandb project "PID-Anchor" with a
# name encoding its config. Edit the arrays below to change the grid.
#
# Usage:
#   bash grid_anchor.sh                 # default grid, EPOCHS=200
#   EPOCHS=100 GPU=0 bash grid_anchor.sh
set -euo pipefail

PY=/opt/anaconda/envs/mambavision/bin/python
CFG="--config-path configs --config-name pid_anchor_p"
EPOCHS="${EPOCHS:-200}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

# ── Primary sweep: retention rate g (P gain), tokens=patch, readout=cls ───────
KP_GRID=(0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05)
for kp in "${KP_GRID[@]}"; do
  echo "=== [grid] P-only  g=${kp}  tokens=patch  readout=cls ==="
  $PY training_pid_anchor.py $CFG \
      epochs="$EPOCHS" anchor_enabled=true \
      anchor_k_p="$kp" anchor_k_i=0.0 anchor_k_d=0.0 anchor_authority=1.0 \
      anchor_tokens=patch readout=cls
done

# ── Structural variants at a mid gain (edit BEST_KP after the primary sweep) ──
BEST_KP="${BEST_KP:-0.2}"
for tok in patch all; do
  for ro in cls mean; do
    [ "$tok" = patch ] && [ "$ro" = cls ] && continue   # already run above
    echo "=== [grid] P-only  g=${BEST_KP}  tokens=${tok}  readout=${ro} ==="
    $PY training_pid_anchor.py $CFG \
        epochs="$EPOCHS" anchor_enabled=true \
        anchor_k_p="$BEST_KP" anchor_k_i=0.0 anchor_k_d=0.0 anchor_authority=1.0 \
        anchor_tokens="$tok" readout="$ro"
  done
done

# ── Control: no anchor (expected to collapse) ────────────────────────────────
echo "=== [grid] baseline: anchor OFF ==="
$PY training_pid_anchor.py $CFG epochs="$EPOCHS" anchor_enabled=false

echo "=== [grid] done ==="
