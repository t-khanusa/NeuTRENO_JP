#!/usr/bin/env bash
# Grid for screened-diffusion JEPA: sweep the sample-axis fidelity λ_s.
#
# Theory predicts an inverted-U of accuracy vs λ_s:
#   λ_s → 0   : pure diffusion, collapse (baseline).
#   λ_s → ∞   : representation pinned to raw random features.
#   sweet spot: random features filtered on the augmentation graph.
# Phase 2 runs the cascade (token-axis NeuTRENO λ_p=0.6) at every λ_s to test
# whether two-axis control compounds like it did for SIGReg (+2x speed).
#
# Usage:
#   bash grid_tether.sh                 # EPOCHS=200, GPU=0
#   EPOCHS=100 GPU=1 bash grid_tether.sh
set -euo pipefail

PY=/opt/anaconda/envs/mambavision/bin/python
CFG="--config-path configs --config-name tether"
EPOCHS="${EPOCHS:-200}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

LS_GRID=(0.03 0.1 0.3 1.0 3.0)

# ── Phase 1: tether alone (does the sample-axis anchor prevent collapse?) ──
for ls in "${LS_GRID[@]}"; do
  echo "=== [grid] tether λ_s=${ls}  λ_p=0.0 ==="
  $PY training_tether.py $CFG epochs="$EPOCHS" tether_lambda="$ls" lambda_p=0.0
done

# ── Control: pure invariance (expected to collapse) ────────────────────────
echo "=== [grid] baseline λ_s=0 ==="
$PY training_tether.py $CFG epochs="$EPOCHS" tether_lambda=0.0 lambda_p=0.0

# ── Phase 2: cascade — both axes controlled ────────────────────────────────
for ls in "${LS_GRID[@]}"; do
  echo "=== [grid] cascade λ_s=${ls}  λ_p=0.6 ==="
  $PY training_tether.py $CFG epochs="$EPOCHS" tether_lambda="$ls" lambda_p=0.6
done

echo "=== [grid] done ==="
