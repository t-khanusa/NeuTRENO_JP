#!/usr/bin/env bash
# Grid search for NeuTRENO-JEPA: sweep the value-residual strength λ_p.
#
# λ_p = 0.0 : plain ViT baseline (expected to collapse fast).
# λ_p = 0.6 : NeuTRENO paper default.
# λ_p > 1.0 : over-controlled regime (values pinned hard to V0).
#
# Runs are SEQUENTIAL (one GPU). Each logs to wandb project "PID-NeuTRENO"
# with the name neutreno_lp<λ>. Edit the array below to change the grid.
#
# Usage:
#   bash grid_neutreno.sh                 # default grid, EPOCHS=200
#   EPOCHS=100 GPU=1 bash grid_neutreno.sh
set -euo pipefail

PY=/opt/anaconda/envs/mambavision/bin/python
CFG="--config-path configs --config-name neutreno"
EPOCHS="${EPOCHS:-200}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

LP_GRID=(1.5 1.0 0.8 0.6 0.4 0.2 0.0)
for lp in "${LP_GRID[@]}"; do
  echo "=== [grid] NeuTRENO  lambda_p=${lp} ==="
  $PY training_neutreno.py $CFG epochs="$EPOCHS" lambda_p="$lp"
done

echo "=== [grid] done ==="
