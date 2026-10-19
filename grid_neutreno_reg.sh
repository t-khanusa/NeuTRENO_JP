#!/usr/bin/env bash
# NeuTRENO × anti-collapse regularizer (training_neutreno_reg.py), one sequential
# queue on one GPU. Data pipeline = LeJEPA (2 global + 6 local views @128, bs=128);
# runs log to the wandb project "LeJEPA" next to neutreno0.6_lejepa / lejepa-sigreg.
#
# Priority: VISReg and λ-JEPA, each with NeuTRENO (lambda_p=0.6) and without
# (timm ViT, as lejepa-sigreg), then the same-network controls (lambda_p=0.0).
#
# Usage:
#   bash grid_neutreno_reg.sh                       # GPU2, 100 epochs
#   GPU=2 EPOCHS=800 bash grid_neutreno_reg.sh
#   RUNS=extra bash grid_neutreno_reg.sh            # the other regularizers
set -euo pipefail

PY=/opt/anaconda/envs/mambavision/bin/python
GPU="${GPU:-2}"
EPOCHS="${EPOCHS:-100}"
LOG_DIR=logs/neutreno_reg
mkdir -p "$LOG_DIR"

# The imagenette loader asks to trust remote code when the Hub is reachable and
# arms a SIGALRM that can kill the run later — read the local cache instead.
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

PRIORITY=(
  "reg=visreg lambda_p=0.6"
  "reg=visreg backbone=timm"
  "reg=lambdajepa lambda_p=0.6"
  "reg=lambdajepa backbone=timm"
  "reg=visreg lambda_p=0.0"
  "reg=lambdajepa lambda_p=0.0"
)
EXTRA=(
  "reg=sigreg lambda_p=0.0"
  "reg=vicreg lambda_p=0.6"
  "reg=vicreg backbone=timm"
  "reg=barlow lambda_p=0.6"
  "reg=barlow backbone=timm"
  "reg=uniformity lambda_p=0.6"
  "reg=uniformity backbone=timm"
)

if [[ "${RUNS:-priority}" == "extra" ]]; then QUEUE=("${EXTRA[@]}"); else QUEUE=("${PRIORITY[@]}"); fi

for args in "${QUEUE[@]}"; do
  name=$(echo "$args" | tr ' =' '_-')
  echo "=== [gpu$GPU] $args (epochs=$EPOCHS) $(date '+%F %T') ==="
  CUDA_VISIBLE_DEVICES=$GPU $PY training_neutreno_reg.py $args epochs="$EPOCHS" \
    > "$LOG_DIR/$name.log" 2>&1 || echo "!!! [gpu$GPU] $args failed (see $LOG_DIR/$name.log)"
done
echo "=== [gpu$GPU] queue done $(date '+%F %T') ==="
