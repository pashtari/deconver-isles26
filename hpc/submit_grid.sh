#!/usr/bin/env bash
# Submit every configuration of the paper (or the ones given as arguments) for all
# five folds as PBS jobs. PBS output goes to logs/pbs/.
#   DATA_DIR=/path/to/ISLES26 VENV=/path/to/venv bash hpc/submit_grid.sh [model ...]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${DATA_DIR:?set DATA_DIR to the ISLES26 directory}"
MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=(nnunet segresnet swinunetr
            deconver_m1224_c64_k3 deconver_m1224_c32_k3 deconver_m11111_c32_k3 deconver_m11111_c32_k5
            deconver_m11111_c64_k3 deconver_m111111_c32_k3 deconver_m12244_c64_k3)
fi
mkdir -p "$REPO/logs/pbs"
for model in "${MODELS[@]}"; do
    for fold in 0 1 2 3 4; do
        qsub -N "${model}_f${fold}" \
            -v "MODEL=$model,FOLD=$fold,REPO=$REPO,DATA_DIR=$DATA_DIR,VENV=${VENV:-}" \
            -o "$REPO/logs/pbs/${model}_fold${fold}.out" -e "$REPO/logs/pbs/${model}_fold${fold}.err" \
            "$REPO/hpc/train.pbs"
    done
done
