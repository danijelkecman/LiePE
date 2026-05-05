#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
python experiments/train.py --task copy_delay --encoding rope --steps 20 --seq-len 32 --batch-size 8 --d-model 64 --n-layers 1 --n-heads 4 --output outputs/smoke_rope.pt
python liepe/viz/attention_maps.py --encoding rope --seq-len 32 --d-model 64 --n-layers 1 --n-heads 4 --output outputs/smoke_attention.png
python liepe/viz/eigenvalues.py --encoding jordan --dim 16 --output outputs/smoke_jordan_eigs.png
