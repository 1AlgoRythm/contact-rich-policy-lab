#!/bin/sh
# Train all 18 models: one process per seed, run in parallel (each uses 2 torch threads).
mkdir -p logs
for s in 0 1 2; do .venv/bin/python scripts/train.py --seeds $s > logs/train_s$s.log 2>&1 & done
wait
