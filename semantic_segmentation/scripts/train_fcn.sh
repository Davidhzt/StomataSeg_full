#!/bin/bash
set -e

CONFIG="configs/fcn_r50-d8_160k_stomata-341x341_patched.py"
WORK_DIR="work_dirs/fcn_r50_patched"

python tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR

echo "Training complete. Results: $WORK_DIR"

