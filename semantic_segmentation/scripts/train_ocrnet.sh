#!/bin/bash
set -e

CONFIG="configs/ocrnet_r50-d8_4xb4-160k_stomata-341x341_patched.py"
WORK_DIR="work_dirs/ocrnet_r50_patched"

python tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR

echo "Training complete. Results: $WORK_DIR"

