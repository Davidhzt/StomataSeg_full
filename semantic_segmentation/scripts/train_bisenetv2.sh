#!/bin/bash
set -e

CONFIG="configs/bisenetv2_4xb4-160k_stomata-341x341_patched.py"
WORK_DIR="work_dirs/bisenetv2_patched"

python tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR

echo "Training complete. Results: $WORK_DIR"

