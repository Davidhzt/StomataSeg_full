#!/bin/bash
set -e

CONFIG="configs/mask2former_r50_patched.py"
WORK_DIR="work_dirs/mask2former_r50_patched"

python tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR

echo "Training complete. Results: $WORK_DIR"

