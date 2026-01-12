#!/bin/bash
set -e

CONFIG="configs/htc_r50_patched.py"
WORK_DIR="work_dirs/htc_r50_patched"

python tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR

echo "Training complete. Results: $WORK_DIR"

