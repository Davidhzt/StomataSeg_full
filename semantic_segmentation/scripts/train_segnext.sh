#!/bin/bash
set -e

CONFIG="configs/segnext_mscan-s_1xb16-adamw-160k_stomata-341x341_patched.py"
WORK_DIR="work_dirs/segnext_mscan_s_patched"

python tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR

echo "Training complete. Results: $WORK_DIR"

