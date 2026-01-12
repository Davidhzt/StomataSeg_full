#!/bin/bash
set -e

CONFIG="configs/upernet_swin-t_8xb2-160k_stomata-341x341_patched.py"
WORK_DIR="work_dirs/upernet_swin_t_patched"

python tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR

echo "Training complete. Results: $WORK_DIR"

