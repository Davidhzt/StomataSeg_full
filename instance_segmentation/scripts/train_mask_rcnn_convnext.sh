#!/bin/bash
set -e

CONFIG="configs/mask_rcnn_patched_dataset.py"
WORK_DIR="work_dirs/mask_rcnn_convnext_v2_patched"

python tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR

echo "Training complete. Results: $WORK_DIR"

