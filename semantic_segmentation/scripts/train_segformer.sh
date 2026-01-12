#!/bin/bash
set -e

CONFIG="configs/segformer_mit-b1_8xb2-160k_stomata-512x512.py"
WORK_DIR="work_dirs/segformer_mit_b1"

python tools/train.py \
    $CONFIG \
    --work-dir $WORK_DIR

echo "Training complete. Results: $WORK_DIR"

