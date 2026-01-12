#!/bin/bash
set -e

echo "Training all instance segmentation methods..."

./train_mask_rcnn_r50.sh
./train_mask2former.sh
./train_htc.sh
./train_mask_rcnn_swin.sh
./train_mask_rcnn_pvtv2.sh
./train_mask_rcnn_convnext.sh

echo "All training complete."

