#!/bin/bash
set -e

echo "Training all semantic segmentation methods..."

./train_fcn.sh
./train_ocrnet.sh
./train_bisenetv2.sh
./train_upernet.sh
./train_segnext.sh
./train_segformer.sh

echo "All training complete."

