# Semi-Supervised Pseudo-Labeling Pipeline

Patch-based pseudo-labeling pipeline using Mask R-CNN with ConvNeXt-V2-Base backbone.

## Overview

The pipeline includes three main steps:
1. Prepare patches from original images
2. Generate pseudo-labels on unlabeled images
3. Merge labeled and pseudo-labeled datasets
4. Used the final merged Dataset to retain the same model

## Usage

### Step 1: Prepare Patches

```bash
python tools/prepare_initial_patches.py \
    --data-root <labeled_data_dir> \
    --output-root <output_dir> \
    --patch-size 341 \
    --overlap 10
```

### Step 2: Generate Pseudo-Labels

```bash
python tools/pseudo_label_with_patches.py \
    --unlabeled-dir <unlabeled_images_dir> \
    --model-config <config_file> \
    --checkpoint <model_checkpoint> \
    --output-dir <output_dir> \
    --confidence-threshold 0.7 \
    --patch-size 341 \
    --overlap 10
```

### Step 3: Merge Datasets

```bash
python tools/merge_datasets.py \
    --labeled-root <labeled_patches_dir> \
    --pseudo-root <pseudo_labels_dir> \
    --output-root <merged_output_dir>
```
### Step 4: Retain the Model

## Configuration

Key parameters:
- `patch_size`: Size of patches (default: 341)
- `overlap`: Overlap between patches (default: 10)
- `confidence_threshold`: Minimum confidence for pseudo-labels (default: 0.7)

Class-specific thresholds can be set:
- Pore area: 0.5
- Guard cell area: 0.7
- Complex area: 0.7

## Requirements

```bash
pip install -r requirements.txt
```

