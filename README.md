# StomataSeg_full

Semi-supervised instance segmentation framework for sorghum stomatal components with comprehensive baselines.

## Overview

This repository provides a complete implementation of the semi-supervised patch-based segmentation framework, including:

- 6 semantic segmentation baselines using MMSegmentation
- 6 instance segmentation baselines using MMDetection  
- Semi-supervised pseudo-labeling pipeline using Mask R-CNN with ConvNeXt-V2-Base backbone

## Repository Structure

```
StomataSeg_full/
├── semantic_segmentation/    # 6 semantic segmentation methods
├── instance_segmentation/    # 6 instance segmentation methods
├── pseudo_labeling/          # Semi-supervised pipeline
└── README.md
```

## Quick Start

### Download Dataset

Our Sorghum Stomata Dataset (StomataSeg) is available for download at [Zenodo](https://zenodo.org/records/18216859).

### Semantic Segmentation

```bash
cd semantic_segmentation
bash scripts/train_all.sh
```

### Instance Segmentation

```bash
cd instance_segmentation
bash scripts/train_all.sh
```

### Pseudo Labeling Pipeline

```bash
cd pseudo_labeling
python tools/pseudo_label_with_patches.py --help
```

## Requirements

Each module has its own requirements file. Install dependencies as needed:

```bash
cd semantic_segmentation && pip install -r requirements.txt
cd instance_segmentation && pip install -r requirements.txt
cd pseudo_labeling && pip install -r requirements.txt
```
## Acknowledgments

Our code is built based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMDetection](https://github.com/open-mmlab/mmdetection) frameworks.
