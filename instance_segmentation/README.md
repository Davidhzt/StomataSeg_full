# Instance Segmentation Baselines

Six instance segmentation methods for stomatal component segmentation.

## Methods

1. Mask R-CNN + ResNet-50
2. Mask2Former + ResNet-50
3. HTC + ResNet-50
4. Mask R-CNN + Swin-T
5. Mask R-CNN + PVTv2-B2
6. Mask R-CNN + ConvNeXt-V2-Base

## Training

Train individual method:
```bash
bash scripts/train_mask_rcnn_r50.sh
bash scripts/train_mask2former.sh
bash scripts/train_htc.sh
bash scripts/train_mask_rcnn_swin.sh
bash scripts/train_mask_rcnn_pvtv2.sh
bash scripts/train_mask_rcnn_convnext.sh
```

Train all methods:
```bash
bash scripts/train_all.sh
```

## Evaluation

```bash
python tools/test.py <config> <checkpoint> --show-dir <output_dir>
```

## Requirements

```bash
pip install -r requirements.txt
```

