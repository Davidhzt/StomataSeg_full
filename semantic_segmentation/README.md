# Semantic Segmentation Baselines

Six semantic segmentation methods for stomatal component segmentation.

## Methods

1. FCN + ResNet-50
2. OCRNet + ResNet-50
3. BiSeNetV2
4. UPerNet + Swin-T
5. SegNeXt + MSCAN-S
6. SegFormer + MIT-B1

## Training

Train individual method:
```bash
bash scripts/train_fcn.sh
bash scripts/train_ocrnet.sh
bash scripts/train_bisenetv2.sh
bash scripts/train_upernet.sh
bash scripts/train_segnext.sh
bash scripts/train_segformer.sh
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

