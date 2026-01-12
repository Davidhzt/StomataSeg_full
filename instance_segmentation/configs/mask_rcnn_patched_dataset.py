# Mask R-CNN on Patched Stomata Dataset
#
# 🎯 INTUITIVE SOLUTION: Patch-Based Detection
#
# Inspired by: "Comparative analysis of stomatal pore instance segmentation:
#               Mask R-CNN vs. YOLOv8 on Phenomics Stomatal dataset"
#


_base_ = './mask_rcnn_convnext_v2_b_stomata.py'

# Data settings - use PATCHED dataset!
data_root = '/data/zhongtian/stomata/DatasetPaper_PatchBased/data/coco_patched_labeled/'

# =============================================================================
# Key Difference: Training on 341×341 patches instead of 1024×1024 images!
# =============================================================================

# Training pipeline for patches
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    
    # Keep patch size (341×341) or slightly larger
    dict(
        type='RandomChoiceResize',
        scales=[
            (341, 341),  # Original patch size
            (384, 384),  # Slightly larger
            (416, 416),  # Medium scale
        ],
        keep_ratio=True),
    
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# Test pipeline for patches
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(341, 341), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Data loaders
train_dataloader = dict(
    batch_size=16,  # Can use larger batch size with smaller patches!
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='train/train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,  # Larger batch size for faster validation
    num_workers=8,  # More workers for parallel data loading
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='val/val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=8,  # Larger batch size for faster inference
    num_workers=8,  # More workers for parallel data loading
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='test/test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/test.json',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True)

# Training schedule
max_epochs = 50

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[400, 470],
        gamma=0.1)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=2)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

auto_scale_lr = dict(base_batch_size=16, enable=True)

# Save best checkpoint
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=2,
        by_epoch=True,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater',
        save_last=True))



