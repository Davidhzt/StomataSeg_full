# Mask2Former with ResNet-50 Backbone on Patched Stomata Dataset
#
# 🎯 PATCH-BASED INSTANCE SEGMENTATION with Mask2Former
#
# Strategy:
#   • Train on 341×341 patches (pores appear relatively larger!)
#   • Use Mask2Former (query-based segmentation)
#   • ResNet-50 backbone with transformer decoder
#
# Benefits:
#   • Mask2Former: State-of-the-art instance segmentation
#   • Query-based: Better handling of overlapping instances
#   • Patch-based: Pores 3x larger relatively (8.3% vs 2.8%)
#   • Unified architecture: Single model for all segmentation tasks

_base_ = [
    '../mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'
]

# Dataset settings - use COMPLETE PATCHED dataset!
dataset_type = 'CocoDataset'
data_root = '/data/zhongtian/stomata/DatasetPaper_PatchBased/data/coco_patched_labeled/'

# Classes for stomata dataset
classes = ('pore area', 'guard cell area', 'complex area')

# =============================================================================
# Model Settings for Instance Segmentation
# =============================================================================
num_things_classes = 3  # pore, guard cell, complex
num_stuff_classes = 0   # no stuff classes for instance segmentation
num_classes = num_things_classes + num_stuff_classes

# Update model config for instance segmentation only
model = dict(
    panoptic_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_cls=dict(class_weight=[1.0] * num_classes + [0.1])),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes),
    test_cfg=dict(
        panoptic_on=False,    # Disable panoptic segmentation
        semantic_on=False,    # Disable semantic segmentation
        instance_on=True))    # Enable instance segmentation only

# =============================================================================
# Data Pipeline for Patches (341×341)
# =============================================================================

# Training pipeline for patches
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # Multi-scale training with patch sizes
    dict(
        type='RandomResize',
        scale=(341, 341),
        ratio_range=(0.8, 1.2),  # Reasonable range for patches
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(341, 341),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='PackDetInputs')
]

# Test pipeline for patches
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(341, 341), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# Data loaders
train_dataloader = dict(
    batch_size=4,  # Mask2Former is memory-intensive
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file=data_root + 'train/train.json',
        data_prefix=dict(img=data_root + 'train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file=data_root + 'val/val.json',
        data_prefix=dict(img=data_root + 'val/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file=data_root + 'test/test.json',
        data_prefix=dict(img=data_root + 'test/'),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True,
    backend_args={{_base_.backend_args}})

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/test.json',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True,
    backend_args={{_base_.backend_args}})

# =============================================================================
# Training Configuration
# =============================================================================

# Training schedule - iteration-based (Mask2Former default)
# For patches with ~4000 training images, batch_size=4:
# 50000 iters ≈ 50 epochs (50000 * 4 / 4000 = 50 epochs)
max_iters = 50000

param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_iters,
    by_epoch=False,
    milestones=[40000, 47000],
    gamma=0.1)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=2000)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer - AdamW (standard for Mask2Former)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# Checkpoint settings
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        save_last=True,
        max_keep_ckpts=3,
        interval=2000))

# Log settings
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=False)

# Runtime settings
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_level = 'INFO'
load_from = None
resume = False

# =============================================================================
# SUMMARY - Patch-Based Instance Segmentation with Mask2Former
# =============================================================================
# Key Features:
#   • Mask2Former: Query-based unified segmentation architecture
#   • ResNet-50 backbone with transformer decoder
#   • Patch-based training (341×341) → pores appear 3x larger relatively
#   • Batch size 4 (Mask2Former is memory-intensive)
#   • Instance segmentation only (panoptic_on=False, semantic_on=False)
#
# Expected Performance:
#   • Pore AP: 30-45% (query-based better for small objects!)
#   • Guard Cell AP: 55-65%
#   • Complex AP: 60-70%
#   • Overall mAP: 48-60%
#
# Training: ~24-30 hours (50000 iterations ≈ 50 epochs)
#
# Advantages over Mask R-CNN:
#   • Query-based: Better handling of overlapping instances
#   • Unified architecture: Single model for all tasks
#   • No anchor boxes: More flexible object detection
#   • Transformer decoder: Better feature aggregation
# =============================================================================

