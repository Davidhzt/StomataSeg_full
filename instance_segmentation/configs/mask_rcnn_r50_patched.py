# Mask R-CNN with ResNet-50 on Patched Stomata Dataset
#
# 🎯 PATCH-BASED DETECTION with Classic ResNet-50
#
# Strategy:
#   • Train on 341×341 patches (pores appear relatively larger!)
#   • Use ResNet-50: The original Mask R-CNN backbone
#   • Classic baseline for fair comparison
#
# Benefits:
#   • ResNet-50: Most widely used backbone (standard baseline)
#   • Patch-based: Pores 3x larger relatively (8.3% vs 2.8%)
#   • Fair comparison: Same architecture as original paper

_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

# Dataset settings - use PATCHED dataset!
dataset_type = 'CocoDataset'
data_root = '/data/zhongtian/stomata/DatasetPaper_PatchBased/data/coco_patched_labeled/'
classes = ('pore area', 'guard cell area', 'complex area')

# =============================================================================
# Model Settings - Standard Mask R-CNN with ResNet-50
# =============================================================================
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3),
        mask_head=dict(num_classes=3)))

# =============================================================================
# Data Pipeline for Patches (341×341)
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
    batch_size=32,  # Standard batch size for ResNet-50
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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
    classwise=True)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/test.json',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True)

# =============================================================================
# Training Configuration
# =============================================================================

# Training schedule - 500 epochs
max_epochs = 500

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
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
    val_interval=20)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer - SGD (standard for ResNet-50)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(base_batch_size=8, enable=True)

# Save best checkpoint
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=20,
        by_epoch=True,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater',
        save_last=True))

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

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

load_from = None
resume = False

# =============================================================================
# SUMMARY - Patch-Based Detection with Classic ResNet-50
# =============================================================================
# Key Features:
#   • ResNet-50: Original Mask R-CNN backbone (standard baseline)
#   • Patch-based training (341×341) → pores appear 3x larger relatively
#   • Batch size 8 (standard for ResNet-50)
#   • SGD optimizer (lr=0.02, momentum=0.9)
#
# Expected Performance:
#   • Pore AP: 30-42% (classic architecture on patches!)
#   • Guard Cell AP: 50-60%
#   • Complex AP: 55-65%
#   • Overall mAP: 45-56%
#
# Training: ~20-24 hours (efficient backbone)
#
# Why This Baseline Matters:
#   • Most widely used: Standard comparison point
#   • Original paper: He et al. 2017
#   • Fair comparison: Shows impact of modern backbones
#   • Efficiency: Faster than transformer backbones
# =============================================================================

