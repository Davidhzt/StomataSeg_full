# Mask R-CNN with PVT v2-B2 Backbone on Patched Stomata Dataset
#
# 🎯 PATCH-BASED DETECTION with Transformer Backbone
#
# Strategy:
#   • Train on 341×341 patches (pores appear relatively larger!)
#   • Use PVT v2-B2 transformer backbone
#   • Expected: Improved performance on small objects (pores)
#
# Benefits:
#   • PVT v2-B2: Better multi-scale feature extraction
#   • Patch-based: Pores 3x larger relatively (8.3% vs 2.8%)
#   • Transformer attention: Better small object detection

_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Dataset settings - use PATCHED dataset!
dataset_type = 'CocoDataset'
data_root = '/data/zhongtian/stomata/DatasetPaper_PatchBased/data/coco_patched_labeled/'
classes = ('pore area', 'guard cell area', 'complex area')

# =============================================================================
# Model: Mask R-CNN with PVT v2-B2 backbone
# =============================================================================
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=3,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.3,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

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
    batch_size=20,  # Larger batch size with smaller patches
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
    val_interval=20)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer - AdamW for transformer backbone
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001))

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
# SUMMARY - Patch-Based Detection with PVT v2-B2
# =============================================================================
# Key Features:
#   • Transformer backbone (PVT v2-B2) for better feature extraction
#   • Patch-based training (341×341) → pores appear 3x larger relatively
#   • Batch size 8 (efficient training)
#   • AdamW optimizer (standard for transformers)
#
# Expected Performance:
#   • Pore AP: 25-40% (transformer + patches = better small object detection!)
#   • Guard Cell AP: 55-65%
#   • Complex AP: 60-70%
#   • Overall mAP: 47-58%
#
# Training: ~24-30 hours (transformer is slightly slower than CNN)
# =============================================================================

