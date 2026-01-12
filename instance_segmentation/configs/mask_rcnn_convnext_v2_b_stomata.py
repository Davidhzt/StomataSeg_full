_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# please install the mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)

# Dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/zhongtian/stomata/StomataSeg/coco_format_stomata/'

# Classes for stomata dataset
classes = ('pore area', 'guard cell area', 'complex area')

# ConvNeXt-V2 pretrained checkpoint
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth'
image_size = (1024, 1024)

# Model settings - ConvNeXt-V2 backbone with Mask R-CNN
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=0.,  # disable layer scale when using GRN
        gap_before_final_norm=False,
        use_grn=True,  # V2 uses GRN (Global Response Normalization)
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        bbox_head=dict(num_classes=3),  # Updated for stomata dataset
        mask_head=dict(num_classes=3)),  # Updated for stomata dataset
    test_cfg=dict(
        rpn=dict(nms=dict(type='nms')),
        rcnn=dict(nms=dict(type='soft_nms'), score_thr=0.3)))  # Score threshold for visualization

# Training pipeline with Large Scale Jittering (LSJ)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# Test pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

# Training settings
train_dataloader = dict(
    batch_size=4,  # Reduced for memory efficiency
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/val.json',
        data_prefix=dict(img='val/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/test.json',
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline))

# Evaluation settings
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/val.json',
    metric=['bbox', 'segm'],
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/test.json',
    metric=['bbox', 'segm'],
    format_only=False)

# Training schedule - increased epochs for better convergence
max_epochs = 500
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Learning rate schedule with layer-wise decay
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[350, 450],  # Adjusted for 500 epochs
        gamma=0.1)
]

# Optimizer with AdamW and layer-wise learning rate decay
optim_wrapper = dict(
    type='AmpOptimWrapper',  # Enable automatic mixed precision
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 12
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ))

# Hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=3,
        interval=5))

# Auto scale LR
auto_scale_lr = dict(enable=False, base_batch_size=16)
