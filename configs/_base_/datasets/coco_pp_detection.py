# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco_pp/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),   # default
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),  # out of CUDA memory
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train_5v5.json',
        # ann_file=data_root + 'annotations/instances_train_7v3.json',
        ann_file=data_root + 'annotations/instances_train_9v1.json',
        # img_prefix=data_root + 'train_5v5/',
        img_prefix=data_root + 'train_9v1/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val.json',
        # img_prefix=data_root + 'val/',
        ann_file='data/coco/annotations/instances_val2017_500.json',
        img_prefix='data/coco/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val.json',
        # img_prefix=data_root + 'val2017/',
        ann_file='data/coco/annotations/instances_val2017_500.json',
        img_prefix='data/coco/val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
