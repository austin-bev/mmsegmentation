num_classes = 12
dataset_type = 'Camvid11Dataset'
data_root = 'data/CamVid11/'
norm_cfg = dict(type='BN', requires_grad=True, momentum=0.01)
crop_size = (480, 640)
batch_size = 4
workers_per_gpu = 1
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(720, 960), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(480, 640), cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(480, 640), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(720, 960),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ResNet',
        out_indices=(0, 1),
        norm_cfg=dict(type='BN', requires_grad=True, momentum=0.01),
        depth=34,
        pretrained='open-mmlab://resnet34',
        num_stages=2,
        strides=(1, 2),
        dilations=(1, 1),
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableFCNHead',
        in_channels=128,
        channels=64,
        concat_input=False,
        num_classes=12,
        in_index=-1,
        norm_cfg=dict(type='BN', requires_grad=True, momentum=0.01),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1)),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type='Camvid11Dataset',
        data_root='data/CamVid11/',
        img_dir='train',
        ann_dir='train_labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(720, 960), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(480, 640), cat_max_ratio=0.75),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(480, 640), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='Camvid11Dataset',
        data_root='data/CamVid11/',
        img_dir='val',
        ann_dir='val_labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(720, 960),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='Camvid11Dataset',
        data_root='data/CamVid11/',
        img_dir='test',
        ann_dir='test_labels',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(720, 960),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(
            type='MMSegWandbHook',
            by_epoch=True,
            init_kwargs=dict(
                entity='austin-bevac',
                project='2023-honours-austin',
                config=dict(
                    learning_rate=0.045,
                    momentum=0.9,
                    batch_size=4,
                    backbone='ResNet',
                    decodehead='DepthwiseSeparableFCNHead',
                    dataset='Camvid11Dataset',
                    epochs=1000,
                    is_pretrained=True,
                    comment='ResNet34, 2 stages'),
                name=
                '2023-04-04-23-37-11_Camvid11Dataset_DepthwiseSeparableFCNHead_ResNet'
            ))
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=1000, max_iters=None)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(
    interval=1, metric='mIoU', pre_eval=True, save_best='mIoU', by_epoch=True)
work_dir = './work_dirs/experimentation/2023-04-04-23-37-11_Camvid11Dataset_DepthwiseSeparableFCNHead_ResNet'
seed = 0
gpu_ids = range(0, 1)
device = 'cuda'
