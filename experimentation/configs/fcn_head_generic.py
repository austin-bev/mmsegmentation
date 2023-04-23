
num_classes = 0
dataset_type = ''
data_root = ''
norm_cfg = dict(type='BN', requires_grad=True, momentum=0.01)

crop_size = (0, 0)

batch_size = 0
workers_per_gpu = 0

train_pipeline = []
test_pipeline = []

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='FastSCNN',
            
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        channels=64,
        num_classes=12,
        in_index=-1,
        norm_cfg=norm_cfg,
        concat_input=False,
        align_corners=False,
        loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='',
        ann_dir='',
        pipeline=test_pipeline)
    )

log_config = dict(
    interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=True)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.045, momentum=0.9, weight_decay=4e-05)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=1000)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=16000, metric='mIoU', pre_eval=True, save_best='auto')
