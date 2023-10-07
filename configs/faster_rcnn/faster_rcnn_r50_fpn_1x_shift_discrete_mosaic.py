_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/shift_discrete_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='FasterRCNN',
    roi_head=dict(
        bbox_head=dict(
            num_classes=6)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (1333, 800)
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='RandomAffine', scaling_ratio_range=(0.1, 2), border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='SHIFTDataset',
            data_root='data/shift/discrete/images',
            ann_file='train/front/det_2d.json',
            img_prefix='train/front/images',
            backend_type='file',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_cfg=dict(
                attributes=dict(weather_coarse='clear', timeofday_coarse='daytime'),
                filter_empty_gt=True,
                min_size=32
            ),
            seq_info='train/front/seq.csv'
        ),
        pipeline=train_pipeline
    ))
