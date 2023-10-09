_base_ = [
    '../swin/faster_rcnn_swin-t-p4-w7_fpn_1x_shift_discrete.py'
]

data = dict(samples_per_gpu=2)
adapter = dict(
    type='Adapter',
    is_adapt=True,
    where='full',
    how='ema-kl',
    gamma=128,
    source_stats='./storage/stats/faster_rcnn_swin-t-p4-w7-fpn_1x_shift_epoch_12.pth'
)
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
