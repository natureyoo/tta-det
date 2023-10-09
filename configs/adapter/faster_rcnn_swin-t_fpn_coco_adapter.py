_base_ = [
    './faster_rcnn_swin-t_fpn_coco.py'
]

data = dict(samples_per_gpu=2)
adapter = dict(
    type='Adapter',
    is_adapt=True,
    where='adapter'
)
model = dict(
    type='FasterRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        adapter_cfg=dict(
            hidden_ratio=32,
            layernorm_option=None,
            scalar=0.1,
            dropout=0.0,
            init_option='lora'
        )),
    neck=dict(in_channels=[96, 192, 384, 768]))
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
