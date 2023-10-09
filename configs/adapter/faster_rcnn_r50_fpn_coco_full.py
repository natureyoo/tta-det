_base_ = [
    '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
]

data = dict(samples_per_gpu=2)
adapter = dict(
    type='Adapter',
    is_adapt=True,
    where='full',
    how='ema-kl',
    gamma=128,
    source_stats='./storage/stats/stfar_coco_r50_fpn_teacher.pth'
)
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
