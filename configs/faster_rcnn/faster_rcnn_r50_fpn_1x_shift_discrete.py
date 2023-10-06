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
