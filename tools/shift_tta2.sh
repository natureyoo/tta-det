#!/bin/bash

# tta swin adapter
for lr in 0.00001
do python tools/shift_adapt.py \
    configs/adapter/faster_rcnn_swin-t_fpn_shift_discrete_adapter.py \
    storage/ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_shift_epoch_12.pth \
    --work-dir "work_dirs/SHIFT/faster_rcnn_swin-t-p4-w7_fpn_1x_tta_ema-kl_adapter_lr-${lr}" \
    --eval "mAP" --wandb --lr ${lr}
done

for lr in 0.00001
do python tools/shift_adapt.py \
    configs/adapter/faster_rcnn_swin-t_fpn_shift_discrete_adapter.py \
    storage/ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_shift_epoch_12.pth \
    --work-dir "work_dirs/SHIFT_CTA/faster_rcnn_swin-t-p4-w7_fpn_1x_cta_ema-kl_adapter_lr-${lr}" \
    --eval "mAP" --wandb --lr ${lr} --continual
done
