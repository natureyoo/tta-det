# tta swin full
for lr in 0.0001
do python tools/shift_adapt.py \
    configs/adapter/faster_rcnn_swin-t_fpn_shift_discrete.py \
    storage/ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_shift_epoch_12.pth \
    --work-dir "work_dirs/SHIFT/faster_rcnn_swin-t-p4-w7_fpn_1x_tta_ema-kl_full_lr-${lr}_clip5.0" \
    --eval "mAP" --wandb --lr ${lr}
done

for lr in 0.0001
do python tools/shift_adapt.py \
    configs/adapter/faster_rcnn_swin-t_fpn_shift_discrete.py \
    storage/ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_shift_epoch_12.pth \
    --work-dir "work_dirs/SHIFT_CTA/faster_rcnn_swin-t-p4-w7_fpn_1x_ema-kl_full_lr-${lr}_clip5.0" \
    --eval "mAP" --wandb --lr ${lr} --continual
done

## tta swin adapter
#python tools/shift_adapt.py \
#    configs/adapter/faster_rcnn_swin-t_fpn_shift_discrete_adapter.py \
#    storage/ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_shift_epoch_12.pth \
#    --work-dir "work_dirs/SHIFT/faster_rcnn_swin-t-p4-w7_fpn_1x_tta_ema-kl_adapter_lr-0.0001" \
#    --eval "mAP" --wandb

# direct test res50
#python tools/coco_corrupt_test.py \
#  configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
#  storage/ckpt/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#  --work-dir "work_dirs/faster_rcnn_r50_fpn_1x_coco_direct_test" \
#  --eval "mAP"

# tta r50 full
#python tools/shift_adapt.py \
#    configs/adapter/faster_rcnn_r50_fpn_shift_discrete_full.py \
#    storage/ckpt/faster_rcnn_r50-fpn_1x_shift_epoch_12.pth \
#    --work-dir "work_dirs/SHIFT/faster_rcnn_r50_fpn_1x_tta_ema-kl_full_lr-0.0001" \
#    --eval "mAP"
