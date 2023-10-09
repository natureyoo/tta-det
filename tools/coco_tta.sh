
# direct test swin
#python tools/coco_corrupt_test.py \
#  configs/swin/faster_rcnn_swin-t-p4-w7_fpn_1x_coco.py \
#  storage/ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_coco_epoch_12.pth \
#  --work-dir "work_dirs/faster_rcnn_swin-t-p4-w7_fpn_1x_coco_direct_test" \
#  --eval "bbox"

# tta swin full
python tools/coco_adapt.py \
    configs/adapter/faster_rcnn_swin-t_fpn_coco.py \
    storage/ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_coco_epoch_12.pth \
    --work-dir "work_dirs/COCO/faster_rcnn_swin-t-p4-w7_fpn_1x_tta_ema-kl_full_lr-0.0001" \
    --eval "bbox" --wandb

# tta swin adapter
python tools/coco_adapt.py \
    configs/adapter/faster_rcnn_swin-t_fpn_coco_adapter.py \
    storage/ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_coco_epoch_12.pth \
    --work-dir "work_dirs/COCO/faster_rcnn_swin-t-p4-w7_fpn_1x_tta_ema-kl_adapter_lr-0.0001" \
    --eval "bbox" --wandb

# direct test res50
#python tools/coco_corrupt_test.py \
#  configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
#  storage/ckpt/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#  --work-dir "work_dirs/faster_rcnn_r50_fpn_1x_coco_direct_test" \
#  --eval "bbox"

# tta r50 full
#python tools/coco_adapt.py \
#    configs/adapter/faster_rcnn_r50_fpn_coco_full.py \
#    storage/ckpt/stfar_coco_r50_fpn_teacher.pth \
#    --work-dir "work_dirs/COCO/faster_rcnn_r50_fpn_1x_tta_ema-kl_full_lr-0.0001" \
#    --eval "bbox"