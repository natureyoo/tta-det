
# direct test
#python tools/coco_corrupt_test.py \
#  configs/swin/faster_rcnn_swin-t-p4-w7_fpn_1x_coco.py \
#  storage/ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_coco_epoch_12.pth \
#  --work-dir "work_dirs/faster_rcnn_swin-t-p4-w7_fpn_1x_coco_direct_test" \
#  --eval "bbox"

# tta
python tools/coco_adapt.py \
    configs/adapter/faster_rcnn_swin-t_fpn_coco.py \
    storage/ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_coco_epoch_12.pth \
    --work-dir "work_dirs/faster_rcnn_swin-t-p4-w7_fpn_1x_coco_tta_ema-kl_adapter_lr-0.01" \
    --eval "bbox"