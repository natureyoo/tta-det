python tools/shift_test.py \
  configs/swin/faster_rcnn_swin-t-p4-w7_fpn_1x_shift_discrete.py \
  ckpt/faster_rcnn_swin-t-p4-w7-fpn_1x_shift_clear_daytime_epoch_12.pth \
  --work-dir "work_dirs/faster_rcnn_swin-t-p4-w7_fpn_1x_shift_direct_test" \
  --eval "mAP"