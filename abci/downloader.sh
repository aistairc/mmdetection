#!/bin/bash

CHECKPOINTS_DIR=data/checkpoints

# ref https://github.com/ishitsuka-hikaru/mmdetection/tree/master/configs
MODEL_URLS=(
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth
    https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth
)

mkdir -p $CHECKPOINTS_DIR
cd $CHECKPOINTS_DIR

for url in ${MODEL_URLS[@]}
do
    curl -O $url
done
