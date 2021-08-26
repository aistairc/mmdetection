#!/bin/bash

#$ -l rt_F=1
#$ -j y
#$ -cwd

DATA_DIR=`pwd`/data
SIMG=`pwd`/abci/simg/mmd
SRC_DIR=$DATA_DIR/images
CONFIG=configs/yolo/yolov3_d53_320_273e_coco.py
CHECKPOINT=$DATA_DIR/checkpoints/yolov3_d53_320_273e_coco-421362b6.pth
DST_DIR=$DATA_DIR/outputs/demo1

source /etc/profile.d/modules.sh
module load singularitypro/3.7

mkdir -p $DST_DIR

for src in $SRC_DIR/*.jpg
do
    singularity exec --nv --pwd /mmdetection --bind $DATA_DIR:/mmdetection/data $SIMG \
	python3 demo/image_demo.py $src $CONFIG $CHECKPOINT --out_file $DST_DIR/`basename $src`
done
