#!/bin/bash

CMD=tools/dist_train.sh
CONFIGS=(
    data/configs/ssd/ssd300_coco_pp_10v1.py
    data/configs/ssd/ssd300_coco_pp_10v2.py
    data/configs/ssd/ssd300_coco_pp_10v3.py
    data/configs/ssd/ssd300_coco_pp_10v4.py
    data/configs/ssd/ssd300_coco_pp_10v5.py
)
GPUS=2
WORK_DIRS=(
    data/coco_pp/checkpoints/10v1
    data/coco_pp/checkpoints/10v2
    data/coco_pp/checkpoints/10v3
    data/coco_pp/checkpoints/10v4
    data/coco_pp/checkpoints/10v5
)

for i in "${!CONFIGS[@]}";
do
    $CMD ${CONFIGS[i]} $GPUS --work-dir ${WORK_DIRS[i]}
    # echo "${CONFIGS[i]} ${WORK_DIRS[i]}"
done
