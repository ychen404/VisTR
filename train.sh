#!/bin/bash

if [ $# == 0 ]
then
    echo "missing input argument"
elif [ $1 == 'dist' ]
then
    echo "Distributed-gpu training"
    python -m torch.distributed.launch --nproc_per_node=4 \
    main.py --backbone resnet50 \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'single' ]
then
    echo "Single-gpu training"
    python main.py --backbone resnet50 \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small' ]
then
    echo "training small transformer"
    python main.py --backbone resnet50 \
    --enc_layers 2 \
    --dec_layers 2 \
    --nheads 2 \
    --num_frames 3 \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
else
    echo "Not supported argument"
fi