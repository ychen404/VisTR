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
    --num_frames 3 \
    --num_queries 6 \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_dist' ]
then
    echo "training small transformer dist"
    python -m torch.distributed.launch --nproc_per_node=4 \
    main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 6 \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'debug' ] # add a simple case for pdb
then
    CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'ffn' ] # add a simple case for pdb
then
    CUDA_VISIBLE_DEVICES=0 python main.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'ffn_debug' ] # add a simple case for pdb
then
    CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'ffn_ee_debug' ] # add a simple case for pdb
then
    CUDA_VISIBLE_DEVICES=0 python -m pdb main_ee.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'ffn_ee' ] # add a simple case for pdb
then
    CUDA_VISIBLE_DEVICES=0 python main_ee.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'ffn_ee_dist' ] # add a simple case for pdb
then
    python main_ee.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --epochs 1 \
        --output_dir res50_ee_5_1_batch \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth 
else
    echo "Not supported argument"
fi

    # --enc_layers 2 \
    # --dec_layers 2 \
    # --nheads 2 \
