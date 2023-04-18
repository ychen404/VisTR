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
    --intermediate \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'single' ]
then
    echo "Single-gpu training"
    python main.py --backbone resnet50 \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --intermediate \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_masks' ]
then
    echo "training small transformer"
    python main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --intermediate \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_masks_no_intermediate' ]
then
    echo "training small transformer"
    python main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_masks_no_intermediate_debug' ]
then
    echo "training small transformer"
    python -m pdb main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_dist' ]
then
    echo "training small transformer dist"
    python -m torch.distributed.launch --nproc_per_node=4 \
    main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --intermediate \
    --output_dir ckpts/small_dist \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_masks_intermediate_debug' ] # add a simple case for pdb
then
    CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --intermediate \
        --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_masks_no_intermediate_debug' ] # add a simple case for pdb
then
    CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_masks_no_intermediate' ] # add a simple case for pdb
then
    CUDA_VISIBLE_DEVICES=0 python main.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_debug' ] # add a simple case for pdb
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
    ee=0
    echo "Early-exit from $ee layer"
    python -m pdb main_ee.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --epochs 1 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_ee_"$ee"_1_batch_test \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'ffn_ee' ] # 
then
    CUDA_VISIBLE_DEVICES=0 python main_ee.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'ffn_ee_dist' ] # 
then
    ee=0
    echo "Early-exit from $ee layer"
    python main_ee.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --epochs 1 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_ee_"$ee"_1_batch_test \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth 
                        # --output_dir test_print \
elif [ $1 == 'segm_ee_debug' ] # add a simple case for pdb
then
    ee=0
    echo "Early-exit from $ee layer"
    CUDA_VISIBLE_DEVICES=0 python -m pdb main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --epochs 1 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_test_segm \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'segm_ee' ] 
then
    ee=0
    echo "Early-exit from $ee layer"
    CUDA_VISIBLE_DEVICES=0 python main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --epochs 1 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_test_segm_ee_"$ee" \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
else
    echo "Not supported argument"
fi