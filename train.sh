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
    CUDA_VISIBLE_DEVICES=0 python main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth \
    --intermediate 
elif [ $1 == 'small_masks' ]
then
    echo "training small transformer"
    CUDA_VISIBLE_DEVICES=0 python main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --epochs 1 \
    --output_dir ckpts/small_masks \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
    --intermediate \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_masks_debug' ]
then
    echo "training small transformer"
    CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --epochs 1 \
    --output_dir ckpts/small_masks_debug \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_small \
    --intermediate \
    --debug \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_masks_tiny' ]
then
    echo "training small transformer"
    CUDA_VISIBLE_DEVICES=0 python main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --output_dir ckpts/tinyds_baseline_from_model_pretrained_on_smallds \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_tiny \
    --intermediate \
    --workspace tinyds_baseline_from_model_pretrained_on_smallds \
    --log logs/tinyds_baseline_from_model_pretrained_on_smallds \
    --masks \
    --pretrained_weights ckpts/small_model_checkpoint_fullytrained/checkpoint.pth
    # --pretrained_weights pretrained_weights/384_coco_r50.pth
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
    --masks \
    --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_dataset_dist' ]
then
    echo "training small transformer dist"
    python -m torch.distributed.launch --nproc_per_node=4 \
    main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --intermediate \
    --output_dir ckpts/small_dist_smallds \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_small \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
        # --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
elif [ $1 == 'small_dataset_debug' ]
then
    echo "training small transformer"
    CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --intermediate \
    --early_break \
    --output_dir ckpts/small_dist_smallds \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_small \
    --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'small_dataset' ]
then
    echo "training small transformer"
    CUDA_VISIBLE_DEVICES=0 python main.py --backbone resnet50 \
    --num_frames 3 \
    --num_queries 15 \
    --early_break \
    --output_dir ckpts/small_dist_smallds \
    --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_small \
    --masks \
    --pretrained_weights pretrained_weights/384_coco_r50.pth
    # --intermediate \
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
        --debug \
        --masks --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'ffn' ] # add a simple case for pdb
then
    CUDA_VISIBLE_DEVICES=0 python main.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --debug \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'ffn_debug' ] # add a simple case for pdb
then
    CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --backbone resnet50 \
        --num_frames 3 \
        --num_queries 15 \
        --debug \
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
        --debug \
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
    python -m torch.distributed.launch --nproc_per_node=4 \
    main_ee.py --backbone resnet50 \
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
        --debug \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_test_segm \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'segm_ee_smallds_debug' ] # small dataset
then
    ee=3
    echo "Early-exit from $ee layer"
    CUDA_VISIBLE_DEVICES=0 python -m pdb main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --epochs 1 \
        --debug \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_test_segm \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_small \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'segm_ee_smallds' ] # small dataset
then
    ee=5
    echo "Early-exit from $ee layer"
    CUDA_VISIBLE_DEVICES=0 python main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_test_segm \
        --log logs/segm_ee_smallds.json \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_small \
        --workspace segm_ee_smallds \
        --pretrained_weights ckpts/small_model_checkpoint_fullytrained/checkpoint.pth
        # --pretrained_weights pretrained_weights/384_coco_r50.pth
        # --early_break \
elif [ $1 == 'segm_ee_smallds_dist' ] # small dataset
then
    ee=5
    echo "Early-exit from $ee layer"
    python -m torch.distributed.launch --nproc_per_node=4 \
        main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_test_segm \
        --log logs/segm_ee_smallds_dist.json \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_small \
        --workspace segm_ee_smallds_dist \
        --pretrained_weights ckpts/small_model_checkpoint_fullytrained/checkpoint.pth
elif [ $1 == 'segm_ee_tinyds' ] # small dataset
then
    ee=5
    echo "Early-exit from $ee layer"
    CUDA_VISIBLE_DEVICES=0 python main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_segm_ee_"$ee"_tinyds_from_scratch \
        --log logs/segm_ee_"$ee"_tinyds_all_segms_from_scratch.json \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_tiny \
        --workspace _segm_ee_"$ee"_tinyds_all_segms_from_scratch \
        --pretrained_weights pretrained_weights/384_coco_r50.pth 
        # --pretrained_weights ckpts/small_model_checkpoint_fullytrained/checkpoint.pth
        # --pretrained_weights pretrained_weights/384_coco_r50.pth
        # --early_break \
elif [ $1 == 'segm_ee_tinyds_debug' ] # small dataset
then
    ee=5
    echo "Early-exit from $ee layer"
    CUDA_VISIBLE_DEVICES=0 python -m pdb main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --debug \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_test_segm \
        --log logs/segm_ee_5_tinyds_all_segms.json \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_tiny \
        --workspace segm_ee_5_tinyds_all_segms \
        --pretrained_weights ckpts/small_model_checkpoint_fullytrained/checkpoint.pth
        # --pretrained_weights pretrained_weights/384_coco_r50.pth
        # --early_break \
elif [ $1 == 'segm_ee' ] 
then
    ee=2
    echo "Early-exit from $ee layer"
    CUDA_VISIBLE_DEVICES=0 python main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_test_segm_ee_"$ee"_fullytrained \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights pretrained_weights/384_coco_r50.pth
        # --epochs 1 \
        # --early_break \
elif [ $1 == 'segm_ee_dist' ]
then
    ee=3
    echo "Early-exit from $ee layer"
    python -m torch.distributed.launch --nproc_per_node=4 \
    main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_test_segm_ee_"$ee"_fullytrained_dist_from_pretrained_smallmodel \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019 \
        --pretrained_weights ckpts/small_model_checkpoint_fullytrained/checkpoint.pth \
        --workspace segm_ee_dist
        # --pretrained_weights pretrained_weights/384_coco_r50.pth
elif [ $1 == 'segm_ee_dist_smallds' ] 
then
    ee=3
    echo "Early-exit from $ee layer"
    python -m torch.distributed.launch --nproc_per_node=4 \
    main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_test_segm_ee_"$ee"_fullytrained_dist_from_pretrained_smallmodel \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_small \
        --workspace segm_ee_smallds \
        --pretrained_weights ckpts/small_model_checkpoint_fullytrained/checkpoint.pth
else
    echo "Not supported argument"
fi