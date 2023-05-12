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
elif [ $1 == 'segm_ee_smallds' ] # small dataset
then
    ee=5
    echo "Early-exit from $ee layer"
    CUDA_VISIBLE_DEVICES=0 python main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/segm_ee_smallds_05122023 \
        --log logs/segm_ee_smallds.json \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_small \
        --workspace segm_ee_smallds_05122023 \
        --pretrained_weights ckpts/small_model_checkpoint_fullytrained/checkpoint.pth
elif [ $1 == 'segm_ee_smallds_dist' ] # small dataset with ee
then
    ee=5
    echo "Early-exit from $ee layer"
    python -m torch.distributed.launch --nproc_per_node=4 \
        main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/segm_ee_smallds_dist_05122023 \
        --log logs/segm_ee_smallds_dist_05122023.json \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_small \
        --workspace segm_ee_smallds_dist_05122023 \
        --pretrained_weights ckpts/small_model_checkpoint_fullytrained/checkpoint.pth
elif [ $1 == 'segm_ee_tinyds' ] # small dataset
then
    ee=5
    echo "Early-exit from $ee layer"
    CUDA_VISIBLE_DEVICES=1 python main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/res50_segm_ee_"$ee"_tinyds_from_scratch_test_merge \
        --log logs/segm_ee_"$ee"_tinyds_all_segms_from_scratch_test_merge.json \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_tiny \
        --workspace _segm_ee_"$ee"_tinyds_all_segms_from_scratch_test_merge \
        --pretrained_weights pretrained_weights/384_coco_r50.pth 
        # --pretrained_weights ckpts/small_model_checkpoint_fullytrained/checkpoint.pth
        # --pretrained_weights pretrained_weights/384_coco_r50.pth
        # --early_break \
elif [ $1 == 'segm_ee_tinyds_test' ] # tiny dataset
then
    ee=5
    echo "Early-exit from $ee layer"
    CUDA_VISIBLE_DEVICES=1 python main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/segm_ee_tinyds_merged_from_tinyds_pretrained \
        --log logs/segm_ee_tinyds_merged.json \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_tiny \
        --workspace segm_ee_tinyds_merged_from_tinyds_pretrained \
        --pretrained_weights ckpts/tinyds_baseline_from_scratch/checkpoint.pth
        # --pretrained_weights pretrained_weights/384_coco_r50.pth
        # --pretrained_weights pretrained_weights/384_coco_r50.pth
        # --early_break \
elif [ $1 == 'segm_ee_tinyds_multigpu' ] # tiny dataset with multiple gpus
then
    ee=5
    echo "Early-exit from $ee layer"
    python -m torch.distributed.launch --nproc_per_node=4 \
        main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/segm_ee_tinyds_merged_from_tinyds_pretrained_multigpu \
        --log logs/segm_ee_tinyds_merged_multigpu.json \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_tiny \
        --workspace segm_ee_tinyds_merged_from_tinyds_pretrained_multigpu \
        --pretrained_weights ckpts/tinyds_baseline_from_scratch/checkpoint.pth
elif [ $1 == 'segm_ee_tinyds_multigpu_test_meter' ] # tiny dataset with multiple gpus
then
    ee=5
    echo "Early-exit from $ee layer"
    python -m torch.distributed.launch --nproc_per_node=4 \
        main_ee.py --backbone resnet50 \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --output_dir ckpts/segm_ee_tinyds_multigpu_test_meter \
        --log logs/segm_ee_tinyds_merged_multigpu.json \
        --ytvos_path /home/users/yitao/Code/IFC/datasets/ytvis_2019_tiny \
        --workspace segm_ee_tinyds_multigpu_test_meter \
        --early_break \
        --pretrained_weights ckpts/tinyds_baseline_from_scratch/checkpoint.pth
elif [ $1 == 'segm_ee_tinyds_debug' ] # tiny dataset
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
else
    echo "Not supported argument"
fi