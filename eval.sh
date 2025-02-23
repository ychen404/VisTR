if [ $# == 0 ]
then
    echo "missing input argument"
elif [ $1 == 'inference' ]
then
    python inference.py \
    --masks \
    --backbone resnet50 \
    --model_path pretrained_weights/vistr_r50.pth \
    --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
    --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
    --save_path results/results.json
elif [ $1 == 'inference_debug' ]
then
    python -m pdb inference.py \
    --masks \
    --backbone resnet50 \
    --model_path pretrained_weights/vistr_r50.pth \
    --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
    --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
    --save_path results/results.json
elif [ $1 == 'ee_inference' ]
then
    for i in 1
    do
    ee=4
    echo "eval exit from $ee layer"
    # taskset -p -c 0 $$ 
    echo "start $i times"
        python inference_ee.py \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --backbone resnet50 \
        --model_path ckpts/segm_ee_smallds_dist_05072023_again/checkpoint.pth \
        --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
        --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
        --save_path results/ee_"$ee"_segm_ee_smallds_dist_05072023_again_multigpu_results.json
        # --save_path results/ee_"$ee"_tinyds_results_from_scratch.json
        # --save_path results/ee_"$ee"_tinyds_results.json
        # --save_path results/ee_results.json >> inference_test_ee_"$ee".txt
        # --model_path ckpts/res50_test_segm_ee_"$ee"/checkpoint.pth \
        # --model_path ckpts/res50_segm_ee_5_tinyds_from_scratch/checkpoint.pth \
        # --model_path ckpts/res50_segm_ee_5_tinyds_again/checkpoint.pth \
    done
elif [ $1 == 'ee_inference_debug' ]
then
    for i in 1
    do
    ee=2
    echo "eval exit from $ee layer"
    # taskset -p -c 0 $$ 
    echo "start $i times"
        python -m pdb inference_ee.py \
        --masks \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --backbone resnet50 \
        --model_path ckpts/res50_test_segm_ee_"$ee"/checkpoint.pth \
        --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
        --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json
    done
elif [ $1 == 'ee_inference_multiple' ]
then
    for i in 1
    do
        for j in {0..5}
            do 
            ee=$j
            echo "eval exit from $ee layer, single GPU"
            # taskset -p -c 0 $$ 
            echo "start $i times"
                CUDA_VISIBLE_DEVICES=0 python inference_ee.py \
                --num_frames 3 \
                --num_queries 15 \
                --masks \
                --early_exit_layer $ee \
                --backbone resnet50 \
                --model_path ckpts/segm_ee_smallds_dist_05072023_again/checkpoint.pth \
                --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
                --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
                --save_path results/ee_"$ee"_segm_ee_smallds_dist_05072023_again_multigpu_results.json
                # --save_path results/ee_"$ee"_again_results.json >> inference_test_ee_"$ee"_tinyds_merged_from_tinyds_pretrained_multigpu.txt
            done
    done
elif [ $1 == 'ee_inference_cpu' ]
then
    for i in 1 2 3
    do
        for j in {0..5}
            do 
            ee=$j
            echo "eval exit from $ee layer, single GPU"
            # taskset -p -c 0 $$ 
            echo "start $i times"
                python inference_ee.py \
                --device cpu \
                --num_frames 3 \
                --num_queries 15 \
                --early_exit_layer $ee \
                --backbone resnet50 \
                --model_path ckpts/res50_ee_"$ee"_1_batch_test/checkpoint.pth \
                --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
                --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
                --save_path results/ee_results.json >> inference_test_ee_"$ee"_1_batch.txt
            done
    done
else
    echo "Not supported argument"
fi