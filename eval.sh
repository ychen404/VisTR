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
elif [ $1 == 'ee' ]
then
    ee=2
    echo "eval exit from $ee layer"
    python inference_ee.py \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --backbone resnet50 \
        --model_path res50_ee_"$ee"_1_batch_test/checkpoint.pth \
        --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
        --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
        --save_path results/ee_results.json
        # --model_path res50_ee_"$exit"_1_batch_testtest/checkpoint.pth \
elif [ $1 == 'ee_debug' ]
then
    ee=1
    echo "eval exit from $ee layer"
    python -m pdb inference_ee.py \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --backbone resnet50 \
        --model_path test_print/checkpoint.pth \
        --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
        --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
        --save_path results/ee_results.json
elif [ $1 == 'ee_test' ]
then
    for i in 1 2 3
    do
    ee=2
    echo "eval exit from $ee layer"
    # taskset -p -c 0 $$ 
    echo "start $i times"
        python inference_ee.py \
        --num_frames 3 \
        --num_queries 15 \
        --early_exit_layer $ee \
        --backbone resnet50 \
        --model_path res50_ee_"$ee"_1_batch_test/checkpoint.pth \
        --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
        --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
        --save_path results/ee_results.json >> inference_test_ee_"$ee".txt
    done
elif [ $1 == 'ee_test_all' ]
then
    for i in 1 2 3
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
                --early_exit_layer $ee \
                --backbone resnet50 \
                --model_path res50_ee_"$ee"_1_batch_test/checkpoint.pth \
                --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
                --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
                --save_path results/ee_results.json >> inference_test_ee_"$ee".txt
            done
    done
elif [ $1 == 'ee_test_all_cpu' ]
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
                --model_path res50_ee_"$ee"_1_batch_test/checkpoint.pth \
                --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
                --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
                --save_path results/ee_results.json >> inference_test_ee_"$ee"_1_batch.txt
            done
    done
elif [ $1 == 'all_layers' ]
then
    python inference_ee.py \
        --num_frames 3 \
        --num_queries 15 \
        --backbone resnet50 \
        --model_path res50_ee_all_layers_1_batch/checkpoint.pth \
        --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
        --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
        --save_path results/ee_results.json
else
    echo "Not supported argument"
fi