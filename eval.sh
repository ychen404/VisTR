if [ $# == 0 ]
then
    echo "missing input argument"
elif [ $1 == 'reference' ]
then
    python inference.py \
    --masks \
    --backbone resnet50 \
    --model_path pretrained_weights/vistr_r50.pth \
    --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
    --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
    --save_path results/results.json
elif [ $1 == 'ee' ]
then
    python inference_ee.py \
        --num_frames 3 \
        --num_queries 15 \
        --backbone resnet50 \
        --model_path res50_ee_0_1_batch/checkpoint.pth \
        --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
        --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
        --save_path results/ee_results.json
elif [ $1 == 'ee_test' ]
then
    for i in 1 2 3
    do
    echo "start $i times"
        python inference_ee.py \
            --num_frames 3 \
            --num_queries 15 \
            --backbone resnet50 \
            --model_path res50_ee_4_1_batch/checkpoint.pth \
            --img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
            --ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
            --save_path results/ee_results.json
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