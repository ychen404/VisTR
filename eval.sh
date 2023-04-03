python inference.py \
--masks \
--backbone resnet50 \
--model_path pretrained_weights/vistr_r50.pth \
--img_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/val/JPEGImages/ \
--ann_path /home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json \
--save_path results/results.json