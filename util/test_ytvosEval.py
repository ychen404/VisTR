# %%
import json
import matplotlib.pyplot as plt
from pycocotools.ytvos import YTVOS
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# %%

res_path = '/home/users/yitao/Code/VisTR/results/tinyds_baseline_from_model_pretrained_on_smallds.json'
f = open(res_path, 'r')
data = json.load(f)

# %%

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print (f'Running demo for *%s* results.'%(annType))

# %%

#initialize COCO ground truth api
# dataDir='../'
# dataType='val2017'
# annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
annFile = '/home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_val_sub.json'
# cocoGt=YTVOS(annFile)
train_annFile = '/home/users/yitao/Code/IFC/datasets/ytvis_2019/annotations/instances_train_sub.json'
cocoGt=YTVOS(train_annFile)

# %%
import json
data = open(annFile, 'r')
ann_data = json.load(data)

# %%
ann_data.keys()

with open(train_annFile, 'r') as f:
    train_ann_data = json.load(f)

# %%
print(f"ann_data keys: {ann_data.keys()},\
      train_ann_data keys: {train_ann_data.keys()}")
"""
There is no 'annotations' in the validation dataset. 
That could be the reason why the ytvosEval is reporting -1. 
"""
# %%
resFile=res_path
cocoDt=cocoGt.loadRes(resFile)

from pycocotools.ytvoseval import YTVOSeval
# %%
E = YTVOSeval(cocoGt,cocoDt)
# %%
vidIds=sorted(cocoGt.getVidIds())
vidIds=vidIds[0:100]

# %%
ytvosEval=YTVOSeval(cocoGt, cocoDt, annType)
# %%
ytvosEval.params.vidIds=vidIds
ytvosEval.evaluate()
ytvosEval.accumulate()
ytvosEval.summarize()
# %%
