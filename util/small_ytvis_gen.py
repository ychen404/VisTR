"""
This script produces a small dataset from ytvis2019

It generates the small dataset using the following two steps:

1) Reads all the video names 
2) Randomly

"""

import json
import random
import copy

dataset_path = "/home/users/yitao/Code/IFC/datasets/ytvis_2019_small/"
file = open(dataset_path + 'annotations/instances_train_sub_full.json')
data = json.load(file)
# print(len(data['videos']))

res = copy.deepcopy(data)
res['videos'] = []

foldernames = []
for item in data['videos']:
    name = item['file_names'][0].split('/')[0]
    foldernames.append(name)


print(f"total entries: {len(foldernames)}")
# print(len(foldernames))

out = random.sample(foldernames, 200)
# test_out = [out[0]]

neg = 0
pos = 0
for idx, item in enumerate(data['videos']):
    name = item['file_names'][0].split('/')[0]
    if name in out:
        # print("found one useful entry")
        pos += 1
        res['videos'].append(item)
    if name not in out:
        neg += 1
        # del data['videos'][idx]
        # print(f"Deleting {name}")

print(f"total neg {neg}, pos {pos}")
# breakpoint()

assert len(res['videos']) == 200, "the length is wrong"

with open('instances_train_sub.json', 'w', encoding='utf-8') as f:
    json.dump(res,f)

