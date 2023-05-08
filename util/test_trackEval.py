#%%
import json


# %%
path = '../TrackEval_data/data/trackers/youtube_vis/youtube_vis_train_sub_split/STEm_Seg/data/results.json'
f = open(path, 'r')
data = json.load(f)

# %%
data.keys()
# %%
data[0].keys()
# %%
len(data)
# %%

res = "/home/users/yitao/Code/VisTR/results/ee_6_tinyds_from_tinyds_pretrained_multigpu_results.json"
with open(res, 'r') as f:
    res_data = json.load(f)

# %%
res_data[0].keys()
# %%


res_data[2]
# %%
data[0]
# %%



# %%
