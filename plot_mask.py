#%%
import numpy as np

masks_path = "/home/users/yitao/Code/VisTR/npy_for_mask_plot/masks.npy"
masks = np.load(masks_path)

logit_path = "/home/users/yitao/Code/VisTR/npy_for_mask_plot/pred_logits.npy"
logit = np.load(logit_path)

pred_mask_path = "/home/users/yitao/Code/VisTR/npy_for_mask_plot/pred_masks.npy"
pred_mask = np.load(pred_mask_path)

img_path = "/home/users/yitao/Code/VisTR/npy_for_mask_plot/img.npy"
img = np.load(img_path)


# %%
import torch

mask_tensor = torch.tensor(masks)
# output = data.transpose((1,2,0))

#%%

num_frames = 36
num_ins = 10
print(f"masks shape: {masks.shape}")

mask_tensor = mask_tensor.reshape(num_frames,num_ins,masks.shape[-2],masks.shape[-1])

#%%

#%%
import matplotlib.pyplot as plt

print(pred_mask.shape)
type(pred_mask)
# plt.imshow(pred_mask[0][0])
np.count_nonzero(pred_mask)
# plt.imshow(pred_mask[3][1])

#%%

idx = 5
image = img[idx].transpose(1,2,0)
image = (image * 255).astype('uint32')


#%%

# maxValue = np.amax(image)
# minValue = np.amin(image)
# print(f"max: {maxValue}, min: {minValue}")

# image = image / maxValue
# image = np.clip(image, 0, 1)

# print("after normalization")
# maxValue = np.amax(image)
# minValue = np.amin(image)
# print(f"max: {maxValue}, min: {minValue}")

# image = image * 255

# maxValue = np.amax(image)
# minValue = np.amin(image)
# print(f"max: {maxValue}, min: {minValue}")

#%%
# print(image);
plt.imshow(image)


# %%

for i in range(5):
    img_transpose = img[i].transpose((1,2,0))
    plt.imshow(img_transpose)
# %%
import matplotlib.pyplot as plt

# %%
plt.figure(figsize=(20, 4))
n = 8

for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(img[i].transpose((1,2,0)))
  plt.title("original")
#   plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  
  ax = plt.subplot(2, n, i + 1 + n )
  plt.imshow(pred_mask[i].transpose((1,2,0)).argmax(axis=2))
  plt.title("mask")
#   plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

# %%

pred_mask[5].shape
# plt.imshow(pred_mask[5].transpose((1,2,0)).argmax(axis=2))

plt.imshow(pred_mask[5][7])
# pred_mask[5][0].shape
# plt.savefig('mask.jpg')
# %%

# fig, (ax1, ax2) = plt.subplots(1,2)
# plt.figure(figsize=(100, 40))
# for i in range(10):
#     ax = plt.subplot(1, n, i+1)
#     ax.imshow(pred_mask[5][i])

# %%
import cv2

mask_needed = pred_mask[5][0]
# type(mask_needed)
plt.imshow(mask_needed)

# %%

cv2.imwrite("mask_needed_skateboard.png", mask_needed * 255)

# %%
