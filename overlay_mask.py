#%%
from PIL import Image
import matplotlib.pyplot as plt

frame1 = "/home/users/yitao/Code/IFC/datasets/ytvis_2019_tiny/val/JPEGImages/00f88c4f0a/00025.jpg"
mask = "/home/users/yitao/Code/VisTR/mask_needed_ee_4.png"
transparent = "girl_on_black_transparent.png"

# %%
import numpy as np

im = Image.open(frame1)
ma = Image.open(mask)
tr = Image.open(transparent)

plt.imshow(im, interpolation='none')
plt.imshow(ma, interpolation='none', alpha=0.5)
# plt.imshow(tr, cmap='rgb')
# plt.imshow(np.abs(ma), cmap=plt.get_cmap('gray'),vmin=0,vmax=255, alpha=0.5)
plt.show()




# %%
##### Convert black pixels to transparent
import cv2
import numpy as np

# load image
img = cv2.imread('/home/users/yitao/Code/VisTR/mask_needed_ee_4.png')

# %%

# threshold on black to make a mask
color = (0,0,0)
mask = np.where((img==color).all(axis=2), 0, 255).astype(np.uint8)

white = (255,255,255)
white_mask = np.where((img == white).all(axis=2),0,255).astype(np.uint8)
# put mask into alpha channel
result = img.copy()
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask

# save resulting masked image
# cv2.imwrite('girl_on_black_transparent.png', result)

# # display result, though it won't show transparency
# cv2.imshow("MASK", mask)
# cv2.imshow("RESULT", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# %%
white_mask

# %%
result.shape
# %%
white_mask.shape
# %%
type(white_mask)
# %%
color_mask = white_mask / 2 
# %%

import matplotlib.pyplot as plt
plt.imshow(color_mask)
# %%