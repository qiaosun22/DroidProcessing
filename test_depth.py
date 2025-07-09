import os
import cv2
import numpy as np
import torch
from d2n import intrins_to_intrins_inv, get_cam_coords, d2n_tblr, normal_to_rgb

depth_path = (
    "ext1_depth_0.npz"
    # "/media/sun/12TB/datasets/droid/droid_raw/processed_0421/AUTOLab/success/2023-11-28/Tue_Nov_28_08:46:19_2023/frames/ext1_depth_0.npz"
    # "../AgiBotWorld-Alpha/observations/352/648544/depth/head_depth_000000.png"
    #"rlbench_4d/train_dataset/microsteps/seed100/take_lid_off_saucepan/variation0/episodes/episode0/front_depth/0.png"
)


# path = "/media/sun/12TB/datasets/droid/droid_raw/processed_0421/AUTOLab/success/2023-11-28/Tue_Nov_28_08:46:19_2023/frames/ext1_depth_0.npz"
metric_depth = np.load(depth_path)['arr_0']

metric_depth = np.nan_to_num(metric_depth, nan=0).astype(np.float64)
np.clip(metric_depth, 1e-2, 2e3, out=metric_depth)
# metric_depth.min(), metric_depthmax()    
# metric_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64) #cv2.cvtColor(cv2.imread(depth_path), cv2.COLOR_BGR2RGB).astype(np.float64)

print("metric_depth.shape", metric_depth.shape)
print(metric_depth.max(), metric_depth.min())

# r = g = b = metric_depth
# print(r.shape, g.shape, b.shape)
depth = metric_depth
# metric_depth = depth / 256000
# print(metric_depth.max(), metric_depth.min())
depth = (depth / depth.max() * 255.0).astype(np.uint8)
# save image to tmp
# cv2.imwrite("tmp/depth.png", depth)



# depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)# cv2.cvtColor(cv2.imread(depth_path), cv2.COLOR_BGR2RGB).astype(np.float64)
# print("depth_image.shape", depth_image.shape)

# r= g= b = depth_image#cv2.split(depth_image)
# print(r.shape, g.shape, b.shape)
# # depth = r * 256 * 256 + g * 256 + b
# # metric_depth = depth / 256000
# depth = r + g + b
# # depth = depth_image #r + g + b
# # metric_depth = depth

# print(metric_depth.max(), metric_depth.min())
# depth = (depth / depth.max() * 255.0).astype(np.uint8)
# # save image to tmp
# cv2.imwrite("tmp/depth.png", depth)

intrins = np.array(
    [
        [-703.3542416, 0.0, 256.0],
        [0.0, -703.3542416, 256.0],
        [0.0, 0.0, 1.0]
    ]
)
intrins_inv = intrins_to_intrins_inv(intrins)
intrins_inv = torch.tensor(intrins_inv).unsqueeze(0).to(0)
metric_depth = torch.tensor(metric_depth).unsqueeze(0).unsqueeze(0).to(0)
points = get_cam_coords(intrins_inv, metric_depth)
normal, valid_mask = d2n_tblr(points, k=5, d_min=1e-3, d_max=1e8)
normal_rgb = normal_to_rgb(normal, valid_mask)[0, ...]

cv2.imwrite("saved/normal.png", cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR))



# import os
# import cv2
# import numpy as np
# import torch
# from utils.d2n import intrins_to_intrins_inv, get_cam_coords, d2n_tblr, normal_to_rgb

# depth_path = (
#     "rlbench_4d/train_dataset/microsteps/seed100/take_lid_off_saucepan/variation0/episodes/episode0/front_depth/0.png"
# )
# depth_image = cv2.cvtColor(cv2.imread(depth_path), cv2.COLOR_BGR2RGB).astype(np.float32)
# r, g, b = cv2.split(depth_image)
# print(r.shape, g.shape, b.shape)
# depth = r * 256 * 256 + g * 256 + b
# metric_depth = depth / 256000
# print(metric_depth.max(), metric_depth.min())
# depth = (depth / depth.max() * 255.0).astype(np.uint8)
# # save image to tmp
# cv2.imwrite("tmp/depth.png", depth)

# intrins = np.array(
#     [
#         [-703.3542416, 0.0, 256.0],
#         [0.0, -703.3542416, 256.0],
#         [0.0, 0.0, 1.0],
#     ]
# )
# intrins_inv = intrins_to_intrins_inv(intrins)
# intrins_inv = torch.from_numpy(intrins_inv).unsqueeze(0).to(0)
# metric_depth = torch.from_numpy(metric_depth).unsqueeze(0).unsqueeze(0).to(0)
# points = get_cam_coords(intrins_inv, metric_depth)
# normal, valid_mask = d2n_tblr(points, k=5, d_min=1e-3, d_max=1000.0)
# normal_rgb = normal_to_rgb(normal, valid_mask)[0, ...]

# cv2.imwrite("tmp/normal.png", cv2.cvtColor(normal_rgb, cv2.COLOR_RGB2BGR))
