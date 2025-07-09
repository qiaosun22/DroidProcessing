import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import pyzed.sl as sl
import os
import time
# from scipy.spatial.transform import Rotation



def get_intrinsic_parameters(svo_filepath):
    # 创建 ZED 相机对象
    zed = sl.Camera()

    # 设置初始化参数
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_filepath)
    init_params.svo_real_time_mode = False  # 不实时播放

    # 打开相机
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        # print("打开 SVO 文件失败: ", err)
        return

    # 获取相机信息
    camera_info = zed.get_camera_information()
    
    # print("\n=== 左摄像头内参 ===")
    left_cam_params = camera_info.camera_configuration.calibration_parameters.left_cam
    # # print(f"分辨率: {left_cam_params.image_size}")
    # resolution = left_cam_params.image_size
    # print(f"分辨率: {resolution.width} x {resolution.height}")
    # print(f"焦距 (fx, fy): {left_cam_params.fx}, {left_cam_params.fy}")
    # print(f"光心 (cx, cy): {left_cam_params.cx}, {left_cam_params.cy}")
    # # print(f"畸变系数: {list(left_cam_params.distortion_parameters)}")

    # print("\n=== 右摄像头内参 ===")
    right_cam_params = camera_info.camera_configuration.calibration_parameters.right_cam
    # # print(f"分辨率: {right_cam_params.image_size}")
    # resolution = left_cam_params.image_size
    # print(f"分辨率: {resolution.width} x {resolution.height}")
    # print(f"焦距 (fx, fy): {right_cam_params.fx}, {right_cam_params.fy}")
    # print(f"光心 (cx, cy): {right_cam_params.cx}, {right_cam_params.cy}")

    zed.close()
    return left_cam_params, right_cam_params

def to_homogeneous_transform(pose_6d):
    """
    将一个 [x, y, z, rx, ry, rz] 格式的 6-DoF 位姿转换为 4x4 齐次变换矩阵
    假设旋转是轴角格式（Axis-angle）
    """
    translation = pose_6d[:3]
    rot_vector = pose_6d[3:]

    # 使用轴角构造旋转对象
    # rotation = R.from_rotvec(rot_vector)

    rotation = R.from_euler("xyz", np.array(rot_vector))#.as_matrix()

    # 构造 4x4 变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = translation

    return transform

def project_to_image(point_world, camera_T_world, K):
    """
    point_world: 世界坐标 (3,)
    camera_T_world: 世界到相机的变换矩阵 (4,4)
    K: 内参矩阵 (3,3)
    返回像素坐标 (u, v)
    """
    # 齐次坐标转换
    point_world_homogeneous = np.append(point_world, 1)
    point_camera_homogeneous = camera_T_world @ point_world_homogeneous
    point_camera = point_camera_homogeneous[:3]

    if point_camera[2] <= 0:
        return None  # 点在相机后方，无法投影

    # 投影到图像平面
    u = K[0, 0] * point_camera[0] / point_camera[2] + K[0, 2]
    v = K[1, 1] * point_camera[1] / point_camera[2] + K[1, 2]

    return (u, v)


# date = "2023-11-24" #"2023-11-13" #"2023-09-05"#"2023-08-12"#"2023-10-27" #"2023-08-12" #"2023-11-30"
# timestemp = "Fri_Nov_24_18:58:47_2023" #"Mon_Nov_13_15:55:29_2023" #"Tue_Sep__5_08:50:21_2023" #"Fri_Jul__7_14:58:30_2023" #"Sat_Aug_12_12:15:52_2023"#"Fri_Oct_27_19:49:00_2023" # #"Thu_Nov_30_07:37:44_2023" #"Fri_Jul__7_09:44:34_2023" #"Fri_Jul__7_09:55:14_2023" #"Fri_Jul__7_09:44:34_2023" #"Fri_Jul__7_09:43:39_2023" #"Fri_Jul__7_09:42:23_2023"

date = "2023-11-21" #"2023-11-24" #"2023-11-13" #"2023-09-05"#"2023-08-12"#"2023-10-27" #"2023-08-12" #"2023-11-30"
timestemp = "Tue_Nov_21_11:23:03_2023"#"Tue_Nov_21_08:51:58_2023" #"Sat_Nov_25_10:06:22_2023" #"Fri_Nov_17_20:30:02_2023" #"Fri_Nov_24_18:42:53_2023" #"Mon_Nov_13_15:55:29_2023" #"Tue_Sep__5_08:50:21_2023" #"Fri_Jul__7_14:58:30_2023" #"Sat_Aug_12_12:15:52_2023"#"Fri_Oct_27_19:49:00_2023" # #"Thu_Nov_30_07:37:44_2023" #"Fri_Jul__7_09:44:34_2023" #"Fri_Jul__7_09:55:14_2023" #"Fri_Jul__7_09:44:34_2023" #"Fri_Jul__7_09:43:39_2023" #"Fri_Jul__7_09:42:23_2023"
# camera = "24400334" #"22008760"#"24400334" #"22008760" #"24400334"

camera = "22008760"#"24400334" #"22008760"#"24400334" #"22008760" #"24400334"
left_or_right = "right" #"left"
file_path = f'droid_raw/1.0.1/AUTOLab/success/{date}/{timestemp}/trajectory.h5'

# 把视频拆分成帧
video_path = f'droid_raw/1.0.1/AUTOLab/success/{date}/{timestemp}/recordings/MP4/{camera}.mp4'
frame_save_path = f'droid_raw/1.0.1/AUTOLab/success/{date}/{timestemp}/recordings/MP4/{camera}_{left_or_right}/'

print(f"视频路径: {video_path}")
print(f"帧保存路径: {frame_save_path}")

if not os.path.exists(video_path):
    print(f"视频文件 {video_path} 不存在，请检查路径。")
if not os.path.exists(frame_save_path):
    os.makedirs(frame_save_path)
# else:
    # os.system(f"ffmpeg -i {video_path} -vf fps=30 {frame_save_path}/frame_%05d.png")
# 使用默认fps
os.system(f"ffmpeg -i {video_path} {frame_save_path}/frame_%05d.png")
    

time.sleep(10)  # 等待文件系统更新

plt.close('all')

with h5py.File(file_path, 'r') as f:
    # 获取 action/cartesian_position 数据
    cartesian_pos = f['action/cartesian_position'][()]
    camera_extrinsic = f[f'observation/camera_extrinsics/{camera}_{left_or_right}'][()]

svo_filepath = f"droid_raw/1.0.1/AUTOLab/success/{date}/{timestemp}/recordings/SVO/{camera}.svo"

left_cam_params, right_cam_params = get_intrinsic_parameters(svo_filepath)
if left_or_right == "left":
    cam_params = left_cam_params
else:
    cam_params = right_cam_params

fx, fy = cam_params.fx, cam_params.fy
cx, cy = cam_params.cx, cam_params.cy
    
for i in tqdm(range(cartesian_pos.shape[0] - 1)):
    # 示例：对第一个时间步进行转换
    cartesian_pose_matrix = to_homogeneous_transform(cartesian_pos[i])
    camera_extrinsic_matrix = to_homogeneous_transform(camera_extrinsic[i])

    # print("Cartesian Pose (4x4):\n", cartesian_pose_matrix)
    # print("\nCamera Extrinsic (4x4):\n", camera_extrinsic_matrix)

    # Step 1: 获取末端位置（平移部分）
    robot_position_world = cartesian_pose_matrix[:3, 3]  # 取自你提供的 world_T_robot 的最后一列前三行

    # Step 2: 将其从世界坐标系变换到相机坐标系
    world_T_camera = camera_extrinsic_matrix  # 即你提供的 camera_T_world 的 inverse
    camera_T_world = np.linalg.inv(world_T_camera)

    # 构造齐次坐标点 [x, y, z, 1]
    point_world_homogeneous = np.append(robot_position_world, 1)
    point_camera_homogeneous = camera_T_world @ point_world_homogeneous
    point_camera = point_camera_homogeneous[:3]

    # 如果深度为负（在相机背后），则不在视野中
    if point_camera[2] < 0:
        print("该点在相机后方，无法投影。")
    else:
        # Step 3: 使用相机内参进行投影
        # fx, fy = 531.7265014648438, 531.7265014648438
        # cx, cy = 636.1519775390625, 344.0089416503906

        u = fx * point_camera[0] / point_camera[2] + cx
        v = fy * point_camera[1] / point_camera[2] + cy

        # # Step 4: 检查是否在图像范围内
        # if 0 <= u < 1280 and 0 <= v < 720:
        #     print(f"末端中心在图像上的像素坐标为: ({u:.2f}, {v:.2f})")
        # else:
        #     print(f"末端中心在图像外: ({u:.2f}, {v:.2f})")


    # 从 world_T_robot 中提取平移和旋转
    position = cartesian_pose_matrix[:3, 3]         # 世界坐标系下的位置
    rotation = cartesian_pose_matrix[:3, :3]        # 3x3 旋转矩阵

    length = 0.1  # 轴线长度（米）

    # 本地坐标系下的点（分别沿 x, y, z 方向延伸 length 米）
    local_x_axis_point = np.array([length, 0, 0])
    local_y_axis_point = np.array([0, length, 0])
    local_z_axis_point = np.array([0, 0, length])

    # 计算这些点在世界坐标系中的位置
    world_x = position + rotation @ local_x_axis_point
    world_y = position + rotation @ local_y_axis_point
    world_z = position + rotation @ local_z_axis_point


    # fx, fy = 531.7265014648438, 531.7265014648438
    # cx, cy = 636.1519775390625, 344.0089416503906

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ]
    ])


    # 获取相机到世界的变换
    camera_T_world = np.linalg.inv(camera_extrinsic_matrix)

    # 原点（末端中心）
    origin_pixel = project_to_image(position, camera_T_world, K)

    # 各轴末端点
    x_pixel = project_to_image(world_x, camera_T_world, K)
    y_pixel = project_to_image(world_y, camera_T_world, K)
    z_pixel = project_to_image(world_z, camera_T_world, K)

    # print("Origin (u0, v0):", origin_pixel)
    # print("X-axis end (ux, vx):", x_pixel)
    # print("Y-axis end (uy, vy):", y_pixel)
    # print("Z-axis end (uz, vz):", z_pixel)

    # === 输入参数 ===
    image_path = f"{frame_save_path}/frame_{i+1:05d}.png"

    # 像素坐标（由之前算法得到）
    # origin_pixel = (712.34, 368.12)
    # x_pixel = (752.34, 370.12)
    # y_pixel = (714.34, 408.12)
    # z_pixel = (708.34, 328.12)

    # === 加载图像 ===
    image = plt.imread(image_path)

    # === 创建绘图 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)


    # 在指定位置画一个红色圆圈（半径为10）
    circle = plt.Circle((u, v), radius=20, color='red', fill=False, linewidth=5)
    ax.add_patch(circle)

    # 添加文字说明
    ax.text(u + 15, v, '  Robot EE', color='red', fontsize=20)

    # # 设置标题
    # ax.set_title(f"output_video-{date}_{timestemp}_{camera}_{left_or_right}")



    # === 绘制坐标轴箭头 ===
    # x-axis - Red
    ax.annotate('', xy=x_pixel, xytext=origin_pixel,
                arrowprops=dict(facecolor='red', edgecolor='red', lw=2, headwidth=5, headlength=5))

    # y-axis - Green
    ax.annotate('', xy=y_pixel, xytext=origin_pixel,
                arrowprops=dict(facecolor='green', edgecolor='green', lw=2, headwidth=5, headlength=5))

    # z-axis - Blue
    ax.annotate('', xy=z_pixel, xytext=origin_pixel,
                arrowprops=dict(facecolor='blue', edgecolor='blue', lw=2, headwidth=5, headlength=5))


    # === 添加图例说明 ===
    legend_elements = [
        mpatches.Patch(color='red', label='X-axis'),
        mpatches.Patch(color='green', label='Y-axis'),
        mpatches.Patch(color='blue', label='Z-axis')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # === 设置标题和关闭坐标轴 ===
    # ax.set_title("End-Effector Orientation Projected on Image")
    ax.set_title(f"output_video-{date}_{timestemp}_{camera}_{left_or_right}")
    plt.axis('off')


    output_dir = f"droid_raw/1.0.1/AUTOLab/success/{date}/{timestemp}/recordings/MP4/{camera}_{left_or_right}_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === 显示图像 ===
    plt.tight_layout()
    # plt.show()
    
    plt.savefig(f"{output_dir}/output_image_{i+1:05d}.png")
    
    plt.close(fig)

    # plt.show()
    # if i > 3:
    #     break

# 把帧合并成视频
# os.system(
#     f"ffmpeg -framerate 30 -i {output_dir}/output_image_%05d.png  -vf 'fps=30,scale=320:180:flags=lanczos,palettegen' -c:v gif -gifflags +transdiff output_video-{date}_{timestemp}_{camera}_{left_or_right}.gif -y"
#           )
# # ffmpeg -framerate 30 -i {output_dir}/output_image_%05d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4
# 设置参数
input_fps = 30
output_fps = 10
scale_width = 320   # 输出宽度（高度自动按比例缩放）

# Step 1: 生成调色板（缩放 + 抽帧）
os.system(
    f"ffmpeg -framerate {input_fps} -i {output_dir}/output_image_%05d.png "
    f'-vf "scale={scale_width}:-1:flags=lanczos,fps={output_fps},palettegen" '
    f'palette.png -y'
)

# Step 2: 使用调色板生成 GIF（再次缩放 + 抽帧）
os.system(
    f"ffmpeg -framerate {input_fps} -i {output_dir}/output_image_%05d.png "
    f"-i palette.png "
    f'-lavfi "[0]scale={scale_width}:-1:flags=lanczos,fps={output_fps}[img];[img][1]paletteuse" '
    f'-gifflags +transdiff '
    f"output_video-{date}_{timestemp}_{camera}_{left_or_right}.gif -y"
)