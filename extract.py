import sys

# import ogl_viewer.viewer as gl
import pyzed.sl as sl
import argparse
import numpy as np
from PIL import Image
import glob
import os
from tqdm import tqdm
import json
from tqdm.contrib.concurrent import process_map

# zed = sl.Camera()

RES = sl.Resolution()
RES.width = 640
RES.height = 360



def save_rgbd_from_svo(svo_file, save_dir, cam_type):
    init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.ULTRA,
        coordinate_units=sl.UNIT.MILLIMETER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, 
    )
    init.set_from_svo_file(svo_file)
    zed = sl.Camera()
    status = zed.open(init)


    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # save the first frame
    os.makedirs(save_dir, exist_ok=True)
    # save the last frame
    index = zed.get_svo_number_of_frames()
    # the origin fps is 60. We sample using 6 fps
    for i in range(0, index - 2, 10):
        zed.set_svo_position(i)
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            get_rgbd_and_save(zed, i, save_dir, cam_type)
    zed.close()


def get_rgbd_and_save(zed, index, save_dir, cam_type):
    image = sl.Mat()
    depth_map = sl.Mat()
    depth_for_display = sl.Mat()

    zed.retrieve_image(image, sl.VIEW.LEFT, resolution=RES)  # Retrieve left image
    zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH, resolution=RES)  # Retrieve depth
    # zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH, resolution=RES)

    # save left iamge and depth
    image.write(f"{save_dir}/{cam_type}_rgb_{index}.png")

    # depth_for_display.write(f"{save_dir}/depth_dis_{index}.png")

    depth_numpy_data = depth_map.get_data()
    np.savez_compressed(f"{save_dir}/{cam_type}_depth_{index}.npz", depth_numpy_data)
    # save_depth_image(depth_numpy_data, f"{save_dir}/depth_{index}.png")

    print(f"save image for index {index}")


def save_depth_image(depth_array, output_file, max_depth_mm=5000, bit_depth=8):
    """
    Saves a depth image in millimeters to a PNG file.

    Parameters:
        depth_array (numpy.ndarray): A 2D numpy array containing depth data in millimeters.
        output_file (str): Path to the output PNG file.
        max_depth_mm (int): The maximum depth expected in the image, in millimeters.
        bit_depth (int): The bit depth for the output image (8 or 16).
    """
    # Normalize the depth values to the range 0-1
    normalized_depth = np.clip(depth_array, 0, max_depth_mm) / max_depth_mm
    normalized_depth = 1 - normalized_depth

    # Scale to the appropriate range for the bit depth
    if bit_depth == 8:
        max_value = 255
    elif bit_depth == 16:
        max_value = 65535
    else:
        raise ValueError("Bit depth must be 8 or 16")

    # Convert to an integer data type
    scaled_depth = (normalized_depth * max_value).astype(
        np.uint16 if bit_depth == 16 else np.uint8
    )

    # Create and save the image
    img = Image.fromarray(scaled_depth)
    img.save(output_file)


def handle_single_episode_dir(episode_dir):
    save_dir = episode_dir.replace("1.0.1", "processed_0421")
    save_dir = f"{save_dir}/frames"
    if os.path.exists(save_dir):
        return
    meta_file = glob.glob(f"{episode_dir}/*.json")[0]
    meta_data = json.load(open(meta_file))
    for cam in ["wrist", "ext1", "ext2"]:
        svo_path = meta_data[f"{cam}_svo_path"]
        svo_path = f"{data_root}/{svo_path}"
        save_rgbd_from_svo(svo_path, save_dir, cam)
    tqdm.write(f"save frames in {save_dir}")


if __name__ == "__main__":
    data_root = "droid_raw/1.0.1/AUTOLab" #"../4dgen/tmp/openx/droid/droid_raw/1.0.1/CLVR" #"/home/ubuntu/droid_data/droid_raw/1.0.1/CLVR"
    processed_root = "./"
    episode_dirs = glob.glob(f"{data_root}/success/*/*")    # eg: scripts/make_data/droid/droid_data/droid_raw/1.0.1/CLVR/success/2023-05-09/Tue_May__9_01:17:11_2023
    # process_map(handle_single_episode_dir, episode_dirs, max_workers=20)


    date = "2023-11-21" #"2023-11-24" #"2023-11-13" #"2023-09-05"#"2023-08-12"#"2023-10-27" #"2023-08-12" #"2023-11-30"
    timestemp = "Tue_Nov_21_11:23:03_2023" #"Tue_Nov_21_08:51:58_2023" #"Sat_Nov_25_10:06:22_2023" #"Fri_Nov_17_20:30:02_2023" #"Fri_Nov_24_18:42:53_2023" #"Mon_Nov_13_15:55:29_2023" #"Tue_Sep__5_08:50:21_2023" #"Fri_Jul__7_14:58:30_2023" #"Sat_Aug_12_12:15:52_2023"#"Fri_Oct_27_19:49:00_2023" # #"Thu_Nov_30_07:37:44_2023" #"Fri_Jul__7_09:44:34_2023" #"Fri_Jul__7_09:55:14_2023" #"Fri_Jul__7_09:44:34_2023" #"Fri_Jul__7_09:43:39_2023" #"Fri_Jul__7_09:42:23_2023"
    camera = "22008760"#"24400334" #"22008760"#"24400334" #"22008760" #"24400334"
    left_or_right = "left" 

    svo_filepath = f"droid_raw/1.0.1/AUTOLab/success/{date}/{timestemp}/recordings/SVO/{camera}.svo"

    # svo_file = f"droid_data/example/Tue_May__9_04:54:19_2023/recordings/SVO/20103212.svo"
    save_dir = f"output/{date}/{timestemp}/frames"

    for camera_type in ["wrist", "ext1", "ext2"]:
        # camera_type = "ext1"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        save_rgbd_from_svo(svo_filepath, save_dir, camera_type)


