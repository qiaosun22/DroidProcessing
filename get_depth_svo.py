########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample demonstrates how to capture a live 3D point cloud   
    with the ZED SDK and display the result in an OpenGL window.    
"""

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


# def parse_args(init):
#     if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo"):
#         init.set_from_svo_file(opt.input_svo_file)
#         print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
#     elif len(opt.ip_address) > 0:
#         ip_str = opt.ip_address
#         if (
#             ip_str.replace(":", "").replace(".", "").isdigit()
#             and len(ip_str.split(".")) == 4
#             and len(ip_str.split(":")) == 2
#         ):
#             init.set_from_stream(ip_str.split(":")[0], int(ip_str.split(":")[1]))
#             print("[Sample] Using Stream input, IP : ", ip_str)
#         elif (
#             ip_str.replace(":", "").replace(".", "").isdigit()
#             and len(ip_str.split(".")) == 4
#         ):
#             init.set_from_stream(ip_str)
#             print("[Sample] Using Stream input, IP : ", ip_str)
#         else:
#             print("Unvalid IP format. Using live stream")
#     if "HD2K" in opt.resolution:
#         init.camera_resolution = sl.RESOLUTION.HD2K
#         print("[Sample] Using Camera in resolution HD2K")
#     elif "HD1200" in opt.resolution:
#         init.camera_resolution = sl.RESOLUTION.HD1200
#         print("[Sample] Using Camera in resolution HD1200")
#     elif "HD1080" in opt.resolution:
#         init.camera_resolution = sl.RESOLUTION.HD1080
#         print("[Sample] Using Camera in resolution HD1080")
#     elif "HD720" in opt.resolution:
#         init.camera_resolution = sl.RESOLUTION.HD720
#         print("[Sample] Using Camera in resolution HD720")
#     elif "SVGA" in opt.resolution:
#         init.camera_resolution = sl.RESOLUTION.SVGA
#         print("[Sample] Using Camera in resolution SVGA")
#     elif "VGA" in opt.resolution:
#         init.camera_resolution = sl.RESOLUTION.VGA
#         print("[Sample] Using Camera in resolution VGA")
#     elif len(opt.resolution) > 0:
#         print("[Sample] No valid resolution entered. Using default")
#     else:
#         print("[Sample] Using default resolution")


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
    episode_dirs = glob.glob(f"{data_root}/success/2023-11-25/*")    # eg: scripts/make_data/droid/droid_data/droid_raw/1.0.1/CLVR/success/2023-05-09/Tue_May__9_01:17:11_2023
    process_map(handle_single_episode_dir, episode_dirs, max_workers=20)
    # svo_file = f"droid_data/example/Tue_May__9_04:54:19_2023/recordings/SVO/20103212.svo"
    # save_rgbd_from_svo(svo_file)
