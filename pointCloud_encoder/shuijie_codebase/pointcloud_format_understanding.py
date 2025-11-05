"""
Here we try to understand what the pointcloud binary is formatted.
We have example images and we shall apply vggt to them to generate predictions.

prediction is a dictionary with following keys

1. pose_enc - Camera Parameters (2×9 array)
Contains encoded camera pose for each image:

Rotation quaternion (4 values): orientation of camera
Translation vector (3 values): camera position
Field of view (2 values): focal length parameters

For image 1: Almost identity pose (very small rotation, minimal translation) - this is the reference frame
For image 2: Significant rotation and translation (-0.215, -0.333, 0.309)
2. depth - Depth Maps (2×294×518×1)
Normalized depth values for each pixel in both images:

Image 1: depths ranging roughly 0.08 to 1.18
Image 2: depths ranging roughly 0.26 to 1.34
These are in the normalized coordinate system mentioned in the paper

3. depth_conf - Depth Confidence (2×294×518)
Uncertainty/confidence scores for depth predictions:

Values close to 1.0 = high confidence
Higher values (e.g., 1.5-1.6) = lower confidence
Based on aleatoric uncertainty mentioned in Section 3.3

4. world_points - 3D Point Maps (2×294×518×3)
The actual 3D coordinates (X, Y, Z) for each pixel, expressed in the world coordinate frame (first camera's frame):

Image 1 points: ranging from (-1.0, 0.58, 1.17) to (0.85, -0.49, 1.03)
Image 2 points: ranging from (-1.06, 0.04, 1.10) to (1.61, -0.10, 0.27)
These are viewpoint-invariant - all points are in the same global coordinate system

5. world_points_conf - Point Map Confidence (2×294×518)
Uncertainty for the 3D point predictions (similar to depth confidence)
6. images - Input Images (2×3×294×518)
The original RGB input images, normalized to [0,1] range
7. extrinsic - Camera Extrinsics (2×3×4)
The full extrinsic matrices (rotation + translation):

Image 1: Nearly identity (reference frame)
Image 2: Shows actual rotation matrix and translation vector

8. intrinsic - Camera Intrinsics (2×3×3)
Camera calibration matrices containing focal lengths and principal point:

Image 1: fx=310.09, fy=310.58, cx=259, cy=147
Image 2: fx=236.33, fy=239.51, cx=259, cy=147

9. world_points_from_depth - Alternative Point Cloud (2×294×518×3)
3D points reconstructed by combining depth maps + camera parameters (mentioned in Section 4.3 as being more accurate than direct point map prediction)

What we need is the world_points_from_depth(pointcloud) and world_points_conf(confidence)

"""

import os
import cv2
import torch
import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np
import gradio as gr
import sys
import trimesh
import shutil
from datetime import datetime
import glob
import gc
import time

sys.path.append("../..//vggt")

# from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
# from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from PC_utils import *
import numpy as np
from numba import jit, prange


def fps_approximate(points, colors, num_samples=512, grid_size=32):
    """
    Approximate FPS using spatial hashing grid
    Much faster but slightly less uniform distribution

    Speed: ~0.001-0.01s for 150k points -> 512 samples

    Algorithm:
    1. Divide space into grid cells
    2. Sample from cells rather than individual points
    3. Within selected cells, use mini-FPS
    """
    N = points.shape[0]

    if num_samples >= N:
        return points, colors, np.arange(N)

    # Voxelize space
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Compute grid indices
    grid_coords = ((points - min_coords) / (max_coords - min_coords + 1e-8) * grid_size).astype(np.int32)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)

    # Create voxel keys
    voxel_keys = grid_coords[:, 0] * grid_size * grid_size + grid_coords[:, 1] * grid_size + grid_coords[:, 2]

    # Sample points from different voxels
    unique_voxels = np.unique(voxel_keys)

    if len(unique_voxels) >= num_samples:
        # Plenty of voxels, sample one point per voxel using FPS-like selection
        selected_indices = []

        # Start with random voxel
        current_voxel = unique_voxels[np.random.randint(len(unique_voxels))]
        selected_voxels = [current_voxel]

        # Get centroid of first voxel
        voxel_mask = voxel_keys == current_voxel
        voxel_points = points[voxel_mask]
        centroid = voxel_points.mean(axis=0)

        # Pick point closest to voxel centroid
        idx_in_voxel = np.argmin(np.sum((voxel_points - centroid) ** 2, axis=1))
        selected_indices.append(np.where(voxel_mask)[0][idx_in_voxel])

        # FPS-like voxel selection
        while len(selected_indices) < num_samples:
            # Find farthest voxel from selected ones
            max_dist = -1
            farthest_voxel = None

            for voxel in unique_voxels:
                if voxel in selected_voxels:
                    continue

                voxel_mask = voxel_keys == voxel
                voxel_center = points[voxel_mask].mean(axis=0)

                # Distance to nearest selected point
                min_dist = min([np.sum((voxel_center - points[idx]) ** 2)
                                for idx in selected_indices])

                if min_dist > max_dist:
                    max_dist = min_dist
                    farthest_voxel = voxel

            selected_voxels.append(farthest_voxel)

            # Pick point from this voxel
            voxel_mask = voxel_keys == farthest_voxel
            voxel_points = points[voxel_mask]
            centroid = voxel_points.mean(axis=0)
            idx_in_voxel = np.argmin(np.sum((voxel_points - centroid) ** 2, axis=1))
            selected_indices.append(np.where(voxel_mask)[0][idx_in_voxel])

        selected_indices = np.array(selected_indices[:num_samples])
    else:
        # Fewer voxels than needed, fall back to fast FPS
        return fps_fast(points, colors, num_samples)

    return points[selected_indices], colors[selected_indices], selected_indices




if __name__ == "__main__":



    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

    vggt_model = load_vggt_model()

    example_img_dir = "./example_input_imgs"
    input_images_list = [cv2.imread(os.path.join(example_img_dir, img_filename))
                         for img_filename in os.listdir(example_img_dir)]

    predictions = run_vggt_model(vggt_model, input_images_list)

    filtered_pointCloud_vertices, filtered_pointCloud_rgb= filter_points_by_confidence(predictions, conf_thres=0.9)

    # now that we have our filtered points, next is to perform the opintcloud processing
    # filtered_pointCloud_vertices, filtered_pointCloud_rgb, _ = fps_approximate(filtered_pointCloud_vertices, filtered_pointCloud_rgb)

    glbscene = convert_to_glb(filtered_pointCloud_vertices, filtered_pointCloud_rgb)

    glbfilepath = "./example_glb/example.glb"
    glbscene.export(file_obj=glbfilepath)


    print(f"filtered_pointCloud shape: {filtered_pointCloud_vertices.shape}")
    print(f"fileterd_rgb shape: {filtered_pointCloud_rgb.shape}")





# def example_run_model(target_dir, model) -> dict:
#     """
#     Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
#     """
#     print(f"Processing images from {target_dir}")
#
#     # Device check
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     if not torch.cuda.is_available():
#         raise ValueError("CUDA is not available. Check your environment.")
#
#     # Move model to device
#     model = model.to(device)
#     model.eval()
#
#     # Load and preprocess images
#     image_names = glob.glob(os.path.join(target_dir, "images", "*"))
#     image_names = sorted(image_names)
#     print(f"Found {len(image_names)} images")
#     if len(image_names) == 0:
#         raise ValueError("No images found. Check your upload.")
#
#     images = load_and_preprocess_images(image_names).to(device)
#     print(f"Preprocessed images shape: {images.shape}")
#
#     # Run inference
#     print("Running inference...")
#     dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
#
#     with torch.no_grad():
#         with torch.cuda.amp.autocast(dtype=dtype):
#             predictions = model(images)
#
#     # Convert pose encoding to extrinsic and intrinsic matrices
#     print("Converting pose encoding to extrinsic and intrinsic matrices...")
#     extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
#     predictions["extrinsic"] = extrinsic
#     predictions["intrinsic"] = intrinsic
#
#     # Convert tensors to numpy
#     for key in predictions.keys():
#         if isinstance(predictions[key], torch.Tensor):
#             predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
#     predictions['pose_enc_list'] = None # remove pose_enc_list
#
#     # Generate world points from depth map
#     print("Computing world points from depth map...")
#     depth_map = predictions["depth"]  # (S, H, W, 1)
#     world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
#     predictions["world_points_from_depth"] = world_points
#
#     # Clean up
#     torch.cuda.empty_cache()
#     return predictions


