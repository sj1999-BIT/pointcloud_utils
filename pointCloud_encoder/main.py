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
# sys.path.append("../..//vggt")


# from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
# from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


from pc_utils import *
from pc_predictor import *
import numpy as np


if __name__ == "__main__":



    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

    # initiate the model
    pc_encoder = PointNetPredictor(
        model_path='./pointnet2/pointNet_pretrained_weights/PointCutMix_R.pth',
        config_path='./pointnet2/configs/pct_r.yaml'
    )

    vggt_model = load_vggt_model()




    example_img_dir = "./example_input_imgs"
    input_images_list = [cv2.imread(os.path.join(example_img_dir, img_filename))
                         for img_filename in os.listdir(example_img_dir)]

    predictions = run_vggt_model(vggt_model, input_images_list)

    filtered_pointCloud_vertices, filtered_pointCloud_rgb = filter_points_by_confidence(predictions, conf_thres=0.5)

    # now that we have our filtered points, next is to perform the opintcloud processing
    # filtered_pointCloud_vertices, filtered_pointCloud_rgb, _ = fps_approximate(filtered_pointCloud_vertices, filtered_pointCloud_rgb)

    # glbscene = convert_to_glb(filtered_pointCloud_vertices, filtered_pointCloud_rgb)
    #
    # glbfilepath = "./example_glb/example.glb"
    # glbscene.export(file_obj=glbfilepath)


    print(f"filtered_pointCloud shape: {filtered_pointCloud_vertices.shape}")
    print(f"fileterd_rgb shape: {filtered_pointCloud_rgb.shape}")

    # Get prediction
    predicted_class = pc_encoder.predict(filtered_pointCloud_vertices)
    print(f"Predicted class: {predicted_class}")

    # Get logits (scores for all classes)
    logits = pc_encoder.predict(filtered_pointCloud_vertices, return_logits=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Class scores: {logits}")





