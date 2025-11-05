"""
vggt prediction is a dictionary with following keys

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
sys.path.append("../..//vggt")

# from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
# from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def load_and_preprocess_images(image_input_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
         image_input_list (list): List of image inputs, can be:
                                - Paths to image files (str)
                                - PIL Images
                                - Numpy arrays (H, W, C) with values in [0, 255] or [0, 1]
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_input_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_input in image_input_list:
        # Convert input to PIL Image
        if isinstance(image_input, str):
            # It's a file path
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            # It's a numpy array
            # Handle different array shapes and value ranges
            if image_input.ndim == 2:
                # Grayscale image (H, W) -> convert to RGB
                image_input = np.stack([image_input] * 3, axis=-1)
            elif image_input.ndim == 3:
                if image_input.shape[2] not in [3, 4]:
                    raise ValueError(f"Numpy array must have 3 or 4 channels, got {image_input.shape[2]}")
            else:
                raise ValueError(f"Numpy array must be 2D or 3D, got {image_input.ndim}D")

            # Convert to uint8 if needed (handle both [0, 1] and [0, 255] ranges)
            if image_input.dtype == np.float32 or image_input.dtype == np.float64:
                if image_input.max() <= 1.0:
                    # Assume [0, 1] range
                    image_input = (image_input * 255).astype(np.uint8)
                else:
                    # Assume already in [0, 255] range
                    image_input = image_input.astype(np.uint8)
            elif image_input.dtype != np.uint8:
                image_input = image_input.astype(np.uint8)

            img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            # It's already a PIL Image
            img = image_input
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}. Expected str, numpy.ndarray, or PIL.Image")

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_input_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

def load_vggt_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)

    return model

def run_vggt_model(model, input_image_list):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move model to device
    model = model.to(device)
    model.eval()

    # format input images into model input
    model_input = load_and_preprocess_images(input_image_list).to(device)

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(model_input)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], model_input.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None  # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions

def filter_points_by_confidence(predictions, conf_thres=0.7):
    """

    Args:
        predictions:
        threshold:

    Returns:

    """

    # assumeing using camera and depth to generate pointcloud
    pred_world_points = predictions["world_points_from_depth"]
    pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions

    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_matrices = predictions["extrinsic"]
    images = predictions["images"]

    # convert bgr to rgb
    images = images[:, [2, 1, 0], :, :]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)


    conf = pred_world_points_conf.reshape(-1)

    # Convert inverse confidence to normal confidence
    # confidence_normal = 1.0 / np.maximum(conf, 1e-6)  # Avoid division by zero
    confidence_normal = conf

    # Now apply threshold (higher conf_thres = stricter filtering)
    if conf_thres == 0.0:
        # Keep all points
        conf_mask = np.ones(len(conf), dtype=bool)
    else:
        # Keep points with confidence >= threshold
        conf_mask = (confidence_normal >= conf_thres) & (conf > 1e-5)

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    return vertices_3d, colors_rgb

def convert_to_glb(vertices_3d, colors_rgb):
    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    return scene_3d




if __name__ == "__main__":


    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

    vggt_model = load_vggt_model()

    # in actual pipeline this will just be an array of numpy images from the sensor
    example_img_dir = "./example_input_imgs"
    input_images_list = [cv2.imread(os.path.join(example_img_dir, img_filename))
                         for img_filename in os.listdir(example_img_dir)]

    # get predictions
    predictions = run_vggt_model(vggt_model, input_images_list)


    print(f"predictions: {predictions}")



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


