import torch
import torch.nn as nn
import numpy as np

from pointnet2.configs import get_cfg_defaults
import pointnet2.models
import pointnet2.PCT_Pytorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils as pointnet2_utils


class PointNetPredictor:
    """
    Simplified PointNet inference wrapper.
    
    Point Cloud Format Requirements:
    - Input shape: (N, 3) where N is the number of points (e.g., 1024 or 2048)
    - Each row represents [x, y, z] coordinates
    - Data type: numpy array or torch tensor
    - Values should be normalized (typically centered around origin)
    
    Example point cloud shape: (1024, 3)
    """
    
    def __init__(self,
                 model_path='./pointNet_pretrained_weights/PointCutMix_R.pth',
                 config_path='./configs/pct_r.yaml',
                 device=None):
        """
        Initialize the PointNet predictor.
        
        Args:
            model_path (str): Path to the trained model checkpoint (.pth file)
            config_path (str): Path to the experiment config file
            device (str, optional): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(config_path)

        # print(self.cfg)
        self.cfg.freeze()

        # assume inputs is 1024
        self.num_points = 1024
        
        # Initialize model
        self.model = pointnet2.models.Pct(
            task=self.cfg.EXP.TASK,
            dataset=self.cfg.EXP.DATASET
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle DataParallel models
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            # If model was saved with DataParallel
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(checkpoint['model_state'])
            self.model = self.model.module
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Dataset: {self.cfg.EXP.DATASET}")
        print(f"Task: {self.cfg.EXP.TASK}")

    def preprocess_point_cloud(self, point_cloud, use_fps=True):
        """
        Preprocess point cloud for inference using PointNet++ CUDA FPS or fallback.

        Args:
            point_cloud (numpy.ndarray or torch.Tensor): Point cloud data
                Shape: (N, 3) where N is ANY number of points
                Format: Each row is [x, y, z] coordinates
            use_fps (bool): If True, use PointNet++ CUDA FPS. If unavailable, fallback to numpy FPS

        Returns:
            torch.Tensor: Preprocessed point cloud with shape (1, 3, num_points)
        """
        # Convert to tensor if needed
        if isinstance(point_cloud, np.ndarray):
            points = torch.FloatTensor(point_cloud).unsqueeze(0)  # (1, N, 3)
        else:
            points = point_cloud
            if len(points.shape) == 2:
                points = points.unsqueeze(0)  # (1, N, 3)

        # Ensure correct shape
        if points.shape[2] != 3:
            raise ValueError(f"Expected point cloud shape (1, N, 3) or (N, 3), got {points.shape}")

        current_num_points = points.shape[1]

        print(f"shuijie debug Current num points: {current_num_points}")

        # Resample to expected number of points
        if current_num_points != self.num_points:
            if use_fps and current_num_points > self.num_points:
                # Try to use PointNet++ CUDA FPS
                try:
                    # Move to device for CUDA operations
                    points = points.to(self.device)

                    # FPS expects (B, N, 3) and returns indices (B, num_points)
                    fps_idx = pointnet2_utils.furthest_point_sample(points, self.num_points)

                    # Gather sampled points: (B, 3, N) -> (B, 3, num_points) -> (B, num_points, 3)
                    sampled_points = pointnet2_utils.gather_operation(
                        points.transpose(1, 2).contiguous(),
                        fps_idx
                    ).transpose(1, 2).contiguous()  # (1, num_points, 3)

                except ImportError:
                    print("Warning: PointNet++ CUDA ops not available, using numpy FPS fallback")
                    # Fallback to numpy FPS
                    points_np = points.cpu().numpy()[0]  # (N, 3)
                    sampled_np = self.fps_resample(points_np, self.num_points)
                    sampled_points = torch.FloatTensor(sampled_np).unsqueeze(0).to(self.device)
            else:
                # Random resampling for upsampling or if FPS not requested
                points_np = points.cpu().numpy()[0]  # (N, 3)
                sampled_np = self.random_resample(points_np, self.num_points)
                sampled_points = torch.FloatTensor(sampled_np).unsqueeze(0).to(self.device)
        else:
            sampled_points = points.to(self.device)

        print(f"shuijie debug Current sampled_points: {sampled_points.shape}")

        # Transpose to (1, 3, num_points) format expected by models
        pc_tensor = sampled_points.contiguous()  # (1, 3, num_points)

        return pc_tensor
    
    def predict(self, point_cloud, return_logits=False):
        """
        Generate prediction for a single point cloud.
        
        Args:
            point_cloud (numpy.ndarray or torch.Tensor): Input point cloud
                Shape: (N, 3) where N is number of points
                Format: [[x1, y1, z1], [x2, y2, z2], ..., [xN, yN, zN]]
                
            return_logits (bool): If True, return raw logits instead of class prediction
        
        Returns:
            If return_logits=False:
                int: Predicted class index
            If return_logits=True:
                numpy.ndarray: Raw logits for all classes, shape (num_classes,)
        
        Example:
            >>> predictor = PointNetPredictor('model.pth', 'config.yaml')
            >>> pc = np.random.randn(1024, 3)  # Random point cloud
            >>> prediction = predictor.predict(pc)
            >>> print(f"Predicted class: {prediction}")
        """

        # Preprocess input
        pc_tensor = self.preprocess_point_cloud(point_cloud)
        
        # Run inference
        with torch.no_grad():
            output = self.model(pc=pc_tensor)
            logits = output['logit']  # Shape: (1, num_classes)
        
        # Convert to numpy and remove batch dimension
        logits_np = logits.cpu().numpy()[0]  # Shape: (num_classes,)
        
        if return_logits:
            return logits_np
        else:
            return int(np.argmax(logits_np))
    
    def predict_batch(self, point_clouds, return_logits=False):
        """
        Generate predictions for multiple point clouds.
        
        Args:
            point_clouds (numpy.ndarray or torch.Tensor): Batch of point clouds
                Shape: (B, N, 3) where B is batch size, N is number of points
                
            return_logits (bool): If True, return raw logits instead of class predictions
        
        Returns:
            If return_logits=False:
                numpy.ndarray: Predicted class indices, shape (B,)
            If return_logits=True:
                numpy.ndarray: Raw logits for all classes, shape (B, num_classes)
        """
        # Process each point cloud individually to ensure proper resampling
        batch_tensors = []
        for pc in point_clouds:
            # Preprocess each point cloud (handles resampling and formatting)
            pc_tensor = self.preprocess_point_cloud(pc)  # Returns (1, 3, num_points)
            batch_tensors.append(pc_tensor)

        # Concatenate into batch
        pc_tensor = torch.cat(batch_tensors, dim=0)  # (B, 3, num_points)

        
        # Run inference
        with torch.no_grad():
            output = self.model(pc=pc_tensor)
            logits = output['logit']  # Shape: (B, num_classes)
        
        logits_np = logits.cpu().numpy()
        
        if return_logits:
            return logits_np
        else:
            return np.argmax(logits_np, axis=1)


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = PointNetPredictor(
        model_path='./pointNet_pretrained_weights/PointCutMix_R.pth',
        config_path='./configs/pct_r.yaml'
    )
    
    # Example 1: Single point cloud prediction
    # Create a random point cloud with 1024 points
    point_cloud = np.random.randn(2048, 3)
    
    # Get prediction
    predicted_class = predictor.predict(point_cloud)
    print(f"Predicted class: {predicted_class}")
    
    # Get logits (scores for all classes)
    logits = predictor.predict(point_cloud, return_logits=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Class scores: {logits}")
    
    # Example 2: Batch prediction
    # Create batch of 4 point clouds
    batch_point_clouds = np.random.randn(4, 1024, 3)
    
    # Get predictions for all point clouds
    predicted_classes = predictor.predict_batch(batch_point_clouds)
    print(f"Batch predictions: {predicted_classes}")
