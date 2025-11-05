发它"""
Both max-pooling and self-attention can be affected by noisy pointcloud

set-mixer: all points together -> sort to reveal structure -> mix across all -> aggregate
dilute bad point effect

sort the point features across axis
"""

import torch.nn.functional as F
import torch.nn as nn

from abc import ABC, abstractmethod

class PointCloudEncoder(ABC):
    def __init__(self, , model_name: str):
        self.model_name = model_name
        self.model = None


    @abstractmethod
    def load_model(self, weight_path = None):
        pass

    @abstractmethod
    def run(self, input_data):
        pass

    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_type": self.__class__.__name__
        }

class simpleMLPEncoder(PointCloudEncoder):
    def __init__(self):
        super().__init__("pointNetEncoder")

    def load_model(self, weight_path = None):
        if weight_path is None:
            pass








