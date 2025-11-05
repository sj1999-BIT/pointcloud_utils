import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pointnet2_ops.pointnet2_modules
import pointnet2_ops.pointnet2_utils
from pointnet2_ops._version import __version__
