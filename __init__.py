"""
This package contains multiple architectures for segmentation. Each module
contains one architecture and can contain multiple different models with minor changes.
"""
# pylint: disable=invalid-name
from .deeplab import DeepLabv3plus
from .densenets import DenseTiramisu
