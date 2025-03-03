"""
Utility modules for Magic Lens.
This includes image preprocessing functions and shape detection helpers.
"""

from .shape_detection import fourier_shape_detection
from .image_preprocessor import ImagePreprocessor

__all__ = ["fourier_shape_detection", "ImagePreprocessor"]
