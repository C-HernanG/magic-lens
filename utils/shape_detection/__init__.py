"""
utils/shape_detection/__init__.py
---------------------------------
Re-exports the shape detection utilities for the project.

Usage:
    from utils.shape_detection import *
"""

from .fourier_descriptor import fourier_shape_detection

__all__ = ["fourier_shape_detection"]
