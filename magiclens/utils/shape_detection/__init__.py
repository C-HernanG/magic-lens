"""
Shape detection utilities for Magic Lens.
Provides functions for detecting shapes within images, including Fourier-based methods.
"""

from .fourier_descriptor import fourier_shape_detection

__all__ = ["fourier_shape_detection"]
