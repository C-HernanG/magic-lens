# image_utils.py
# --------------
# This module provides utility functions for image preprocessing in the Magic Lens application.
# It enhances images to improve card detection and OCR accuracy.
#
# Main Features:
# - Grayscale conversion
# - Image thresholding and edge detection
# - Noise reduction and resizing
#
# Usage:
# Import and use the utility functions for preprocessing image frames before detection.
#
# Dependencies:
# - OpenCV (cv2)
# - NumPy
#
# Author: Carlos Hernán Guirao
# Date: 2025-02-20

import cv2
import numpy as np


class ImageUtils:
    """Utility class for image preprocessing."""

    @staticmethod
    def resize(image, target_width=1024):
        """
        Resize the image to a given width while maintaining aspect ratio.
        """
        (h, w) = image.shape[:2]
        ratio = target_width / float(w)
        dim = (target_width, int(h * ratio))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    @staticmethod
    def to_grayscale(image):
        """Convert image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def enhance_contrast(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to boost local contrast.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    @staticmethod
    def denoise_image(image, d=9, sigma_color=75, sigma_space=75):
        """
        Bilateral filtering reduces noise while preserving edges.
        Parameters can be tuned for more or less aggressive filtering.
        """
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    @staticmethod
    def deskew_image(image):
        """
        Deskew the image by detecting the angle of the text region
        and rotating it so lines are horizontal.
        """
        # Threshold (Otsu) for shape detection
        thresh = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.size == 0:
            # If there's no text region, return as is
            return image

        # Determine angle of the bounding box that encloses the text
        angle = cv2.minAreaRect(coords)[-1]
        # Adjust angle to the proper range
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate the image to correct the skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed

    @staticmethod
    def apply_threshold(
        image,
        max_value=255,
        method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        threshold_type=cv2.THRESH_BINARY,
        block_size=11,
        C=2
    ):
        """
        Apply adaptive thresholding to enhance contrast.
        block_size and C can be tweaked to handle different
        lighting or text thickness conditions.
        """
        return cv2.adaptiveThreshold(image, max_value, method, threshold_type, block_size, C)

    @staticmethod
    def morphological_ops(image, kernel_size=(2, 2), iterations=1):
        """
        Remove small noise and bridge gaps in the text using opening and closing.
        Adjust kernel_size and iterations for more or less aggressive cleanup.
        """
        kernel = np.ones(kernel_size, np.uint8)
        # Opening: remove small white noise
        opened = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        # Closing: close small holes within the foreground
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return closed

    @staticmethod
    def crop_region(image, x, y, w, h):
        return image[y:y+h, x:x+w]

    @staticmethod
    def preprocess_for_ocr(
        image,
        target_width=800,
        clip_limit=3.0,
        tile_grid_size=(8, 8),
        bilateral_d=9,
        sigma_color=75,
        sigma_space=75,
        threshold_block_size=11,
        threshold_C=2,
        morph_kernel_size=(2, 2),
        morph_iterations=1,
        invert=False
    ):
        """
        Full pipeline for OCR:
          1. Resize
          2. Grayscale
          3. Contrast enhancement
          4. Denoise
          5. Deskew
          6. Adaptive threshold
          7. Morphological operations
        """
        # Resize image with a high-quality interpolation method
        # Ensure this uses, e.g., cv2.INTER_CUBIC

        # Apply bilateral filter to reduce noise while preserving edges
        image = cv2.bilateralFilter(image, 9, 75, 75)

        # Convert to grayscale
        image = ImageUtils.to_grayscale(image)

        # Optionally enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # Crop image - consider dynamic cropping if possible
        image = image[50:130, 40:750]

        # Use adaptive thresholding for better text segmentation
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

        # Optionally, apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        return image
