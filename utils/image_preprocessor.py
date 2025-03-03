"""
image_preprocessor.py
---------------------
Provides functionality for image preprocessing pipelines using OpenCV.

The main class, `ImagePreprocessor`, is the intended public API for image preprocessing. It:
  1. Offers multiple pipelines for different preprocessing needs.
  2. Uses internal helper methods (prefixed with an underscore) to modularize steps such as resizing, grayscale conversion, normalization, thresholding, and histogram equalization.
  3. Allows easy extension by adding new pipelines or modifying existing ones to suit various image processing tasks.

Usage Example:
    from utils import ImagePreprocessor
    import cv2

    # Initialize preprocessor and load an image
    preprocessor = ImagePreprocessor()
    image = cv2.imread('sample.jpg')
    if image is not None:
        # Process the image using pipeline one
        processed_image = preprocessor.pipeline_one(image)
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image not found or unable to load.")

Dependencies:
    - OpenCV (cv2)
    - NumPy
    - PIL (Python Imaging Library)
"""

import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter


class ImagePreprocessor:

    # Preprocessing pipelines

    @staticmethod
    def preprocessing1(image):
        # 1. Convert to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # 2. Adjust gamma to enhance contrast
        gray = ImagePreprocessor._adjust_gamma(gray, 1.5)

        # 3. Blur the gamma-corrected image
        blurred = cv.GaussianBlur(gray, (5, 5), 0)

        # 4. Edge detection (Canny) - adjust thresholds as needed
        edges = cv.Canny(blurred, threshold1=50, threshold2=150)

        # 5. (Optional) Morphological steps to close small gaps
        kernel = np.ones((2, 2), np.uint8)
        edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=1)

        return edges

    @staticmethod
    def preprocessing2(image):
        image = ImagePreprocessor._crop_name_region(image)
        # Convert to grayscale and upscale/sharpen for better OCR results
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = ImagePreprocessor._upscale_and_sharpen(gray)

        # Apply a sharpening filter using a convolution kernel
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        sharpened_cv = cv.filter2D(src=gray, ddepth=-1, kernel=kernel_sharpen)

        # Threshold the image to isolate text (using Otsu's method)
        _, thresh = cv.threshold(sharpened_cv, 0, 255,
                                 cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Optionally perform a morphological opening to reduce noise
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
        processed = cv.morphologyEx(
            thresh, cv.MORPH_OPEN, kernel, iterations=1)

        return processed

    # Private helper methods

    @staticmethod
    def _adjust_gamma(image, gamma=1.0):
        """
        Apply gamma correction to an image.

        Parameters:
            image (ndarray): Input image.
            gamma (float): Gamma correction factor.

        Returns:
            ndarray: Gamma-corrected image.
        """
        invGamma = 1.0 / gamma
        # Build a lookup table for gamma correction.
        table = np.array([((i / 255.0) ** invGamma) *
                          255 for i in np.arange(256)]).astype("uint8")
        return cv.LUT(image, table)

    @staticmethod
    def _upscale_and_sharpen(gray_img, scale=2):
        """
        Upscale a grayscale image using bicubic interpolation and then apply an
        unsharp mask using Pillow for clarity.
        """
        height, width = gray_img.shape
        resized = cv.resize(gray_img, (width * scale, height * scale),
                            interpolation=cv.INTER_CUBIC)
        pil_img = Image.fromarray(resized)
        sharpened_pil = pil_img.filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        return np.array(sharpened_pil)

    @staticmethod
    def _crop_name_region(card_image):
        """
        Crop out the top portion of the card where the name is expected.
        Adjust the coordinates if your card layout differs.
        """
        h, w = card_image.shape[:2]
        y_start = int(h * 0.0)
        y_end = int(h * 0.15)
        x_start = int(w * 0.0)
        x_end = int(w * 1)
        return card_image[y_start:y_end, x_start:x_end]
