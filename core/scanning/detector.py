# detector.py
# -----------
# This module handles card detection within an image frame for the Magic Lens application.
# It identifies the region of interest (ROI) where a card appears and prepares it for OCR.
#
# Main Features:
# - Card region detection using image processing techniques
# - Contour detection to identify card boundaries
# - ROI extraction for OCR processing
#
# Usage:
# Import and use the CardDetector class to detect cards in images or video frames.
#
# Dependencies:
# - OpenCV (cv2)
# - NumPy
#
# Author: Carlos Hernán Guirao
# Date: 2025-02-20

from .image_utils import ImageUtils
import cv2


class CardDetector:
    @staticmethod
    def detect_card(frame):
        """Detect cards in the preprocessed image."""
        preprocessed = ImageUtils.preprocess_for_ocr(frame)
        contours, _ = cv2.findContours(
            preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:  # Likely a card
                x, y, w, h = cv2.boundingRect(approx)
                return ImageUtils.crop_region(frame, x, y, w, h)

        return None
