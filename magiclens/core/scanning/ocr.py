# ocr.py
# ------
# This module handles Optical Character Recognition (OCR) for the Magic Lens application.
# It extracts card names and other relevant text from detected card images using Tesseract OCR.
#
# Main Features:
# - OCR processing of image regions containing card names.
# - Preprocessing to enhance text detection accuracy.
# - Error handling and optional spell-checking for OCR results.
#
# Usage:
# Import and use the extract_text function to perform OCR on card images.
#
# Dependencies:
# - Tesseract OCR
# - OpenCV (cv2)
# - NumPy
#
# Author: Carlos Hernán Guirao
# Date: 2025-02-20

import pytesseract
from .image_utils import ImageUtils


class OCRProcessor:
    @staticmethod
    def extract_text(image):
        """Extract text from preprocessed image."""
        preprocessed = ImageUtils.preprocess_for_ocr(image)
        text = pytesseract.image_to_string(preprocessed)
        return text.strip()
