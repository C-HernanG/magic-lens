# __init__.py
# -----------
# This package handles the card scanning functionality for the Magic Lens application.
# It includes webcam integration, card detection, image preprocessing, and OCR processing.
#
# Modules:
# - scanner: Manages real-time webcam scanning.
# - detector: Detects cards in image frames.
# - image_utils: Provides image preprocessing functions.
# - ocr: Handles text extraction from detected cards.
#
# Usage:
# from core.scanning import CardScanner, CardDetector, ImageUtils, OCRProcessor
#
# Author: Carlos Hernán Guirao
# Date: 2025-02-20

from .scanner import CardScanner
from .detector import CardDetector
from .image_utils import ImageUtils
from .ocr import OCRProcessor

# Export
__all__ = ["CardScanner", "CardDetector", "ImageUtils", "OCRProcessor"]
