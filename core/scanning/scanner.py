# scanner.py
# ----------
# This module handles real-time webcam integration for card scanning in the Magic Lens application.
# It captures frames, preprocesses the images, and sends them to the card detection pipeline for OCR.
#
# Main Features:
# - Webcam frame capture
# - Image preprocessing for better detection
# - Card detection and OCR extraction
# - Display of results and optional saving
#
# Dependencies:
# - OpenCV (cv2)
# - Tesseract OCR
# - NumPy
#
# Author: Carlos Hernán Guirao
# Date: 2025-02-20

import cv2 as cv
import numpy as np
