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
