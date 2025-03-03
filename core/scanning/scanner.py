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
from .image_utils import ImageUtils


class CardScanner:
    """Handles real-time webcam integration for card scanning in Magic Lens."""

    def __init__(self, camera_index=0):
        """
        Initialize the webcam scanner.

        Parameters:
        - camera_index (int): Index of the webcam (0 for default camera).
        """
        self.camera_index = camera_index
        self.cap = cv.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open webcam at index {camera_index}")

    def start(self):
        """Start the webcam and display the video feed with real-time processing."""
        print("[INFO] Starting webcam... Press 'q' to exit.")

        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            # Preprocess the frame for card detection
            processed_frame = ImageUtils.preprocess_for_ocr(frame)

            # Display the original and processed frames
            cv.imshow("Magic Lens - Webcam Feed", frame)
            cv.imshow("Magic Lens - Processed Frame", processed_frame)

            # Exit on 'q' key press
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exiting webcam...")
                break

        self.stop()

    def stop(self):
        """Release the webcam and close all OpenCV windows."""
        print("[INFO] Releasing webcam resources...")
        self.cap.release()
        cv.destroyAllWindows()
