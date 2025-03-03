"""
fourier_descriptor.py
---------------------
Provides functionality for detecting shapes via Fourier descriptors.

The main function, `fourier_shape_detection`, is the only intended public API. It:
  1. Loads and processes a model image to compute its Fourier descriptor.
  2. Loads and processes a target image, extracts contours, and compares their 
     descriptors to the model’s descriptor.
  3. Returns the processed target image, the matched contours (with distances), 
     and their original-scale counterparts.
  4. Optionally visualizes the results and/or saves them to a file.

Usage Example:
    from utils import fourier_shape_detection

    def preprocess_func(image):
        # Example binarization: convert to grayscale and apply Otsu’s threshold
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, bin_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return bin_img

    target_img, found, found_original = fourier_shape_detection(
        "model.jpg", 
        "target.jpg", 
        preprocess=preprocess_func, 
        visualize=True
    )

Dependencies:
  - OpenCV (cv2)
  - NumPy
  - Matplotlib (optional for visualization)
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional


def _compute_fourier_descriptor(contour: np.ndarray, wmax: int = 10) -> np.ndarray:
    """
    Compute a Fourier descriptor for the given contour.
    The descriptor is normalized to be invariant to scale and rotation.

    Parameters:
        contour (np.ndarray): Array of contour points (N, 2).
        wmax (int): Maximum number of Fourier coefficients to include.

    Returns:
        np.ndarray: Normalized Fourier descriptor.
    """
    # Represent contour points as complex numbers.
    x, y = contour.T
    z = x + 1j * y

    # Compute the Fourier transform.
    f = np.fft.fft(z)
    fa = np.abs(f)

    # Fallback if degenerate.
    if len(fa) < 2:
        return fa

    # Use the sum of the first and last Fourier components as a normalizing factor.
    norm_factor = fa[1] + fa[-1]
    if norm_factor == 0:
        return fa

    # Build the descriptor vector with 2*wmax + 1 elements.
    descriptor = np.zeros(2 * wmax + 1)

    # Fill the front coefficients (skip the first two coefficients).
    front_len = min(wmax, len(fa) - 2)
    descriptor[:front_len] = fa[2:2 + front_len]

    # Fill the back coefficients.
    back_len = min(wmax, len(fa) - 1)
    descriptor[wmax:wmax + back_len] = fa[-back_len - 1:-1]

    # Save the first Fourier component for reference.
    descriptor[-1] = fa[1]

    # Optionally flip the descriptor if needed to handle orientation.
    if fa[-1] > fa[1]:
        descriptor[:-1] = descriptor[-2::-1]

    return descriptor / norm_factor


def _resize_image(image: np.ndarray, max_width: int = 600, max_height: int = 800) -> Tuple[np.ndarray, float]:
    """
    Resize the image while maintaining the aspect ratio.

    Parameters:
        image (np.ndarray): Input image.
        max_width (int): Maximum width for the resized image.
        max_height (int): Maximum height for the resized image.

    Returns:
        Tuple[np.ndarray, float]: (Resized image, scaling factor)
    """
    height, width = image.shape[:2]
    ratio_w = max_width / float(width)
    ratio_h = max_height / float(height)
    ratio = min(ratio_w, ratio_h)

    new_width = int(width * ratio)
    new_height = int(height * ratio)

    resized = cv.resize(image, (new_width, new_height),
                        interpolation=cv.INTER_AREA)
    return resized, ratio


def _extract_contours(image: np.ndarray, preprocess: Callable[[np.ndarray], np.ndarray]) -> List[np.ndarray]:
    """
    Binarize the input image using the provided preprocessing function,
    and extract contours sorted by area (largest first).

    Parameters:
        image (np.ndarray): Input image.
        preprocess (Callable): Function to binarize the image.

    Returns:
        List[np.ndarray]: List of contours (each contour is an ndarray of shape (N,2)).
    """
    bin_img = preprocess(image)

    # Find contours in the binarized image.
    contours, _ = cv.findContours(bin_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    # Reshape contours from (N,1,2) to (N,2).
    contours = [c.reshape(-1, 2) for c in contours]

    # Sort contours by area in descending order.
    return sorted(contours, key=cv.contourArea, reverse=True)


def _is_reasonable_contour(contour: np.ndarray) -> bool:
    """
    Check if the contour's area is within a reasonable range.

    Parameters:
        contour (np.ndarray): Contour points.

    Returns:
        bool: True if the area is between 1000 and 400000.
    """
    area = cv.contourArea(contour.reshape(-1, 1, 2))
    return 1000 <= area <= 400000


def _is_counter_clockwise(contour: np.ndarray) -> bool:
    """
    Determine if a contour is oriented counter-clockwise.

    Parameters:
        contour (np.ndarray): Contour points.

    Returns:
        bool: True if the contour is counter-clockwise.
    """
    return cv.contourArea(contour.astype(np.float32), oriented=True) > 0


def _draw_text(image: np.ndarray, text: str, position: Tuple[int, int]) -> None:
    """
    Draw text on an image.

    Parameters:
        image (np.ndarray): Image to draw on.
        text (str): Text string.
        position (Tuple[int, int]): (x, y) coordinates for the text.
    """
    cv.putText(
        image,
        text,
        position,
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,              # Font scale.
        (255, 255, 255),  # White color.
        2,                # Thickness.
        cv.LINE_AA
    )


def _visualize_results(target_img: np.ndarray,
                       found: List[Tuple[np.ndarray, float]],
                       preprocess: Callable[[np.ndarray], np.ndarray],
                       save_path: Optional[str] = None) -> np.ndarray:
    """
    Visualize the results of the Fourier shape detection.

    Creates a horizontal stack showing:
      1. The original target image.
      2. The binarized image.
      3. The target image annotated with matched contours.

    Parameters:
        target_img (np.ndarray): The target image.
        found (List[Tuple[np.ndarray, float]]): List of tuples (contour, distance) for each matched contour.
        preprocess (Callable): Function to binarize the image.
        display (bool): If True, display the visualization using matplotlib.
        save_path (Optional[str]): Optional path to save the visualization image.

    Returns:
        np.ndarray: The stacked visualization image.
    """
    # Copy the original image.
    original = target_img.copy()

    # Obtain the binarized image.
    bin_img = preprocess(target_img)
    bin_img_bgr = cv.cvtColor(bin_img, cv.COLOR_GRAY2BGR)

    # Prepare a copy for drawing contours.
    contours_img = target_img.copy()
    for contour, dist in found:
        contour_reshaped = contour.reshape(-1, 1, 2)
        cv.drawContours(contours_img, [contour_reshaped], -1, (0, 255, 0), 2)

        # Compute the contour's centroid.
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = contour.mean(axis=0).astype(int)

        _draw_text(contours_img, f"{dist:.2f}", (cx, cy))

    # Stack the images horizontally.
    stacked = np.hstack([original, bin_img_bgr, contours_img])
    print("Contours matched:", len(found))

    # Save the visualization if a path is provided.
    if save_path:
        cv.imwrite(save_path, stacked)

    # Display the visualization.
    plt.figure(figsize=(16, 8))
    plt.imshow(cv.cvtColor(stacked, cv.COLOR_BGR2RGB))
    plt.title("Original | Binarized | Contours")
    plt.axis("off")
    plt.show()


def fourier_shape_detection(model_image_path: str,
                            input_image_path: str,
                            preprocess: Callable[[np.ndarray], np.ndarray],
                            visualize: bool = True,
                            save_path: Optional[str] = None
                            ) -> Tuple[np.ndarray,
                                       List[Tuple[np.ndarray, float]],
                                       List[Tuple[np.ndarray, float]]]:
    """
    Detect contours in the target image that match the shape of the model image
    using Fourier descriptors.

    Parameters:
        model_image_path (str): File path for the model image.
        input_image_path (str): File path for the target image.
        preprocess (Callable): Function to preprocess (binarize) images.
        visualize (bool): If True, display visualization of the detection results.
        save_path (Optional[str]): Optional file path to save the visualization image.

    Returns:
        Tuple:
          - Resized target image.
          - List of matched contours with distances (resized).
          - List of matched contours scaled to original dimensions.
    """
    # Process the model image.
    model_img = cv.imread(model_image_path)
    if model_img is None:
        raise ValueError(
            f"Could not load model image from '{model_image_path}'.")
    model_img, _ = _resize_image(model_img)

    # Extract contours from the model image.
    model_contours = _extract_contours(model_img, preprocess)
    if not model_contours:
        raise ValueError("No contours found in model image!")

    # Assume the largest contour is the model.
    model_contour = model_contours[0]
    model_descriptor = _compute_fourier_descriptor(model_contour)

    # Process the target image.
    target_img = cv.imread(input_image_path)
    if target_img is None:
        raise ValueError(
            f"Could not load target image from '{input_image_path}'.")
    target_img, ratio = _resize_image(target_img)

    # Extract contours from the target image.
    contours = _extract_contours(target_img, preprocess)

    # Filter contours by size and orientation.
    valid_contours = [
        c for c in contours if _is_reasonable_contour(c) and not _is_counter_clockwise(c)
    ]

    # Compare each valid contour with the model descriptor.
    MAX_DIST = 0.25  # Distance threshold for a match.
    found = []
    for contour in valid_contours:
        descriptor = _compute_fourier_descriptor(contour)
        dist = np.linalg.norm(descriptor - model_descriptor)
        if dist < MAX_DIST:
            found.append((contour, dist))

    # Convert contours to the original image scale.
    found_original = [((contour / ratio).astype(np.int32), dist)
                      for contour, dist in found]

    # Visualize the results if requested.
    if visualize:
        _visualize_results(target_img, found, preprocess,
                           save_path=save_path)

    return target_img, found, found_original
