"""
Display utility functions for the Virtual Try-On Game
"""

import cv2
import numpy as np


def create_vertical_display(frame):
    """
    Convert a regular camera frame to a vertical display with 9:16 aspect ratio
    This is useful for creating a portrait mode display similar to mobile devices

    Args:
        frame (numpy.ndarray): Input image frame

    Returns:
        numpy.ndarray: Vertically oriented frame with black padding
    """
    h, w = frame.shape[:2]

    # Target aspect ratio is 9:16 (portrait mode)
    target_aspect = 9 / 16  # width / height

    # Calculate dimensions for the vertical display
    if w / h > target_aspect:
        # Width is too large for target aspect ratio
        new_width = int(h * target_aspect)
        # Crop from left side instead of center
        start_x = 0
        cropped_frame = frame[:, start_x:start_x + new_width]
    else:
        # Height is too large for target aspect ratio
        new_height = int(w / target_aspect)
        # Crop from center for height (typically this is less common)
        start_y = (h - new_height) // 2
        cropped_frame = frame[start_y:start_y + new_height, :]

    # This function doesn't resize the frame, just crops and pads it to maintain the aspect ratio
    # If you need to resize to specific dimensions, uncomment and modify the following:
    # vertical_frame = cv2.resize(cropped_frame, (target_width, target_height))

    return cropped_frame
