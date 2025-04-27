"""
Clothing Overlay Module for Virtual Try-On Game
Handles virtual clothing rendering and positioning
"""

import cv2
import numpy as np
import os


class ClothingOverlay:
    def __init__(self):
        """Initialize the clothing overlay system"""
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.clothing_path = os.path.join(self.base_path, 'clothing')

        # Current clothing items
        self.current_top = None
        self.current_bottom = None
        self.current_clothing_images = []
        self.clothing_index = 0

        # Clothing offsets (will be adjusted based on body position)
        self.top_offset = (0, 0)
        self.bottom_offset = (0, 0)

    def load_clothing_for_body_type(self, gender, body_type, clothing_type='top'):
        """
        Load clothing options for a specific body type and gender

        Args:
            gender: 'male' or 'female'
            body_type: 'ideal', 'under weight', or 'over weight'
            clothing_type: 'top' or 'bottom'

        Returns:
            List of clothing image paths
        """
        clothing_folder = os.path.join(
            self.clothing_path, body_type, gender, clothing_type)
        clothing_files = []

        # Check if the path exists
        if os.path.exists(clothing_folder):
            for file in os.listdir(clothing_folder):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    clothing_files.append(os.path.join(clothing_folder, file))

        return clothing_files

    def load_clothing_options(self, gender, body_type):
        """
        Load all available clothing options for a specific body type and gender

        Args:
            gender: 'male' or 'female' 
            body_type: 'ideal', 'under weight', or 'over weight'
        """
        top_files = self.load_clothing_for_body_type(gender, body_type, 'top')
        bottom_files = self.load_clothing_for_body_type(
            gender, body_type, 'bottom')

        # Combine and store all clothing options
        self.current_clothing_images = top_files + bottom_files
        self.clothing_index = 0

        # Load the first clothing item if available
        if len(self.current_clothing_images) > 0:
            self.current_top = self.current_clothing_images[0]

    def next_clothing(self):
        """Switch to the next clothing item"""
        if len(self.current_clothing_images) > 0:
            self.clothing_index = (
                self.clothing_index + 1) % len(self.current_clothing_images)
            return self.clothing_index
        return -1

    def previous_clothing(self):
        """Switch to the previous clothing item"""
        if len(self.current_clothing_images) > 0:
            self.clothing_index = (
                self.clothing_index - 1) % len(self.current_clothing_images)
            return self.clothing_index
        return -1

    def get_current_clothing(self):
        """Get the current clothing item path"""
        if len(self.current_clothing_images) > 0:
            return self.current_clothing_images[self.clothing_index]
        return None

    def _convert_to_grayscale_preserving_alpha(self, image):
        """
        Convert a color image to grayscale while preserving the alpha channel

        Args:
            image: Input RGBA image

        Returns:
            Grayscale image with alpha channel preserved
        """
        # Check if image has an alpha channel
        if image.shape[2] == 4:
            # Split the image into color and alpha channels
            rgb = image[:, :, :3]
            alpha = image[:, :, 3]

            # Convert the color channels to grayscale
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

            # Create a new grayscale image with the alpha channel
            grayscale = np.zeros(
                (image.shape[0], image.shape[1], 4), dtype=np.uint8)
            grayscale[:, :, 0] = gray
            grayscale[:, :, 1] = gray
            grayscale[:, :, 2] = gray
            grayscale[:, :, 3] = alpha

            return grayscale
        else:
            # If no alpha channel, just convert to grayscale
            return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    def overlay_clothing(self, frame, body_landmarks=None):
        """
        Overlay clothing on the person in the frame

        Args:
            frame: Input frame
            body_landmarks: MediaPipe pose landmarks (optional)

        Returns:
            Frame with clothing overlay
        """
        result_frame = frame.copy()

        # Get current clothing
        clothing_path = self.get_current_clothing()
        if not clothing_path:
            return result_frame

        try:
            # Load clothing image with alpha channel
            clothing_img = cv2.imread(clothing_path, cv2.IMREAD_UNCHANGED)

            # Skip if clothing image couldn't be loaded
            if clothing_img is None:
                return result_frame

            # Convert clothing image to monochrome while preserving alpha
            clothing_img = self._convert_to_grayscale_preserving_alpha(
                clothing_img)

            # If no body landmarks, use default positioning in center of frame
            h, w = frame.shape[:2]
            clothing_h, clothing_w = clothing_img.shape[:2]

            if body_landmarks is None:
                # Default position in center
                x_offset = (w - clothing_w) // 2
                y_offset = (h - clothing_h) // 2
            else:
                # Adjust position based on body landmarks
                # This would typically use shoulder and hip positions
                # For now, let's just use a simple approximation
                x_offset = (w - clothing_w) // 2
                y_offset = h // 3  # Position at about 1/3 from top

            # Check if this is a top or bottom
            is_top = 'top' in clothing_path.lower()
            if is_top:
                self.top_offset = (x_offset, y_offset)
            else:
                # If it's a bottom, position it lower
                y_offset = h // 2  # Position at about middle of frame
                self.bottom_offset = (x_offset, y_offset)

            # For this example, use simple alpha blending
            # A more sophisticated approach would consider body pose and deform clothing
            self._alpha_blend(result_frame, clothing_img, x_offset, y_offset)

            # Add clothing info
            clothing_name = os.path.basename(clothing_path).split('.')[0]
            cv2.putText(result_frame, f"Outfit: {clothing_name}",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (220, 220, 220), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error overlaying clothing: {e}")

        return result_frame

    def _alpha_blend(self, background, foreground, x_offset, y_offset):
        """
        Alpha blend foreground onto background at specified position

        Args:
            background: Background image (modified in-place)
            foreground: Foreground image with alpha channel
            x_offset: X position
            y_offset: Y position
        """
        # Get dimensions
        bg_h, bg_w = background.shape[:2]
        fg_h, fg_w = foreground.shape[:2]

        # Calculate the region where the foreground will be placed
        x_start = max(0, x_offset)
        y_start = max(0, y_offset)
        x_end = min(bg_w, x_offset + fg_w)
        y_end = min(bg_h, y_offset + fg_h)

        # Calculate corresponding region in foreground image
        fg_x_start = max(0, -x_offset)
        fg_y_start = max(0, -y_offset)
        fg_x_end = fg_x_start + (x_end - x_start)
        fg_y_end = fg_y_start + (y_end - y_start)

        # Extract regions
        roi = background[y_start:y_end, x_start:x_end]

        # Check if foreground has alpha channel
        if foreground.shape[2] == 4:
            fg_region = foreground[fg_y_start:fg_y_end,
                                   fg_x_start:fg_x_end, :3]
            alpha = foreground[fg_y_start:fg_y_end,
                               fg_x_start:fg_x_end, 3] / 255.0

            # Alpha blending
            for c in range(3):
                roi[:, :, c] = (1 - alpha) * roi[:, :, c] + \
                    alpha * fg_region[:, :, c]
        else:
            # No alpha channel, direct copy
            foreground_region = foreground[fg_y_start:fg_y_end,
                                           fg_x_start:fg_x_end]
            roi[:] = foreground_region
