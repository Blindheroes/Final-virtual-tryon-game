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
        self.top_images = []
        self.bottom_images = []
        self.top_index = 0
        self.bottom_index = 0
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
        self.top_images = self.load_clothing_for_body_type(
            gender, body_type, 'top')
        self.bottom_images = self.load_clothing_for_body_type(
            gender, body_type, 'bottom')

        # Combine and store all clothing options
        self.current_clothing_images = self.top_images + self.bottom_images
        self.clothing_index = 0

        # Load the first clothing item if available
        if len(self.top_images) > 0:
            self.current_top = self.top_images[0]
        if len(self.bottom_images) > 0:
            self.current_bottom = self.bottom_images[0]

    def next_clothing(self, clothing_type='top'):
        """Switch to the next clothing item"""
        if clothing_type == 'top' and len(self.top_images) > 0:
            self.top_index = (self.top_index + 1) % len(self.top_images)
            self.current_top = self.top_images[self.top_index]
            return self.top_index
        elif clothing_type == 'bottom' and len(self.bottom_images) > 0:
            self.bottom_index = (self.bottom_index +
                                 1) % len(self.bottom_images)
            self.current_bottom = self.bottom_images[self.bottom_index]
            return self.bottom_index
        return -1

    def previous_clothing(self, clothing_type='top'):
        """Switch to the previous clothing item"""
        if clothing_type == 'top' and len(self.top_images) > 0:
            self.top_index = (self.top_index - 1) % len(self.top_images)
            self.current_top = self.top_images[self.top_index]
            return self.top_index
        elif clothing_type == 'bottom' and len(self.bottom_images) > 0:
            self.bottom_index = (self.bottom_index -
                                 1) % len(self.bottom_images)
            self.current_bottom = self.bottom_images[self.bottom_index]
            return self.bottom_index
        return -1

    def get_current_clothing(self, clothing_type='top'):
        """Get the current clothing item path"""
        if clothing_type == 'top' and len(self.top_images) > 0:
            return self.current_top
        elif clothing_type == 'bottom' and len(self.bottom_images) > 0:
            return self.current_bottom
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
        top_path = self.get_current_clothing('top')
        bottom_path = self.get_current_clothing('bottom')

        try:
            # Load top clothing image with alpha channel
            if top_path:
                top_img = cv2.imread(top_path, cv2.IMREAD_UNCHANGED)
                if top_img is not None:
                    result_frame = self._overlay_clothing_item(
                        result_frame, top_img, body_landmarks, 'top')

            # Load bottom clothing image with alpha channel
            if bottom_path:
                bottom_img = cv2.imread(bottom_path, cv2.IMREAD_UNCHANGED)
                if bottom_img is not None:
                    result_frame = self._overlay_clothing_item(
                        result_frame, bottom_img, body_landmarks, 'bottom')

            # Add clothing info
            top_name = os.path.basename(top_path).split('.')[
                0] if top_path else "None"
            bottom_name = os.path.basename(bottom_path).split('.')[
                0] if bottom_path else "None"
            cv2.putText(result_frame, f"Top: {top_name}, Bottom: {bottom_name}",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (220, 220, 220), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Error overlaying clothing: {e}")

        return result_frame

    def _overlay_clothing_item(self, frame, clothing_img, body_landmarks, clothing_type):
        """
        Overlay a single clothing item on the person in the frame

        Args:
            frame: Input frame
            clothing_img: Clothing image with alpha channel
            body_landmarks: MediaPipe pose landmarks (optional)
            clothing_type: 'top' or 'bottom'

        Returns:
            Frame with clothing item overlay
        """
        result_frame = frame.copy()
        h, w = frame.shape[:2]

        # Import needed for mediapipe landmarks
        import mediapipe as mp
        mp_pose = mp.solutions.pose

        # If we have body landmarks, use them for better positioning
        if body_landmarks is not None and body_landmarks.pose_landmarks:
            lm = body_landmarks.pose_landmarks.landmark

            # Define scaling factors
            shirt_scale = 0.55
            pants_scale = 1.5
            offset_y = -20

            if clothing_type == 'top':
                # Extract shoulder landmarks
                left_shoulder = np.array([
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h
                ], dtype=np.float32)
                right_shoulder = np.array([
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
                ], dtype=np.float32)

                # Extract hip landmarks for the bottom of the shirt
                left_hip = np.array([
                    lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                    lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h
                ], dtype=np.float32)
                right_hip = np.array([
                    lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h
                ], dtype=np.float32)
                mid_hip = (left_hip + right_hip) / 2.0

                left_ankle = np.array([
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h
                ], dtype=np.float32)
                right_ankle = np.array([
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h
                ], dtype=np.float32)
                mid_ankle = (left_ankle + right_ankle) / 2.0

                # Define destination points for the shirt
                pts_dst = np.float32([
                    left_shoulder + np.array([0, offset_y]),
                    right_shoulder + np.array([0, offset_y]),
                    # mid_hip
                    mid_ankle
                ])

                # Apply scaling relative to the center
                center = (left_shoulder + right_shoulder) / 2.0
                pts_dst_scaled = center + shirt_scale * (pts_dst - center)

                # Define source triangle points from the shirt image
                shirt_h, shirt_w = clothing_img.shape[:2]
                pts_src = np.float32([
                    [shirt_w * 0.3, 0],        # Left shoulder point
                    [shirt_w * 0.7, 0],        # Right shoulder point
                    [shirt_w * 0.5, shirt_h]   # Bottom center of shirt
                ])

                # Compute affine transformation and warp the shirt
                M = cv2.getAffineTransform(pts_src, pts_dst_scaled)
                warped_clothing = cv2.warpAffine(clothing_img, M, (w, h),
                                                 flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_TRANSPARENT)

                # Overlay the warped shirt on the frame
                result_frame = self._overlay_image(
                    result_frame, warped_clothing)

            elif clothing_type == 'bottom':
                # Extract hip landmarks
                left_hip = np.array([
                    lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                    lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h
                ], dtype=np.float32)
                right_hip = np.array([
                    lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h
                ], dtype=np.float32)

                # Extract ankle landmarks
                left_ankle = np.array([
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                    lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h
                ], dtype=np.float32)
                right_ankle = np.array([
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h
                ], dtype=np.float32)
                mid_ankle = (left_ankle + right_ankle) / 2.0

                # Define destination points for the pants
                pts_dst = np.float32([left_hip, right_hip, mid_ankle])

                # Apply scaling relative to the center
                center = (left_hip + right_hip) / 2.0
                pts_dst_scaled = center + pants_scale * (pts_dst - center)

                # Define source triangle points for the pants
                pants_h, pants_w = clothing_img.shape[:2]
                pts_src = np.float32([
                    [pants_w * 0.3, 0],        # Left hip point
                    [pants_w * 0.7, 0],        # Right hip point
                    [pants_w * 0.5, pants_h]   # Bottom center of pants
                ])

                # Compute affine transformation and warp the pants
                M = cv2.getAffineTransform(pts_src, pts_dst_scaled)
                warped_clothing = cv2.warpAffine(clothing_img, M, (w, h),
                                                 flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_TRANSPARENT)

                # Overlay the warped pants on the frame
                result_frame = self._overlay_image(
                    result_frame, warped_clothing)

        else:
            # If no body landmarks, use simple positioning
            clothing_h, clothing_w = clothing_img.shape[:2]

            if clothing_type == 'top':
                # Default position for top in center
                x_offset = (w - clothing_w) // 2
                y_offset = h // 3  # Position at about 1/3 from top
                self.top_offset = (x_offset, y_offset)

                # Simple alpha blending
                self._alpha_blend(
                    result_frame, clothing_img, x_offset, y_offset)

            elif clothing_type == 'bottom':
                # Default position for bottom
                x_offset = (w - clothing_w) // 2
                y_offset = h // 2  # Position at about middle of frame
                self.bottom_offset = (x_offset, y_offset)

                # Simple alpha blending
                self._alpha_blend(
                    result_frame, clothing_img, x_offset, y_offset)

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

    def _overlay_image(self, background, foreground):
        """
        Overlay foreground image on background using the alpha channel

        Args:
            background: Background image
            foreground: Foreground image with alpha channel

        Returns:
            Combined image
        """
        # Create a copy of the background to avoid modifying the original
        result = background.copy()

        # Check if foreground has an alpha channel
        if foreground.shape[2] == 4:
            # Get the alpha channel
            alpha = foreground[:, :, 3] / 255.0

            # Get the RGB channels
            fg_rgb = foreground[:, :, :3]

            # Create an alpha mask with 3 channels
            alpha_3channel = np.stack([alpha, alpha, alpha], axis=2)

            # Blend using the alpha mask
            result = result * (1 - alpha_3channel) + fg_rgb * alpha_3channel

        else:
            # No alpha channel, use simple blending
            result = cv2.addWeighted(background, 0.5, foreground, 0.5, 0)

        return result.astype(np.uint8)

        # # DARI BERYL
        # result = background.copy()

        # if foreground.shape[2] == 4:
        #     # Separate the color and alpha channels.
        #     fg_rgb = foreground[:, :, :3]
        #     alpha_mask = foreground[:, :, 3] / 255.0
        #     # Blend the foreground and background.
        #     for c in range(3):
        #         result[:, :, c] = alpha_mask * fg_rgb[:, :, c] + \
        #             (1 - alpha_mask) * background[:, :, c]
        # else:
        #     result = cv2.addWeighted(background, 1, foreground, 0.5, 0)
        # return result
