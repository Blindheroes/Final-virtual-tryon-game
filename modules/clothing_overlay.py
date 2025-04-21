"""
Clothing Overlay Module for Virtual Try-On Game
This module handles clothing overlay on the user's body using MediaPipe Pose.
Based on the provided VTO_2D_polo.py example.
"""

import cv2
import mediapipe as mp
import numpy as np
import os

class ClothingOverlay:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Clothing options (these would be actual file paths in a real application)
        self.tops = {
            "male": ["polo", "tshirt", "shirt"],
            "female": ["blouse", "tshirt", "top"]
        }
        
        self.bottoms = {
            "male": ["jeans", "shorts", "pants"],
            "female": ["skirt", "jeans", "pants"]
        }
        
        # Current selection indices
        self.current_top_index = 0
        self.current_bottom_index = 0
        
        # Scaling factors
        self.shirt_scale = 0.55
        self.pants_scale = 2.0
        self.offset_y = -50
    
    def load_clothing_image(self, clothing_type, gender, item):
        """
        Load clothing images - for MVP, we'll just create colored placeholders
        
        Args:
            clothing_type: "top" or "bottom"
            gender: "male" or "female"
            item: specific clothing item name
            
        Returns:
            Image with alpha channel
        """
        # In a real application, we would load actual clothing images
        # For MVP, create colored rectangles with alpha channel
        
        if clothing_type == "top":
            # Create a placeholder top (shirt)
            width, height = 300, 400
            image = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Set color based on gender and item
            if gender == "male":
                if item == "polo":
                    color = (0, 0, 255, 200)  # Red with alpha
                elif item == "tshirt":
                    color = (0, 255, 0, 200)  # Green with alpha
                else:
                    color = (255, 0, 0, 200)  # Blue with alpha
            else:  # female
                if item == "blouse":
                    color = (255, 0, 255, 200)  # Pink with alpha
                elif item == "tshirt":
                    color = (0, 255, 255, 200)  # Cyan with alpha
                else:
                    color = (255, 255, 0, 200)  # Yellow with alpha
            
            # Create a simple shirt shape
            cv2.rectangle(image, (100, 50), (200, 350), color, -1)  # Body
            cv2.rectangle(image, (50, 50), (100, 200), color, -1)   # Left arm
            cv2.rectangle(image, (200, 50), (250, 200), color, -1)  # Right arm
            
            # Add alpha channel (make background transparent)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(mask, (100, 50), (200, 350), 255, -1)
            cv2.rectangle(mask, (50, 50), (100, 200), 255, -1)
            cv2.rectangle(mask, (200, 50), (250, 200), 255, -1)
            
            # Set alpha channel
            image[:, :, 3] = mask
            
        else:  # bottom
            # Create a placeholder bottom (pants/skirt)
            width, height = 300, 500
            image = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Set color based on gender and item
            if gender == "male":
                if item == "jeans":
                    color = (0, 0, 128, 200)  # Dark blue with alpha
                elif item == "shorts":
                    color = (0, 128, 0, 200)  # Dark green with alpha
                else:
                    color = (128, 0, 0, 200)  # Dark red with alpha
            else:  # female
                if item == "skirt":
                    color = (128, 0, 128, 200)  # Dark pink with alpha
                elif item == "jeans":
                    color = (0, 0, 128, 200)  # Dark blue with alpha
                else:
                    color = (128, 128, 0, 200)  # Dark yellow with alpha
            
            # Create pants/skirt shape
            if item == "skirt" and gender == "female":
                # A-line skirt shape
                pts = np.array([[100, 0], [200, 0], [250, height-1], [50, height-1]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(image, [pts], color)
            else:
                # Pants shape
                cv2.rectangle(image, (100, 0), (200, height-1), color, -1)  # Center
                cv2.rectangle(image, (75, height//2), (125, height-1), color, -1)  # Left leg
                cv2.rectangle(image, (175, height//2), (225, height-1), color, -1)  # Right leg
            
            # Add alpha channel
            mask = np.zeros((height, width), dtype=np.uint8)
            if item == "skirt" and gender == "female":
                cv2.fillPoly(mask, [pts], 255)
            else:
                cv2.rectangle(mask, (100, 0), (200, height-1), 255, -1)
                cv2.rectangle(mask, (75, height//2), (125, height-1), 255, -1)
                cv2.rectangle(mask, (175, height//2), (225, height-1), 255, -1)
            
            # Set alpha channel
            image[:, :, 3] = mask
            
        return image
    
    def overlay_image(self, background, foreground):
        """
        Overlay foreground image on background using alpha channel
        
        Args:
            background: Background image (BGR)
            foreground: Foreground image with alpha channel (BGRA)
            
        Returns:
            Combined image (BGR)
        """
        # Make sure foreground is not larger than background
        h, w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        if h > bg_h or w > bg_w:
            return background
        
        # Extract alpha channel and create a mask
        if foreground.shape[2] == 4:
            alpha = foreground[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            # Convert foreground to BGR (remove alpha channel)
            fg_rgb = foreground[:, :, :3]
            
            # Apply alpha blending
            blended = alpha * fg_rgb + (1 - alpha) * background[:h, :w, :3]
            
            # Copy the blended result back to the background
            result = background.copy()
            result[:h, :w, :3] = blended
            
            return result
        else:
            # If no alpha channel, just use addWeighted
            result = background.copy()
            cv2.addWeighted(foreground, 0.5, background[:h, :w, :], 0.5, 0, result[:h, :w, :])
            return result
    
    def apply_clothing(self, frame, gender, body_type, selected_clothing):
        """
        Apply clothing overlays to the frame based on pose landmarks
        
        Args:
            frame: Video frame
            gender: "male" or "female"
            body_type: Body type from scanner
            selected_clothing: Dict with "top" and "bottom" keys
            
        Returns:
            Frame with clothing overlaid
        """
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        # Create a copy of the frame for overlaying
        output_frame = frame.copy()
        
        if not results.pose_landmarks:
            return output_frame
            
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Get landmark positions
        lm = results.pose_landmarks.landmark
        
        # Extract key landmarks for clothing positioning
        try:
            # Shoulders
            left_shoulder = np.array([
                lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h
            ], dtype=np.float32)
            
            right_shoulder = np.array([
                lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h
            ], dtype=np.float32)
            
            # Hips
            left_hip = np.array([
                lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                lm[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h
            ], dtype=np.float32)
            
            right_hip = np.array([
                lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                lm[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h
            ], dtype=np.float32)
            
            # Ankles
            left_ankle = np.array([
                lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                lm[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h
            ], dtype=np.float32)
            
            right_ankle = np.array([
                lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                lm[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h
            ], dtype=np.float32)
            
            # Calculate midpoints
            mid_hip = (left_hip + right_hip) / 2.0
            mid_ankle = (left_ankle + right_ankle) / 2.0
            center_shoulder = (left_shoulder + right_shoulder) / 2.0
            center_pants = (left_hip + right_hip) / 2.0
            
            # Apply top (shirt)
            # Get current top item
            top_item = self.tops[gender][self.current_top_index]
            top_image = self.load_clothing_image("top", gender, top_item)
            
            # Define destination triangle for shirt
            pts_dst_shirt = np.float32([
                left_shoulder + np.array([0, self.offset_y]),
                right_shoulder + np.array([0, self.offset_y]),
                mid_hip
            ])
            
            # Scale around the shoulder center
            pts_dst_shirt_scaled = center_shoulder + self.shirt_scale * (pts_dst_shirt - center_shoulder)
            
            # Define source triangle points from the shirt image
            shirt_h, shirt_w = top_image.shape[:2]
            pts_src_shirt = np.float32([
                [shirt_w * 0.3, 0],        # Left shoulder
                [shirt_w * 0.7, 0],        # Right shoulder
                [shirt_w * 0.5, shirt_h]   # Bottom center
            ])
            
            # Compute the affine transformation and warp the shirt image
            M_shirt = cv2.getAffineTransform(pts_src_shirt, pts_dst_shirt_scaled)
            warped_shirt = cv2.warpAffine(top_image, M_shirt, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_TRANSPARENT)
            
            # Apply bottom (pants/skirt)
            # Get current bottom item
            bottom_item = self.bottoms[gender][self.current_bottom_index]
            bottom_image = self.load_clothing_image("bottom", gender, bottom_item)
            
            # Define destination triangle for pants
            pts_dst_pants = np.float32([left_hip, right_hip, mid_ankle])
            pts_dst_pants_scaled = center_pants + self.pants_scale * (pts_dst_pants - center_pants)
            
            # Define source triangle points from the pants image
            pants_h, pants_w = bottom_image.shape[:2]
            pts_src_pants = np.float32([
                [pants_w * 0.3, 0],        # Left hip
                [pants_w * 0.7, 0],        # Right hip
                [pants_w * 0.5, pants_h]   # Bottom center
            ])
            
            # Compute the affine transform and warp the pants image
            M_pants = cv2.getAffineTransform(pts_src_pants, pts_dst_pants_scaled)
            warped_pants = cv2.warpAffine(bottom_image, M_pants, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_TRANSPARENT)
            
            # Overlay pants first (they go underneath the shirt)
            output_frame = self.overlay_image(output_frame, warped_pants)
            # Then overlay shirt
            output_frame = self.overlay_image(output_frame, warped_shirt)
            
        except Exception as e:
            print(f"Error applying clothing overlay: {e}")
        
        return output_frame
    
    def next_top(self):
        """Switch to next top option"""
        self.current_top_index = (self.current_top_index + 1) % len(self.tops["male"])
        
    def prev_top(self):
        """Switch to previous top option"""
        self.current_top_index = (self.current_top_index - 1) % len(self.tops["male"])
        
    def next_bottom(self):
        """Switch to next bottom option"""
        self.current_bottom_index = (self.current_bottom_index + 1) % len(self.bottoms["male"])
        
    def prev_bottom(self):
        """Switch to previous bottom option"""
        self.current_bottom_index = (self.current_bottom_index - 1) % len(self.bottoms["male"])
