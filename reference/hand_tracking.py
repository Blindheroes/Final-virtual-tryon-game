"""
Hand Tracking Module for Virtual Try-On Game
This module handles hand gesture recognition for controlling the virtual try-on interface.
"""

import cv2
import mediapipe as mp
import numpy as np
import math


class HandTracker:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,          # Track only one hand for simplicity
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Finger landmarks
        self.landmarks = None
        self.frame_height = 0
        self.frame_width = 0

        # Finger angle thresholds (in degrees)
        self.extension_threshold = 160  # Angle above which a finger is considered extended
        # Angle below which a finger is considered flexed/closed
        self.flexion_threshold = 120

    def process_frame(self, frame):
        """
        Process the frame and detect hand landmarks

        Args:
            frame: RGB image frame

        Returns:
            Frame with hand landmarks drawn (if enabled)
        """
        self.frame_height, self.frame_width = frame.shape[:2]

        # Process the frame to find hands
        results = self.hands.process(frame)
        self.landmarks = results.multi_hand_landmarks

        return frame

    def get_pointer_position(self):
        """
        Get the position of the index finger tip (pointer)

        Returns:
            (x, y) tuple of the pointer position, or None if not detected
        """
        if not self.landmarks:
            return None

        # Get the index finger tip position (landmark 8)
        hand_landmarks = self.landmarks[0]  # Use first hand only
        index_tip = hand_landmarks.landmark[8]

        # Convert normalized coordinates to pixel coordinates
        x = int(index_tip.x * self.frame_width)
        y = int(index_tip.y * self.frame_height)

        return (x, y)

    def _calculate_angle(self, p1, p2, p3):
        """
        Calculate the angle between three points

        Args:
            p1, p2, p3: Three points where p2 is the middle point (joint)

        Returns:
            Angle in degrees
        """
        # Convert landmarks to numpy arrays
        p1_array = np.array([p1.x, p1.y, p1.z])
        p2_array = np.array([p2.x, p2.y, p2.z])
        p3_array = np.array([p3.x, p3.y, p3.z])

        # Calculate vectors
        v1 = p1_array - p2_array
        v2 = p3_array - p2_array

        # Calculate angle using dot product
        cosine_angle = np.dot(v1, v2) / \
            (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        # Convert to degrees
        return math.degrees(angle)

    def get_finger_angles(self):
        """
        Calculate the angles for each finger

        Returns:
            Dictionary of finger angles or None if no hand detected
        """
        if not self.landmarks:
            return None

        hand_landmarks = self.landmarks[0]
        landmarks = hand_landmarks.landmark

        # Landmark indices for each finger joint
        # Format: [finger_name, [mcp, pip, dip, tip]]
        finger_indices = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }

        angles = {}

        # Calculate angle for each finger
        for finger_name, indices in finger_indices.items():
            # For non-thumb fingers
            if finger_name != 'thumb':
                # Angle at MCP joint (between palm and proximal phalanx)
                mcp_angle = self._calculate_angle(
                    landmarks[0],      # wrist
                    landmarks[indices[0]],  # mcp
                    landmarks[indices[1]]   # pip
                )

                # Angle at PIP joint (between proximal and middle phalanx)
                pip_angle = self._calculate_angle(
                    landmarks[indices[0]],  # mcp
                    landmarks[indices[1]],  # pip
                    landmarks[indices[2]]   # dip
                )

                angles[finger_name] = {
                    'mcp': mcp_angle,
                    'pip': pip_angle
                }
            else:
                # Special case for thumb
                cmc_angle = self._calculate_angle(
                    landmarks[0],      # wrist
                    landmarks[indices[0]],  # cmc
                    landmarks[indices[1]]   # mcp
                )

                mcp_angle = self._calculate_angle(
                    landmarks[indices[0]],  # cmc
                    landmarks[indices[1]],  # mcp
                    landmarks[indices[2]]   # ip
                )

                angles[finger_name] = {
                    'cmc': cmc_angle,
                    'mcp': mcp_angle
                }

        return angles

    def is_finger_extended(self, finger_name):
        """
        Determine if a specific finger is extended based on joint angles

        Args:
            finger_name: String name of the finger ('thumb', 'index', 'middle', 'ring', 'pinky')

        Returns:
            Boolean indicating if the finger is extended
        """
        angles = self.get_finger_angles()
        if not angles:
            return False

        if finger_name == 'thumb':
            # Thumb is extended if both CMC and MCP joints are relatively straight
            return angles[finger_name]['cmc'] > self.extension_threshold * 0.7 and \
                angles[finger_name]['mcp'] > self.extension_threshold * 0.8
        else:
            # For other fingers, check MCP and PIP joints
            # A finger is extended if MCP and PIP joints are relatively straight
            return angles[finger_name]['mcp'] > self.extension_threshold and \
                angles[finger_name]['pip'] > self.extension_threshold * 0.8

    def is_selecting(self):
        """
        Determine if the selection gesture is being made
        (index finger and little finger extended, others closed)

        Returns:
            Boolean indicating whether selection gesture is detected
        """
        if not self.landmarks:
            return False

        # Check finger states using angle-based detection
        index_extended = self.is_finger_extended('index')
        middle_extended = self.is_finger_extended('middle')
        ring_extended = self.is_finger_extended('ring')
        pinky_extended = self.is_finger_extended('pinky')

        # Return true if index and pinky are extended, others closed
        return index_extended and pinky_extended and not middle_extended and not ring_extended

    def is_pointing(self):
        """
        Determine if the hand is making a pointing gesture
        (only index finger extended, all others closed)

        Returns:
            Boolean indicating whether pointing gesture is detected
        """
        if not self.landmarks:
            return False

        # Check finger states using angle-based detection
        thumb_extended = self.is_finger_extended('thumb')
        index_extended = self.is_finger_extended('index')
        middle_extended = self.is_finger_extended('middle')
        ring_extended = self.is_finger_extended('ring')
        pinky_extended = self.is_finger_extended('pinky')

        # Return true if only index finger is extended
        return index_extended and not thumb_extended and not middle_extended and not ring_extended and not pinky_extended

    def is_grabbing(self):
        """
        Determine if the hand is making a grabbing gesture
        (all fingers closed/flexed)

        Returns:
            Boolean indicating whether grabbing gesture is detected
        """
        if not self.landmarks:
            return False

        # Check if all fingers are flexed
        return not any([
            self.is_finger_extended('thumb'),
            self.is_finger_extended('index'),
            self.is_finger_extended('middle'),
            self.is_finger_extended('ring'),
            self.is_finger_extended('pinky')
        ])

    def is_open_palm(self):
        """
        Determine if the hand is making an open palm gesture
        (all fingers extended)

        Returns:
            Boolean indicating whether open palm gesture is detected
        """
        if not self.landmarks:
            return False

        # Check if all fingers are extended
        return all([
            self.is_finger_extended('thumb'),
            self.is_finger_extended('index'),
            self.is_finger_extended('middle'),
            self.is_finger_extended('ring'),
            self.is_finger_extended('pinky')
        ])

    def adjust_thresholds(self, extension_threshold=None, flexion_threshold=None):
        """
        Adjust the angle thresholds for finger state detection

        Args:
            extension_threshold: Angle above which a finger is considered extended (in degrees)
            flexion_threshold: Angle below which a finger is considered flexed/closed (in degrees)
        """
        if extension_threshold is not None:
            self.extension_threshold = extension_threshold
        if flexion_threshold is not None:
            self.flexion_threshold = flexion_threshold

    def visualize_finger_states(self, frame):
        """
        Draw visual indicators for finger states on the frame

        Args:
            frame: Image frame to draw on

        Returns:
            Frame with finger state visualization
        """
        if not self.landmarks:
            return frame

        # Get finger states
        finger_states = {
            'thumb': self.is_finger_extended('thumb'),
            'index': self.is_finger_extended('index'),
            'middle': self.is_finger_extended('middle'),
            'ring': self.is_finger_extended('ring'),
            'pinky': self.is_finger_extended('pinky')
        }

        # Draw hand landmarks
        for hand_landmarks in self.landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        # Add finger state indicators
        y_pos = 30
        for finger, is_extended in finger_states.items():
            color = (0, 255, 0) if is_extended else (
                0, 0, 255)  # Green if extended, red if flexed
            cv2.putText(frame, f"{finger}: {'Extended' if is_extended else 'Flexed'}",
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30

        # Add current gesture label
        gesture = "None"
        if self.is_selecting():
            gesture = "Selecting"
        elif self.is_pointing():
            gesture = "Pointing"
        elif self.is_grabbing():
            gesture = "Grabbing"
        elif self.is_open_palm():
            gesture = "Open Palm"

        cv2.putText(frame, f"Gesture: {gesture}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame