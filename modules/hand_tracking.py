"""
Hand Tracking Module for Virtual Try-On Game
This module handles hand gesture recognition for controlling the virtual try-on interface.
"""

import cv2
import mediapipe as mp
import numpy as np

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
        
    def is_selecting(self):
        """
        Determine if the selection gesture is being made
        (index finger and little finger extended, others closed)
        
        Returns:
            Boolean indicating whether selection gesture is detected
        """
        if not self.landmarks:
            return False
            
        hand_landmarks = self.landmarks[0]  # Use first hand only
        
        # Get positions of relevant finger landmarks
        wrist = hand_landmarks.landmark[0]
        index_mcp = hand_landmarks.landmark[5]  # Index finger base
        index_tip = hand_landmarks.landmark[8]  # Index finger tip
        middle_tip = hand_landmarks.landmark[12]  # Middle finger tip
        ring_tip = hand_landmarks.landmark[16]  # Ring finger tip
        pinky_tip = hand_landmarks.landmark[20]  # Little finger tip
        
        # Check if index finger is extended (pointing up)
        index_extended = index_tip.y < index_mcp.y
        
        # Check if little finger is extended
        pinky_extended = pinky_tip.y < wrist.y
        
        # Check if other fingers are closed (below their base position)
        middle_closed = middle_tip.y > index_mcp.y
        ring_closed = ring_tip.y > index_mcp.y
        
        # Return true if index and pinky are extended, others closed
        return index_extended and pinky_extended and middle_closed and ring_closed
