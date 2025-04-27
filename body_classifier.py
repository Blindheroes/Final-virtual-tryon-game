"""
Body Classifier Module for Virtual Try-On Game
Handles body type detection and classification
"""

import cv2
import numpy as np
import mediapipe as mp


class BodyClassifier:
    def __init__(self):
        """Initialize the body classifier"""
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Body type classifications
        self.body_types = {
            'underweight': 'under weight',
            'normal': 'ideal',
            'overweight': 'over weight'
        }

        # Store measurements
        self.measurements = {
            'shoulder_width': 0,
            'hip_width': 0,
            'height': 0
        }

        # Current classification
        self.current_body_type = None
        self.gender = None

    def detect_body_landmarks(self, frame):
        """
        Detect body landmarks using MediaPipe Pose

        Args:
            frame: Input frame

        Returns:
            Tuple of (processed frame, results)
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self.pose.process(frame_rgb)

        return results

    def classify_body_type(self, results, frame, gender='male'):
        """
        Classify body type based on pose landmarks

        Args:
            results: MediaPipe pose results
            frame: Input frame for visualization
            gender: Gender to use for classification ('male' or 'female')

        Returns:
            Tuple of (body type, visualized frame)
        """
        self.gender = gender
        frame_with_landmarks = frame.copy()
        h, w, _ = frame.shape

        if not results.pose_landmarks:
            return None, frame

        # Draw landmarks in monochrome
        self.mp_drawing.draw_landmarks(
            frame_with_landmarks,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(220, 220, 220), thickness=2, circle_radius=1),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=(180, 180, 180), thickness=2)
        )

        # Extract key landmarks for measurements
        landmarks = results.pose_landmarks.landmark

        # Shoulder measurement (distance between LEFT_SHOULDER and RIGHT_SHOULDER)
        l_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                      int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
        r_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                      int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
        shoulder_width = np.sqrt((r_shoulder[0] - l_shoulder[0])**2 +
                                 (r_shoulder[1] - l_shoulder[1])**2)

        # Hip measurement (distance between LEFT_HIP and RIGHT_HIP)
        l_hip = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x * w),
                 int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y * h))
        r_hip = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                 int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y * h))
        hip_width = np.sqrt((r_hip[0] - l_hip[0]) **
                            2 + (r_hip[1] - l_hip[1])**2)

        # Calculate approximate height (from HEAD to ANKLE)
        top_head = (int(landmarks[self.mp_pose.PoseLandmark.NOSE].x * w),
                    int((landmarks[self.mp_pose.PoseLandmark.NOSE].y - 0.2) * h))  # Approximate top of head
        l_ankle = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x * w),
                   int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * h))
        height = np.sqrt((l_ankle[0] - top_head[0]) **
                         2 + (l_ankle[1] - top_head[1])**2)

        # Store measurements
        self.measurements['shoulder_width'] = shoulder_width
        self.measurements['hip_width'] = hip_width
        self.measurements['height'] = height

        # Calculate body type based on measurements
        # This is a simplified model - real classification would be more complex
        shoulder_hip_ratio = shoulder_width / \
            max(hip_width, 1)  # Prevent division by zero

        # Different ratio thresholds for male and female
        if gender == 'male':
            if shoulder_hip_ratio < 1.1:
                body_type = 'underweight'
            elif shoulder_hip_ratio > 1.4:
                body_type = 'overweight'
            else:
                body_type = 'normal'
        else:  # female
            if shoulder_hip_ratio > 0.9:
                body_type = 'underweight'
            elif shoulder_hip_ratio < 0.75:
                body_type = 'overweight'
            else:
                body_type = 'normal'

        # Draw measurements on frame
        # Shoulder line
        cv2.line(frame_with_landmarks, l_shoulder,
                 r_shoulder, (220, 220, 220), 2)
        cv2.putText(frame_with_landmarks, f"Shoulder: {shoulder_width:.0f}",
                    (r_shoulder[0] + 10, r_shoulder[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Hip line
        cv2.line(frame_with_landmarks, l_hip, r_hip, (220, 220, 220), 2)
        cv2.putText(frame_with_landmarks, f"Hip: {hip_width:.0f}",
                    (r_hip[0] + 10, r_hip[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Height line
        cv2.line(frame_with_landmarks, top_head, l_ankle, (220, 220, 220), 1)
        cv2.putText(frame_with_landmarks, f"Height: {height:.0f}",
                    (top_head[0] - 100, int((top_head[1] + l_ankle[1])/2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Overall body type
        cv2.putText(frame_with_landmarks, f"Type: {self.body_types[body_type]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

        # Store current classification
        self.current_body_type = self.body_types[body_type]

        return self.body_types[body_type], frame_with_landmarks

    def get_folder_path(self):
        """
        Get the appropriate folder path for clothing based on body type and gender

        Returns:
            Path string for clothing folder
        """
        if not self.current_body_type or not self.gender:
            return None

        return f"clothing/{self.current_body_type}/{self.gender}"
