"""
Test utilities for Virtual Try-On Game
Contains common functions used across all test modules
"""

import os
import cv2
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class TestLogger:
    """Class for logging test results"""
    
    def __init__(self, test_name):
        """Initialize test logger with test name"""
        self.test_name = test_name
        self.results = []
        self.start_time = datetime.now()
        
        # Create test results directory if it doesn't exist
        self.results_dir = "test_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
    def log_result(self, params, metrics):
        """
        Log a test result
        
        Args:
            params: Dictionary of test parameters (e.g. distance, lighting, etc.)
            metrics: Dictionary of performance metrics (e.g. accuracy, precision, etc.)
        """
        result = {**params, **metrics, "timestamp": datetime.now()}
        self.results.append(result)
        
    def save_results(self):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return
            
        # Generate filename with timestamp
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/{self.test_name}_{timestamp}.csv"
        
        # Convert results to DataFrame and save
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        return filename
        
    def generate_report(self):
        """Generate and save visualization of results"""
        if not self.results:
            print("No results to visualize")
            return
            
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Generate filename with timestamp
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/{self.test_name}_{timestamp}_report.png"
        
        # Create visualization based on test type
        plt.figure(figsize=(12, 8))
        
        if "distance" in self.test_name.lower():
            self._plot_distance_results(df)
        elif "light" in self.test_name.lower():
            self._plot_lighting_results(df)
        elif "adapt" in self.test_name.lower():
            self._plot_adaptation_results(df)
        elif "filter" in self.test_name.lower() or "temporal" in self.test_name.lower():
            self._plot_filtering_results(df)
        else:
            self._plot_general_results(df)
            
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Report saved to {filename}")
        plt.close()
        
        return filename
        
    def _plot_distance_results(self, df):
        """Plot results for distance-based tests"""
        # Group by distance and gesture
        grouped = df.groupby(['distance', 'gesture'])['accuracy'].mean().reset_index()
        
        distances = sorted(grouped['distance'].unique())
        gestures = sorted(grouped['gesture'].unique())
        
        # Plot bar chart
        plt.subplot(2, 1, 1)
        bar_width = 0.35
        index = np.arange(len(distances))
        
        for i, gesture in enumerate(gestures):
            gesture_data = grouped[grouped['gesture'] == gesture]
            plt.bar(index + i*bar_width, gesture_data['accuracy'], 
                    bar_width, label=gesture.capitalize())
            
        plt.xlabel('Distance (meters)')
        plt.ylabel('Accuracy')
        plt.title('Gesture Recognition Accuracy by Distance')
        plt.xticks(index + bar_width/2, distances)
        plt.legend()
        
        # Plot line chart
        plt.subplot(2, 1, 2)
        for gesture in gestures:
            gesture_data = grouped[grouped['gesture'] == gesture]
            plt.plot(gesture_data['distance'], gesture_data['accuracy'], 
                    marker='o', label=gesture.capitalize())
            
        plt.xlabel('Distance (meters)')
        plt.ylabel('Accuracy')
        plt.title('Gesture Recognition Accuracy Trend by Distance')
        plt.grid(True)
        plt.legend()
        
    def _plot_lighting_results(self, df):
        """Plot results for lighting-based tests"""
        # Group by lighting and gesture
        grouped = df.groupby(['lighting', 'gesture'])['accuracy'].mean().reset_index()
        
        lightings = sorted(grouped['lighting'].unique())
        gestures = sorted(grouped['gesture'].unique())
        
        # Plot bar chart
        plt.subplot(2, 1, 1)
        bar_width = 0.35
        index = np.arange(len(lightings))
        
        for i, gesture in enumerate(gestures):
            gesture_data = grouped[grouped['gesture'] == gesture]
            plt.bar(index + i*bar_width, gesture_data['accuracy'], 
                    bar_width, label=gesture.capitalize())
            
        plt.xlabel('Lighting Condition (lux)')
        plt.ylabel('Accuracy')
        plt.title('Gesture Recognition Accuracy by Lighting')
        plt.xticks(index + bar_width/2, lightings)
        plt.legend()
        
        # Plot line chart
        plt.subplot(2, 1, 2)
        for gesture in gestures:
            gesture_data = grouped[grouped['gesture'] == gesture]
            plt.plot(gesture_data['lighting'], gesture_data['accuracy'], 
                    marker='o', label=gesture.capitalize())
            
        plt.xlabel('Lighting Condition (lux)')
        plt.ylabel('Accuracy')
        plt.title('Gesture Recognition Accuracy Trend by Lighting')
        plt.grid(True)
        plt.legend()
        
    def _plot_adaptation_results(self, df):
        """Plot results for adaptation tests"""
        # Group by adaptation and distance
        grouped = df.groupby(['adaptation', 'distance'])['accuracy'].mean().reset_index()
        
        # Plot bar chart comparing with and without adaptation
        plt.subplot(2, 1, 1)
        
        # Create pivot table for easier plotting
        pivot = grouped.pivot(index='distance', columns='adaptation', values='accuracy')
        pivot.plot(kind='bar', ax=plt.gca())
        
        plt.xlabel('Distance (meters)')
        plt.ylabel('Accuracy')
        plt.title('Effect of Distance Adaptation on Accuracy')
        plt.legend(title='Adaptation')
        
        # Plot line chart showing improvement
        plt.subplot(2, 1, 2)
        
        # Calculate improvement
        improvement = pd.DataFrame()
        for distance in df['distance'].unique():
            with_adapt = df[(df['adaptation'] == True) & (df['distance'] == distance)]['accuracy'].mean()
            without_adapt = df[(df['adaptation'] == False) & (df['distance'] == distance)]['accuracy'].mean()
            improvement = improvement.append({
                'distance': distance,
                'improvement': with_adapt - without_adapt
            }, ignore_index=True)
            
        plt.plot(improvement['distance'], improvement['improvement'], marker='o')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Distance (meters)')
        plt.ylabel('Accuracy Improvement')
        plt.title('Improvement from Distance Adaptation')
        plt.grid(True)
        
    def _plot_filtering_results(self, df):
        """Plot results for temporal filtering tests"""
        # Group by filtering and gesture
        grouped = df.groupby(['filtering', 'gesture'])['accuracy'].mean().reset_index()
        
        # Plot bar chart comparing with and without filtering
        plt.subplot(2, 1, 1)
        
        # Create pivot table for easier plotting
        pivot = grouped.pivot(index='gesture', columns='filtering', values='accuracy')
        pivot.plot(kind='bar', ax=plt.gca())
        
        plt.xlabel('Gesture')
        plt.ylabel('Accuracy')
        plt.title('Effect of Temporal Filtering on Accuracy')
        plt.legend(title='Filtering')
        
        # Plot stability comparison 
        plt.subplot(2, 1, 2)
        
        # Group by filtering and calculate stability (inverse of standard deviation)
        stability = df.groupby(['filtering', 'gesture'])['stability'].mean().reset_index()
        pivot_stability = stability.pivot(index='gesture', columns='filtering', values='stability')
        pivot_stability.plot(kind='bar', ax=plt.gca())
        
        plt.xlabel('Gesture')
        plt.ylabel('Stability Score')
        plt.title('Effect of Temporal Filtering on Gesture Stability')
        plt.legend(title='Filtering')
        
    def _plot_general_results(self, df):
        """Plot general results when specific plot not available"""
        # Create a simple bar chart of accuracy by test condition
        if 'accuracy' in df.columns:
            # Find the main parameter that varies
            variance_cols = []
            for col in df.columns:
                if col not in ['accuracy', 'timestamp', 'stability', 'latency']:
                    if len(df[col].unique()) > 1:
                        variance_cols.append(col)
            
            if variance_cols:
                main_param = variance_cols[0]
                plt.figure(figsize=(10, 6))
                df.groupby(main_param)['accuracy'].mean().plot(kind='bar')
                plt.xlabel(main_param.capitalize())
                plt.ylabel('Accuracy')
                plt.title(f'Accuracy by {main_param.capitalize()}')
                plt.grid(True, axis='y')
        else:
            plt.text(0.5, 0.5, "No specific visualization available for this test", 
                    horizontalalignment='center', verticalalignment='center')


def calculate_lux(frame):
    """
    Estimate lighting level in lux from camera frame
    This is a rough approximation
    
    Args:
        frame: Input camera frame
        
    Returns:
        Estimated lighting level in lux
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness
    avg_brightness = np.mean(gray)
    
    # Map brightness (0-255) to approximate lux values
    # This is a very rough estimation and would need calibration
    # for accurate readings
    estimated_lux = avg_brightness * 2.5
    
    return estimated_lux
    
    
def estimate_distance(landmark_results):
    """
    Estimate distance from camera based on hand landmarks
    This is a rough approximation based on relative size of hand in frame
    
    Args:
        landmark_results: MediaPipe landmark results
        
    Returns:
        Estimated distance in meters
    """
    if not landmark_results or not landmark_results[0]:
        return None
        
    # Get hand landmarks
    landmarks = landmark_results[0].landmark
    
    # Calculate distance between thumb base and pinky base
    # as a measure of hand size in the frame
    thumb_base = np.array([landmarks[1].x, landmarks[1].y])
    pinky_base = np.array([landmarks[17].x, landmarks[17].y])
    
    distance_in_frame = np.linalg.norm(thumb_base - pinky_base)
    
    # Map this to approximate real-world distance
    # This needs calibration for accurate readings
    # Assuming standard hand size and camera parameters
    # Larger distance_in_frame = closer to camera
    if distance_in_frame > 0.25:
        estimated_meters = 1.0  # Close
    elif distance_in_frame > 0.125:
        estimated_meters = 2.0  # Medium
    else:
        estimated_meters = 3.0  # Far
        
    return estimated_meters


def draw_test_interface(frame, current_test, expected_gesture, detected_gesture, 
                       lighting, distance, additional_info=None):
    """
    Draw test information overlay on frame
    
    Args:
        frame: Input frame
        current_test: Description of current test
        expected_gesture: The gesture user should be performing
        detected_gesture: The gesture detected by system
        lighting: Current lighting level
        distance: Current distance
        additional_info: Any additional information to display
        
    Returns:
        Frame with test interface drawn
    """
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    # Draw top info bar
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    
    # Draw bottom info bar
    cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
    
    # Blend overlay with original frame
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    
    # Add test information text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Test: {current_test}", (20, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Expected gesture: {expected_gesture.upper()}", (20, 60), font, 0.7, (255, 255, 255), 2)
    
    # Add right-aligned info
    light_text = f"Lighting: {lighting:.1f} lux"
    light_size = cv2.getTextSize(light_text, font, 0.7, 2)[0]
    cv2.putText(frame, light_text, (w - light_size[0] - 20, 30), font, 0.7, (255, 255, 255), 2)
    
    dist_text = f"Distance: {distance:.1f} m"
    dist_size = cv2.getTextSize(dist_text, font, 0.7, 2)[0]
    cv2.putText(frame, dist_text, (w - dist_size[0] - 20, 60), font, 0.7, (255, 255, 255), 2)
    
    # Add detection result in bottom bar
    result_color = (0, 255, 0) if expected_gesture == detected_gesture else (0, 0, 255)
    cv2.putText(frame, f"Detected: {detected_gesture.upper()}", (20, h-60), font, 0.9, result_color, 2)
    
    # Add additional info if provided
    if additional_info:
        cv2.putText(frame, additional_info, (20, h-20), font, 0.7, (255, 255, 255), 2)
    
    # Add instructions
    instr_text = "Press 'n' for next test, 'q' to quit"
    cv2.putText(frame, instr_text, (w//2 - 150, h-20), font, 0.7, (255, 255, 255), 2)
    
    return frame
