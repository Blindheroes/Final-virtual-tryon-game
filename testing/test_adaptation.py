"""
Test Module: Distance-Based Threshold Adaptation
Compares hand gesture recognition with and without distance-based threshold adaptation
"""

import cv2
import numpy as np
import time
import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from ..hand_tracking import HandTracker
from test_utils import TestLogger, estimate_distance, calculate_lux, draw_test_interface


class HandTrackerNoAdaptation(HandTracker):
    """
    Modified HandTracker class that doesn't use distance-based threshold adaptation
    """
    
    def is_finger_extended(self, finger_name, threshold_modifier=1.0):
        """
        Override to ignore threshold_modifier parameter
        """
        # Call parent method but always use 1.0 as threshold_modifier
        return super().is_finger_extended(finger_name, 1.0)


def run_adaptation_test(args):
    """
    Run test comparing hand gesture recognition with and without distance adaptation
    
    Args:
        args: Command line arguments
    """
    # Initialize test logger
    logger = TestLogger("HandGesture_Adaptation")
    
    # Initialize both tracker types
    tracker_with_adaptation = HandTracker()
    tracker_without_adaptation = HandTrackerNoAdaptation()
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Try to set camera to HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Test parameters
    distances = [1.0, 2.0, 3.0]  # meters
    gestures = ["pointing", "selecting"]
    adaptation_modes = [True, False]  # With and without adaptation
    min_samples = args.samples  # Minimum samples per condition
    
    # Initialize test state
    current_distance_idx = 0
    current_gesture_idx = 0
    current_adaptation_idx = 0
    samples_collected = 0
    test_running = True
    
    # Create window
    cv2.namedWindow("Threshold Adaptation Test", cv2.WINDOW_NORMAL)
    
    print("\n===== DISTANCE THRESHOLD ADAPTATION TEST =====")
    print(f"Testing will collect {min_samples} samples for each combination of:")
    print(f"Distances: {distances} meters")
    print(f"Gestures: {gestures}")
    print(f"Adaptation modes: With Adaptation, Without Adaptation")
    print("\nFollow the on-screen instructions.")
    print("Press 'n' to manually advance to next test.")
    print("Press 'q' to quit at any time.\n")
    
    input("Press Enter to begin testing...")
    
    while test_running:
        # Get current test parameters
        current_distance = distances[current_distance_idx]
        current_gesture = gestures[current_gesture_idx]
        use_adaptation = adaptation_modes[current_adaptation_idx]
        
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
            
        # Process with both trackers
        tracker_with_adaptation.process_frame(frame)
        tracker_without_adaptation.process_frame(frame)
        
        # Choose active tracker based on current adaptation setting
        active_tracker = tracker_with_adaptation if use_adaptation else tracker_without_adaptation
        
        # Get gesture state from active tracker
        is_pointing = active_tracker.is_pointing()
        is_selecting = active_tracker.is_selecting()
        
        # Determine detected gesture
        detected_gesture = "none"
        if is_pointing and not is_selecting:
            detected_gesture = "pointing"
        elif is_selecting:
            detected_gesture = "selecting"
            
        # Get current conditions
        estimated_distance = estimate_distance(active_tracker.landmarks) or 0
        current_lux = calculate_lux(frame)
        
        # Check if distance is close to target
        distance_correct = abs(estimated_distance - current_distance) <= 0.5
        
        # Check if the correct gesture is being performed
        correct_detection = detected_gesture == current_gesture
        
        # If correct gesture and distance, record sample
        if correct_detection and distance_correct and not args.manual:
            samples_collected += 1
            
            # Log the result
            logger.log_result(
                params={
                    "distance": current_distance,
                    "gesture": current_gesture,
                    "adaptation": use_adaptation,
                    "lighting": current_lux,
                },
                metrics={
                    "accuracy": 1.0 if correct_detection else 0.0,
                    "estimated_distance": estimated_distance,
                }
            )
            
            # Print progress
            adaptation_status = "WITH" if use_adaptation else "WITHOUT"
            print(f"Sample {samples_collected}/{min_samples} for {current_gesture} at {current_distance}m {adaptation_status} adaptation")
        
        # Create distance instruction message
        if estimated_distance < current_distance - 0.5:
            distance_message = "Please move BACK"
        elif estimated_distance > current_distance + 0.5:
            distance_message = "Please move CLOSER"
        else:
            distance_message = "Distance is good!"
            
        # Determine which tracker to use for visualization
        display_tracker = active_tracker
            
        # Draw test interface with current state
        adaptation_status = "ON" if use_adaptation else "OFF"
        additional_info = f"{distance_message} | Samples: {samples_collected}/{min_samples}"
        
        frame = draw_test_interface(
            frame, 
            f"Adaptation Test - {current_gesture} at {current_distance}m (Adaptation {adaptation_status})",
            current_gesture,
            detected_gesture,
            current_lux,
            estimated_distance,
            additional_info
        )
        
        # Visualize hand tracker results
        frame = display_tracker.visualize_finger_states(frame)
        
        # Show the frame
        cv2.imshow("Threshold Adaptation Test", frame)
        
        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        
        # Record sample manually if in manual mode
        if args.manual and key == ord('s'):
            samples_collected += 1
            
            # Log the result
            logger.log_result(
                params={
                    "distance": current_distance,
                    "gesture": current_gesture,
                    "adaptation": use_adaptation,
                    "lighting": current_lux,
                },
                metrics={
                    "accuracy": 1.0 if correct_detection else 0.0,
                    "estimated_distance": estimated_distance,
                }
            )
            
            adaptation_status = "WITH" if use_adaptation else "WITHOUT"
            print(f"Sample {samples_collected}/{min_samples} for {current_gesture} at {current_distance}m {adaptation_status} adaptation")
            
        # Advance to next test condition
        if samples_collected >= min_samples or key == ord('n'):
            samples_collected = 0
            
            # Cycle through all combinations
            current_adaptation_idx = (current_adaptation_idx + 1) % len(adaptation_modes)
            
            # If we've gone through both adaptation modes, move to next gesture
            if current_adaptation_idx == 0:
                current_gesture_idx = (current_gesture_idx + 1) % len(gestures)
                
                # If we've gone through all gestures, move to next distance
                if current_gesture_idx == 0:
                    current_distance_idx = (current_distance_idx + 1) % len(distances)
                    
                    # If we've gone through all distances, end the test
                    if current_distance_idx == 0:
                        test_running = False
                    
        # Quit if q is pressed
        if key == ord('q'):
            test_running = False
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Save and visualize results
    csv_path = logger.save_results()
    report_path = logger.generate_report()
    
    print("\n===== TEST COMPLETED =====")
    print(f"Results saved to {csv_path}")
    print(f"Report saved to {report_path}")
    
    # Open report if requested
    if args.show_report:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            img = plt.imread(report_path)
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error showing report: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test threshold adaptation effectiveness")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--samples", type=int, default=10, help="Samples per test condition")
    parser.add_argument("--manual", action="store_true", help="Manually record samples with 's' key")
    parser.add_argument("--show-report", action="store_true", help="Show report after testing")
    
    args = parser.parse_args()
    run_adaptation_test(args)
