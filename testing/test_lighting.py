"""
Test Module: Hand Gesture Recognition Accuracy by Lighting
Evaluates the accuracy of hand gesture recognition under different lighting conditions
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


def run_lighting_test(args):
    """
    Run hand gesture recognition test under different lighting conditions
    
    Args:
        args: Command line arguments
    """
    # Initialize test logger
    logger = TestLogger("HandGesture_Lighting")
    
    # Initialize HandTracker
    hand_tracker = HandTracker()
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Try to set camera to HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Test parameters
    lighting_conditions = [
        {"name": "Low", "target_lux": 50, "tolerance": 30},
        {"name": "Medium", "target_lux": 300, "tolerance": 100},
        {"name": "High", "target_lux": 600, "tolerance": 150}
    ]
    gestures = ["pointing", "selecting"]
    min_samples = args.samples  # Minimum samples per lighting/gesture combination
    optimal_distance = 2.0  # Test at optimal distance
    
    # Initialize test state
    current_lighting_idx = 0
    current_gesture_idx = 0
    samples_collected = 0
    test_running = True
    
    # Create window
    cv2.namedWindow("Hand Gesture Lighting Test", cv2.WINDOW_NORMAL)
    
    print("\n===== HAND GESTURE LIGHTING TEST =====")
    print(f"Testing will collect {min_samples} samples for each combination of:")
    print(f"Lighting conditions: Low (~50 lux), Medium (~300 lux), High (~600 lux)")
    print(f"Gestures: {gestures}")
    print("\nFollow the on-screen instructions to adjust lighting.")
    print("Press 'n' to manually advance to next test.")
    print("Press 'q' to quit at any time.\n")
    
    input("Press Enter to begin testing...")
    
    while test_running:
        # Get current test parameters
        current_lighting = lighting_conditions[current_lighting_idx]
        current_gesture = gestures[current_gesture_idx]
        
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
            
        # Process hand tracking
        processed_frame = hand_tracker.process_frame(frame)
        
        # Get gesture state
        is_pointing = hand_tracker.is_pointing()
        is_selecting = hand_tracker.is_selecting()
        
        # Determine detected gesture
        detected_gesture = "none"
        if is_pointing and not is_selecting:
            detected_gesture = "pointing"
        elif is_selecting:
            detected_gesture = "selecting"
            
        # Get current conditions
        estimated_distance = estimate_distance(hand_tracker.landmarks) or 0
        current_lux = calculate_lux(frame)
        
        # Check if lighting is within target range
        lighting_in_range = abs(current_lux - current_lighting["target_lux"]) <= current_lighting["tolerance"]
        
        # Check if the correct gesture is being performed
        correct_detection = detected_gesture == current_gesture
        
        # Check if distance is close to optimal
        distance_correct = abs(estimated_distance - optimal_distance) <= 0.5
        
        # If lighting, gesture, and distance are correct, record sample
        if correct_detection and lighting_in_range and distance_correct and not args.manual:
            samples_collected += 1
            
            # Log the result
            logger.log_result(
                params={
                    "lighting": current_lux,
                    "lighting_condition": current_lighting["name"],
                    "gesture": current_gesture,
                    "distance": estimated_distance,
                },
                metrics={
                    "accuracy": 1.0 if correct_detection else 0.0,
                }
            )
            
            # Print progress
            print(f"Sample {samples_collected}/{min_samples} for {current_gesture} in {current_lighting['name']} lighting")
        
        # Create lighting instruction message
        lighting_message = ""
        if current_lux < current_lighting["target_lux"] - current_lighting["tolerance"]:
            lighting_message = "Please INCREASE lighting"
        elif current_lux > current_lighting["target_lux"] + current_lighting["tolerance"]:
            lighting_message = "Please DECREASE lighting"
        else:
            lighting_message = "Lighting is good!"
            
        # Add distance message
        if estimated_distance < optimal_distance - 0.5:
            distance_message = "Please move BACK"
        elif estimated_distance > optimal_distance + 0.5:
            distance_message = "Please move CLOSER"
        else:
            distance_message = "Distance is good!"
            
        # Combine messages
        additional_info = f"{lighting_message} | {distance_message} | Samples: {samples_collected}/{min_samples}"
        
        # Draw test interface with current state
        frame = draw_test_interface(
            frame, 
            f"Lighting Test - {current_gesture} in {current_lighting['name']} light",
            current_gesture,
            detected_gesture,
            current_lux,
            estimated_distance,
            additional_info
        )
        
        # Visualize hand tracker results
        frame = hand_tracker.visualize_finger_states(frame)
        
        # Show the frame
        cv2.imshow("Hand Gesture Lighting Test", frame)
        
        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        
        # Record sample manually if in manual mode
        if args.manual and key == ord('s'):
            samples_collected += 1
            
            # Log the result
            logger.log_result(
                params={
                    "lighting": current_lux,
                    "lighting_condition": current_lighting["name"],
                    "gesture": current_gesture,
                    "distance": estimated_distance,
                },
                metrics={
                    "accuracy": 1.0 if correct_detection else 0.0,
                }
            )
            
            print(f"Sample {samples_collected}/{min_samples} for {current_gesture} in {current_lighting['name']} lighting")
            
        # Advance to next test condition
        if samples_collected >= min_samples or key == ord('n'):
            samples_collected = 0
            current_gesture_idx = (current_gesture_idx + 1) % len(gestures)
            
            # If we've gone through all gestures, move to next lighting condition
            if current_gesture_idx == 0:
                current_lighting_idx = (current_lighting_idx + 1) % len(lighting_conditions)
                
                # If we've gone through all lighting conditions, end the test
                if current_lighting_idx == 0:
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
    parser = argparse.ArgumentParser(description="Test hand gesture recognition by lighting")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--samples", type=int, default=10, help="Samples per test condition")
    parser.add_argument("--manual", action="store_true", help="Manually record samples with 's' key")
    parser.add_argument("--show-report", action="store_true", help="Show report after testing")
    
    args = parser.parse_args()
    run_lighting_test(args)
