"""
Test Module: Temporal Filtering Effectiveness
Compares hand gesture recognition with and without temporal filtering
"""

import cv2
import numpy as np
import time
import argparse
import os
import sys
from collections import deque
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from ..hand_tracking import HandTracker
from test_utils import TestLogger, estimate_distance, calculate_lux, draw_test_interface


class HandTrackerNoFiltering(HandTracker):
    """
    Modified HandTracker class that doesn't use temporal filtering
    """
    
    def _update_gesture_history(self, gesture_name, detected):
        """
        Override to disable history updates - just store current state
        """
        self.gesture_history[gesture_name] = deque(maxlen=5)
        self.gesture_history[gesture_name].append(1 if detected else 0)
    
    def _get_gesture_confidence(self, gesture_name):
        """
        Override to return binary result without confidence calculation
        """
        if not self.gesture_history[gesture_name]:
            return 0.0
        
        # Return 1.0 if the single stored value is 1, else 0.0
        return float(self.gesture_history[gesture_name][-1])


def run_filtering_test(args):
    """
    Run test comparing hand gesture recognition with and without temporal filtering
    
    Args:
        args: Command line arguments
    """
    # Initialize test logger
    logger = TestLogger("HandGesture_TemporalFiltering")
    
    # Initialize both tracker types
    tracker_with_filtering = HandTracker()
    tracker_without_filtering = HandTrackerNoFiltering()
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Try to set camera to HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Test parameters
    gestures = ["pointing", "selecting"]
    filtering_modes = [True, False]  # With and without filtering
    min_samples = args.samples  # Minimum samples per condition
    optimal_distance = 2.0
    
    # Initialize test state
    current_gesture_idx = 0
    current_filtering_idx = 0
    samples_collected = 0
    test_running = True
    
    # Additional metrics for stability evaluation
    stability_window = 30  # Number of frames to evaluate stability
    gesture_history = {
        True: {"pointing": [], "selecting": []},
        False: {"pointing": [], "selecting": []}
    }
    
    # Create window
    cv2.namedWindow("Temporal Filtering Test", cv2.WINDOW_NORMAL)
    
    print("\n===== TEMPORAL FILTERING TEST =====")
    print(f"Testing will collect {min_samples} samples for each combination of:")
    print(f"Gestures: {gestures}")
    print(f"Filtering modes: With Filtering, Without Filtering")
    print("\nFollow the on-screen instructions.")
    print("Press 'n' to manually advance to next test.")
    print("Press 'q' to quit at any time.\n")
    
    input("Press Enter to begin testing...")
    
    while test_running:
        # Get current test parameters
        current_gesture = gestures[current_gesture_idx]
        use_filtering = filtering_modes[current_filtering_idx]
        
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
            
        # Process with both trackers
        tracker_with_filtering.process_frame(frame)
        tracker_without_filtering.process_frame(frame)
        
        # Choose active tracker based on current filtering setting
        active_tracker = tracker_with_filtering if use_filtering else tracker_without_filtering
        
        # Get gesture states from both trackers for comparison
        with_filtering_pointing = tracker_with_filtering.is_pointing()
        with_filtering_selecting = tracker_with_filtering.is_selecting()
        without_filtering_pointing = tracker_without_filtering.is_pointing()
        without_filtering_selecting = tracker_without_filtering.is_selecting()
        
        # Determine detected gestures for both trackers
        with_filtering_gesture = "none"
        if with_filtering_pointing and not with_filtering_selecting:
            with_filtering_gesture = "pointing"
        elif with_filtering_selecting:
            with_filtering_gesture = "selecting"
            
        without_filtering_gesture = "none"
        if without_filtering_pointing and not without_filtering_selecting:
            without_filtering_gesture = "pointing"
        elif without_filtering_selecting:
            without_filtering_gesture = "selecting"
        
        # Get detected gesture from active tracker
        detected_gesture = with_filtering_gesture if use_filtering else without_filtering_gesture
        
        # Get current conditions
        estimated_distance = estimate_distance(active_tracker.landmarks) or 0
        current_lux = calculate_lux(frame)
        
        # Check if the correct gesture is being performed
        correct_detection = detected_gesture == current_gesture
        
        # Check if distance is close to optimal
        distance_correct = abs(estimated_distance - optimal_distance) <= 0.5
        
        # Store results for stability calculation
        with_filtered_result = 1 if with_filtering_gesture == current_gesture else 0
        without_filtered_result = 1 if without_filtering_gesture == current_gesture else 0
        
        gesture_history[True][current_gesture].append(with_filtered_result)
        gesture_history[False][current_gesture].append(without_filtered_result)
        
        # Trim history to stability window
        for mode in [True, False]:
            for g in gestures:
                if len(gesture_history[mode][g]) > stability_window:
                    gesture_history[mode][g] = gesture_history[mode][g][-stability_window:]
        
        # Calculate stability for current settings
        # Stability is measured as 1 - standard deviation of detection results
        if len(gesture_history[use_filtering][current_gesture]) > 5:
            stability = 1.0 - np.std(gesture_history[use_filtering][current_gesture])
        else:
            stability = 0.0
        
        # If correct gesture and distance, record sample
        if correct_detection and distance_correct and not args.manual:
            samples_collected += 1
            
            # Log the result
            logger.log_result(
                params={
                    "gesture": current_gesture,
                    "filtering": use_filtering,
                    "lighting": current_lux,
                    "distance": estimated_distance,
                },
                metrics={
                    "accuracy": 1.0 if correct_detection else 0.0,
                    "stability": stability,
                }
            )
            
            # Print progress
            filtering_status = "WITH" if use_filtering else "WITHOUT"
            print(f"Sample {samples_collected}/{min_samples} for {current_gesture} {filtering_status} filtering")
        
        # Create stability bar visual
        stability_bar_width = int(stability * 100)
        stability_bar = np.zeros((20, 100, 3), dtype=np.uint8)
        stability_bar[:, :stability_bar_width, :] = (0, 255, 0)
        stability_bar[:, stability_bar_width:, :] = (0, 0, 255)
        
        # Add stability metrics
        if len(gesture_history[True][current_gesture]) > 5:
            with_filtering_stability = 1.0 - np.std(gesture_history[True][current_gesture])
        else:
            with_filtering_stability = 0.0
            
        if len(gesture_history[False][current_gesture]) > 5:
            without_filtering_stability = 1.0 - np.std(gesture_history[False][current_gesture])
        else:
            without_filtering_stability = 0.0
            
        # Draw test interface with current state
        filtering_status = "ON" if use_filtering else "OFF"
        additional_info = f"Stability: {stability:.2f} | Samples: {samples_collected}/{min_samples}"
        
        frame = draw_test_interface(
            frame, 
            f"Filtering Test - {current_gesture} (Filtering {filtering_status})",
            current_gesture,
            detected_gesture,
            current_lux,
            estimated_distance,
            additional_info
        )
        
        # Add stability bar
        h, w = frame.shape[:2]
        bar_y = h - 130
        bar_x = 200
        frame[bar_y:bar_y+20, bar_x:bar_x+100] = stability_bar
        
        # Add comparison text
        cv2.putText(frame, f"With filtering stability: {with_filtering_stability:.2f}", 
                   (bar_x + 110, bar_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Without filtering stability: {without_filtering_stability:.2f}", 
                   (bar_x + 110, bar_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Visualize hand tracker results
        frame = active_tracker.visualize_finger_states(frame)
        
        # Show the frame
        cv2.imshow("Temporal Filtering Test", frame)
        
        # Check for user input
        key = cv2.waitKey(1) & 0xFF
        
        # Record sample manually if in manual mode
        if args.manual and key == ord('s'):
            samples_collected += 1
            
            # Log the result
            logger.log_result(
                params={
                    "gesture": current_gesture,
                    "filtering": use_filtering,
                    "lighting": current_lux,
                    "distance": estimated_distance,
                },
                metrics={
                    "accuracy": 1.0 if correct_detection else 0.0,
                    "stability": stability,
                }
            )
            
            filtering_status = "WITH" if use_filtering else "WITHOUT"
            print(f"Sample {samples_collected}/{min_samples} for {current_gesture} {filtering_status} filtering")
            
        # Advance to next test condition
        if samples_collected >= min_samples or key == ord('n'):
            samples_collected = 0
            
            # Cycle through all combinations
            current_filtering_idx = (current_filtering_idx + 1) % len(filtering_modes)
            
            # If we've gone through both filtering modes, move to next gesture
            if current_filtering_idx == 0:
                current_gesture_idx = (current_gesture_idx + 1) % len(gestures)
                
                # If we've gone through all gestures, end the test
                if current_gesture_idx == 0:
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
    parser = argparse.ArgumentParser(description="Test temporal filtering effectiveness")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--samples", type=int, default=10, help="Samples per test condition")
    parser.add_argument("--manual", action="store_true", help="Manually record samples with 's' key")
    parser.add_argument("--show-report", action="store_true", help="Show report after testing")
    
    args = parser.parse_args()
    run_filtering_test(args)
