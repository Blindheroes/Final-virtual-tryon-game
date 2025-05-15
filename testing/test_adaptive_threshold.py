"""
Testing module for comparing hand gesture recognition with and without distance-based threshold adaptation.
This tests how the adaptive thresholds improve gesture recognition at different distances.
"""

import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime
import sys
import copy

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import after adding the parent directory to the path
from hand_tracking import HandTracker

# Import after adding parent directory to path


def test_adaptive_threshold():
    """
    Test comparison between adaptive thresholds and fixed thresholds for hand gesture recognition.

    Tests performed:
    - With adaptive thresholds (based on hand distance)
    - Without adaptive thresholds (fixed value)

    At distances:
    - Close (1 meter)
    - Optimal (2 meters)
    - Far (3 meters)
    """
    # Initialize hand trackers - one with adaptive thresholds, one without
    adaptive_tracker = HandTracker()

    # Create a new tracker instance for fixed thresholds
    fixed_tracker = HandTracker()  # Create fresh instance instead of deepcopy

    # Override the methods that use distance adaptation
    fixed_tracker.is_finger_extended = lambda finger_name, threshold_modifier=1.0: \
        super(type(fixed_tracker), fixed_tracker).is_finger_extended(
            finger_name, 1.0)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create output directory
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create CSV file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(
        output_dir, f"adaptive_threshold_test_{timestamp}.csv")

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Distance", "Gesture", "Threshold Mode",
                        "Trials", "Successes", "Accuracy"])

        # Test distances
        distances = ["Close (1m)", "Optimal (2m)", "Far (3m)"]
        gestures = ["Pointing", "Selecting"]
        modes = ["Adaptive", "Fixed"]

        for distance in distances:
            print(f"\n=== Testing at {distance} ===")

            for gesture in gestures:
                print(f"\nPerforming {gesture} gesture test")
                print(f"Please stand at {distance} from the camera")

                for mode in modes:
                    print(f"\nTesting with {mode} thresholds")
                    print("Press SPACE when ready to begin the test")

                    # Select the appropriate tracker
                    tracker = adaptive_tracker if mode == "Adaptive" else fixed_tracker

                    # Wait for user to position themselves
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Failed to capture frame")
                            break

                        # Flip frame horizontally
                        frame = cv2.flip(frame, 1)

                        # Process frame for hand tracking
                        frame_copy = frame.copy()
                        processed_frame = tracker.process_frame(frame_copy)

                        # Draw instructions
                        cv2.putText(processed_frame, f"Distance: {distance}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"Gesture: {gesture}", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"Mode: {mode} Thresholds", (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(processed_frame, "Press SPACE when ready", (20, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Display gesture if detected
                        pointing = tracker.is_pointing()
                        selecting = tracker.is_selecting()

                        if pointing:
                            cv2.putText(processed_frame, "Detected: Pointing", (20, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        if selecting:
                            cv2.putText(processed_frame, "Detected: Selecting", (20, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        cv2.imshow("Adaptive Threshold Test", processed_frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        if key == 32:  # SPACE
                            break

                    # Begin test
                    trials = 10
                    successes = 0

                    for trial in range(trials):
                        print(f"\nTrial {trial+1}/{trials}")
                        # Countdown 5 seconds
                        print(f"Perform the {gesture} gesture now")
                        countdown_start = time.time()
                        countdown_duration = 5  # 5 seconds countdown

                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                print("Error: Failed to capture frame")
                                break

                            # Flip frame horizontally
                            frame = cv2.flip(frame, 1)

                            # Process frame for display
                            frame_copy = frame.copy()

                            # Get current countdown time
                            elapsed = time.time() - countdown_start
                            remaining = max(0, countdown_duration - elapsed)

                            # Display countdown
                            cv2.putText(frame_copy, f"Starting in: {int(remaining)+1}", (frame_copy.shape[1]//2-150, frame_copy.shape[0]//2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)

                            # Display instructions
                            cv2.putText(frame_copy, f"Perform the {gesture} gesture", (20, frame_copy.shape[0]-50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            cv2.imshow("Adaptive Threshold Test", frame_copy)

                            # Check if countdown is complete
                            if remaining <= 0:
                                break

                            # Process at approximately 30fps
                            cv2.waitKey(33)  # ~30fps

                        print("Now!")

                        # Record 30 frames (approx. 1 second) and check for gesture
                        detection_frames = 30
                        detected_count = 0

                        for _ in range(detection_frames):
                            ret, frame = cap.read()
                            if not ret:
                                print("Error: Failed to capture frame")
                                break

                            # Flip frame horizontally
                            frame = cv2.flip(frame, 1)

                            # Process frame for hand tracking
                            frame_copy = frame.copy()
                            processed_frame = tracker.process_frame(frame_copy)

                            # Check for gesture
                            if gesture == "Pointing" and tracker.is_pointing():
                                detected_count += 1
                            elif gesture == "Selecting" and tracker.is_selecting():
                                detected_count += 1

                            # Display frame
                            cv2.putText(processed_frame, f"Distance: {distance}", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(processed_frame, f"Gesture: {gesture}", (20, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(processed_frame, f"Mode: {mode} Thresholds", (20, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.putText(processed_frame, f"Trial: {trial+1}/{trials}", (20, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            # Display detection status
                            if (gesture == "Pointing" and tracker.is_pointing()) or \
                               (gesture == "Selecting" and tracker.is_selecting()):
                                cv2.putText(processed_frame, "DETECTED", (20, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            cv2.imshow("Adaptive Threshold Test",
                                       processed_frame)
                            cv2.waitKey(1)

                        # Consider trial successful if gesture was detected in at least 50% of frames
                        if detected_count >= detection_frames * 0.5:
                            successes += 1
                            print("Success!")
                        else:
                            print("Not detected")

                        # Wait between trials
                        time.sleep(1)

                    # Calculate accuracy
                    accuracy = (successes / trials) * 100
                    print(
                        f"\n{gesture} at {distance} with {mode} thresholds: {accuracy:.2f}% accuracy")

                    # Write results to CSV
                    writer.writerow(
                        [distance, gesture, mode, trials, successes, f"{accuracy:.2f}%"])

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nTest results saved to: {csv_file}")


if __name__ == "__main__":
    test_adaptive_threshold()
