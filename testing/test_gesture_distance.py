"""
Testing module for hand gesture recognition accuracy based on distance.
This module tests the accuracy of hand gesture detection at different distances.
"""

import cv2
import numpy as np
import time
import os
import csv
import argparse
from datetime import datetime
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import after adding the parent directory to the path
from hand_tracking import HandTracker

# Import after adding parent directory to path

# Import the hand tracking module after adding parent directory to path


def test_gesture_distance():
    """
    Test hand gesture recognition accuracy at different distances.

    Distances tested:
    - Close (1 meter)
    - Optimal (2 meters)
    - Far (3 meters)

    Gestures tested:
    - Pointing (index finger extended)
    - Selecting (index and pinky finger extended)
    """
    # Initialize hand tracker
    hand_tracker = HandTracker()

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
        output_dir, f"gesture_distance_test_{timestamp}.csv")

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Distance", "Gesture", "Trials", "Successes", "Accuracy"])

        # Test distances
        distances = ["Close (1m)", "Optimal (2m)", "Far (3m)"]
        gestures = ["Pointing", "Selecting"]

        for distance in distances:
            print(f"\n=== Testing at {distance} ===")

            for gesture in gestures:
                print(f"\nPerforming {gesture} gesture test")
                print(f"Please stand at {distance} from the camera")
                print("Press SPACE when ready to begin the test")
                print("Perform the gesture when prompted")

                # Wait for user to position themselves
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame")
                        break

                    # Flip frame horizontally
                    frame = cv2.flip(frame, 1)

                    # Draw instructions
                    cv2.putText(frame, f"Distance: {distance}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"Gesture: {gesture}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, "Press SPACE when ready", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Process frame for hand tracking
                    frame = hand_tracker.process_frame(frame)

                    # Display gesture if detected
                    pointing = hand_tracker.is_pointing()
                    selecting = hand_tracker.is_selecting()

                    if pointing:
                        cv2.putText(frame, "Detected: Pointing", (20, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if selecting:
                        cv2.putText(frame, "Detected: Selecting", (20, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    cv2.imshow("Gesture Distance Test", frame)

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

                        cv2.imshow("Gesture Distance Test", frame_copy)

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
                        frame = hand_tracker.process_frame(frame)

                        # Check for gesture
                        if gesture == "Pointing" and hand_tracker.is_pointing():
                            detected_count += 1
                        elif gesture == "Selecting" and hand_tracker.is_selecting():
                            detected_count += 1

                        # Display frame
                        cv2.putText(frame, f"Distance: {distance}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Gesture: {gesture}", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(frame, f"Trial: {trial+1}/{trials}", (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Display detection status
                        if (gesture == "Pointing" and hand_tracker.is_pointing()) or \
                           (gesture == "Selecting" and hand_tracker.is_selecting()):
                            cv2.putText(frame, "DETECTED", (20, 160),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        cv2.imshow("Gesture Distance Test", frame)
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
                print(f"\n{gesture} at {distance}: {accuracy:.2f}% accuracy")

                # Write results to CSV
                writer.writerow([distance, gesture, trials,
                                successes, f"{accuracy:.2f}%"])

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nTest results saved to: {csv_file}")


if __name__ == "__main__":
    test_gesture_distance()
