"""
Testing module for evaluating the effectiveness of temporal filtering in hand gesture recognition.
This tests how temporal filtering improves stability and reduces false positives/negatives.
"""

from hand_tracking import HandTracker
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

# Import after adding parent directory to path

# Now import hand tracking after adding parent directory to path

# Import the hand tracking module directly using absolute import


def test_temporal_filtering():
    """
    Test comparison between hand gesture recognition with and without temporal filtering.

    Tests performed:
    - With temporal filtering (using gesture history)
    - Without temporal filtering (single frame detection only)

    At conditions:
    - Static posture (holding gesture steady)
    - Dynamic movement (moving hand while maintaining gesture)
    - Transitional (changing between gestures)
    """
    # Initialize hand trackers - one with temporal filtering, one without
    # Create a new tracker instance for unfiltered tracking
    filtered_tracker = HandTracker()
    unfiltered_tracker = HandTracker()  # Create fresh instance instead of deepcopy

    # Copy relevant attributes from filtered tracker
    unfiltered_tracker.gesture_history = filtered_tracker.gesture_history.copy()

    # Store original update history method
    original_update_history = unfiltered_tracker._update_gesture_history

    def no_temporal_filtering(gesture_name, detected):
        # Still update the history for tracking purposes
        original_update_history(gesture_name, detected)
        # But the confidence will directly reflect the current frame detection
        unfiltered_tracker.gesture_history[gesture_name] = [
            1.0 if detected else 0.0]

    unfiltered_tracker._update_gesture_history = no_temporal_filtering

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
        output_dir, f"temporal_filtering_test_{timestamp}.csv")

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Type", "Gesture", "Filter Mode",
                        "Stability Score", "False Positives", "False Negatives"])

        # Test conditions
        conditions = ["Static", "Dynamic", "Transitional"]
        gestures = ["Pointing", "Selecting"]
        modes = ["Filtered", "Unfiltered"]

        for condition in conditions:
            print(f"\n=== Testing under {condition} condition ===")

            for gesture in gestures:
                print(f"\nPerforming {gesture} gesture test")
                print(f"Condition: {condition}")
                print("Stand at optimal distance (2m) with good lighting")

                for mode in modes:
                    print(f"\nTesting with {mode} mode")
                    print("Press SPACE when ready to begin the test")

                    # Select the appropriate tracker
                    tracker = filtered_tracker if mode == "Filtered" else unfiltered_tracker

                    # Wait for user to get ready
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Failed to capture frame")
                            break

                        # Flip frame horizontally
                        frame = cv2.flip(frame, 1)

                        # Process both trackers for display
                        frame_filtered = frame.copy()
                        frame_unfiltered = frame.copy()

                        filtered_tracker.process_frame(frame_filtered)
                        unfiltered_tracker.process_frame(frame_unfiltered)

                        # Create side-by-side display
                        h, w = frame.shape[:2]
                        display = np.zeros((h, w*2, 3), dtype=np.uint8)
                        display[:, :w] = frame_filtered
                        display[:, w:] = frame_unfiltered

                        # Add labels
                        cv2.putText(display, "Filtered", (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(display, "Unfiltered", (w + 20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Add test info
                        cv2.putText(display, f"Condition: {condition}", (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display, f"Gesture: {gesture}", (20, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display, f"Testing: {mode}", (20, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(display, "Press SPACE when ready", (20, 190),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Show detection status - filtered
                        pointing_filtered = filtered_tracker.is_pointing()
                        selecting_filtered = filtered_tracker.is_selecting()

                        if pointing_filtered:
                            cv2.putText(display, "Pointing", (20, 230),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        if selecting_filtered:
                            cv2.putText(display, "Selecting", (20, 270),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        # Show detection status - unfiltered
                        pointing_unfiltered = unfiltered_tracker.is_pointing()
                        selecting_unfiltered = unfiltered_tracker.is_selecting()

                        if pointing_unfiltered:
                            cv2.putText(display, "Pointing", (w + 20, 230),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        if selecting_unfiltered:
                            cv2.putText(display, "Selecting", (w + 20, 270),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        cv2.imshow("Temporal Filtering Test", display)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        if key == 32:  # SPACE
                            break

                    # Begin test
                    print("\nStarting test...")
                    print(f"Instructions for {condition} test:")

                    if condition == "Static":
                        print("Hold the gesture steady for 5 seconds")
                        instruction = f"Hold {gesture} gesture steady"
                    elif condition == "Dynamic":
                        print(
                            "Maintain the gesture while moving your hand around for 5 seconds")
                        instruction = f"Move hand while maintaining {gesture}"
                    else:  # Transitional
                        print(
                            "Alternate between different gestures while occasionally making the test gesture")
                        # Countdown 5 seconds
                        instruction = f"Transition between gestures, occasionally make {gesture}"
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
                        cv2.putText(frame_copy, instruction, (20, frame_copy.shape[0]-50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        cv2.imshow("Temporal Filtering Test", frame_copy)

                        # Check if countdown is complete
                        if remaining <= 0:
                            break

                        # Process at approximately 30fps
                        cv2.waitKey(33)  # ~30fps

                    print("Begin!")

                    # Record for 5 seconds (150 frames at ~30fps)
                    test_frames = 150
                    stability_count = 0
                    false_positives = 0
                    false_negatives = 0

                    # For static/dynamic tests, we expect continuous detection
                    # For transitional tests, we'll ask the user to indicate when they're making the gesture
                    is_true_gesture = False
                    last_true_state = False

                    for frame_idx in range(test_frames):
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
                        is_detected = False
                        if gesture == "Pointing" and tracker.is_pointing():
                            is_detected = True
                        elif gesture == "Selecting" and tracker.is_selecting():
                            is_detected = True

                        # Update metrics based on condition
                        if condition == "Static" or condition == "Dynamic":
                            # In static/dynamic mode, we expect continuous detection
                            is_true_gesture = True

                            # Count stability (consecutive frames with consistent detection)
                            if frame_idx > 0 and last_true_state == is_detected:
                                stability_count += 1

                            # Count false negatives (when gesture should be detected but isn't)
                            if not is_detected:
                                false_negatives += 1
                        else:  # Transitional
                            # In transitional mode, we'll toggle the true state with spacebar
                            # Display prompt to indicate when making the gesture
                            cv2.putText(processed_frame, "Press SPACE when making the gesture", (20, 400),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                            # Check for spacebar press to toggle gesture state
                            key = cv2.waitKey(1) & 0xFF
                            if key == 32:  # SPACE
                                is_true_gesture = not is_true_gesture

                            # Count stability (consecutive frames with consistent detection)
                            if frame_idx > 0 and last_true_state == is_detected:
                                stability_count += 1

                            # Count false positives and negatives
                            if is_detected and not is_true_gesture:
                                false_positives += 1
                            elif not is_detected and is_true_gesture:
                                false_negatives += 1

                        # Update last state
                        last_true_state = is_detected

                        # Display frame
                        cv2.putText(processed_frame, f"Condition: {condition}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"Gesture: {gesture}", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"Mode: {mode}", (20, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(processed_frame, instruction, (20, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(processed_frame, f"Frame: {frame_idx}/{test_frames}", (20, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Display true gesture state for transitional test
                        if condition == "Transitional":
                            state_text = "MAKING GESTURE" if is_true_gesture else "NOT MAKING GESTURE"
                            state_color = (
                                0, 255, 0) if is_true_gesture else (0, 0, 255)
                            cv2.putText(processed_frame, state_text, (20, 240),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)                        # Display detection status
                        detection_text = "DETECTED" if is_detected else "NOT DETECTED"
                        detection_color = (
                            0, 255, 0) if is_detected else (0, 0, 255)
                        cv2.putText(processed_frame, detection_text, (20, 280),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, detection_color, 2)
                        cv2.imshow("Temporal Filtering Test", processed_frame)

                        # Process at approximately 30fps - use waitKey instead of sleep for proper window updating
                        key = cv2.waitKey(33) & 0xFF  # ~30fps
                        if key == ord('q'):
                            # Allow quitting during the test
                            cap.release()
                            cv2.destroyAllWindows()
                            return

                    # Calculate stability percentage (consecutive frames with same detection state)
                    stability_score = (stability_count /
                                       (test_frames - 1)) * 100

                    # Normalize false positives/negatives for transitional tests
                    if condition == "Transitional":
                        # We're measuring rates, not absolute counts
                        false_positive_rate = false_positives / test_frames * 100
                        false_negative_rate = false_negatives / test_frames * 100
                    else:
                        # For static/dynamic, we expect continuous detection
                        false_positive_rate = 0  # No false positives expected
                        false_negative_rate = (
                            false_negatives / test_frames) * 100

                    print(
                        f"\nResults for {gesture} in {condition} condition with {mode} mode:")
                    print(f"Stability: {stability_score:.2f}%")
                    print(f"False Positive Rate: {false_positive_rate:.2f}%")
                    print(f"False Negative Rate: {false_negative_rate:.2f}%")

                    # Write results to CSV
                    writer.writerow([condition, gesture, mode, f"{stability_score:.2f}%",
                                    f"{false_positive_rate:.2f}%", f"{false_negative_rate:.2f}%"])

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nTest results saved to: {csv_file}")


if __name__ == "__main__":
    test_temporal_filtering()
