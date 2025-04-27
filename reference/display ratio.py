import cv2
import numpy as np


def main():
    # Open the default camera (usually webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # flip the camera feed horizontally for a mirror effect

    # Get the original dimensions
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate dimensions for 9:16 aspect ratio
    # We'll maintain the original height and calculate the new width
    new_width = int(original_height * (9/16))

    # For display purposes, we might need to scale down if the resulting image is too large
    display_scale = 1.0
    if original_height > 600:  # Lower threshold to fit both views
        display_scale = 600 / original_height

    # Display dimensions for the original and vertical frames
    orig_display_width = int(original_width * display_scale)
    orig_display_height = int(original_height * display_scale)
    vert_display_width = int(new_width * display_scale)
    vert_display_height = int(original_height * display_scale)

    # Calculate total width for side-by-side display
    # Add a small gap (10 pixels) between the frames
    total_width = orig_display_width + vert_display_width + 10

    print(f"Original dimensions: {original_width}x{original_height}")
    print(f"9:16 dimensions: {new_width}x{original_height}")
    print(f"Display scale: {display_scale:.2f}")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # flip the camera feed horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        # Calculate center point for cropping
        center_x = original_width // 2

        # Crop from the center to achieve 9:16 aspect ratio
        x_start = max(0, center_x - new_width // 2)
        x_end = min(original_width, center_x + new_width // 2)

        # In case the calculated crop region doesn't precisely match our desired width
        # (this can happen due to integer division), we adjust it
        if x_end - x_start != new_width:
            x_end = min(original_width, x_start + new_width)
            # If still not matching and we have room to adjust the start
            if x_end - x_start != new_width and x_start > 0:
                x_start = max(0, x_end - new_width)

        # Crop the frame to 9:16 aspect ratio
        vertical_frame = frame[:, x_start:x_end]

        # Draw crop region on original frame for visualization
        frame_with_overlay = frame.copy()
        cv2.rectangle(frame_with_overlay, (x_start, 0),
                      (x_end, original_height), (0, 255, 0), 2)

        # Resize both frames for display
        orig_display = cv2.resize(
            frame_with_overlay, (orig_display_width, orig_display_height))
        vert_display = cv2.resize(
            vertical_frame, (vert_display_width, vert_display_height))

        # Create a canvas for side-by-side display
        comparison = np.zeros(
            (orig_display_height, total_width, 3), dtype=np.uint8)

        # Place the original frame on the left
        comparison[:, 0:orig_display_width] = orig_display

        # Place the vertical frame on the right (with a small gap)
        comparison[:, orig_display_width+10:] = vert_display

        # Add labels
        cv2.putText(comparison, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "9:16 Crop", (orig_display_width+20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display both frames side by side
        cv2.imshow('Original vs 9:16 Vertical Comparison', comparison)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
