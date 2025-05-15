import cv2
import numpy as np

# Constants
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
TITLE_FONT_SIZE = 0.7
TEXT_FONT_SIZE = 0.45
TEXT_COLOR = (255, 255, 255)  # White
OPACITY = 0.6
SELECTED_BUTTON_COLOR = (0, 255, 0)  # Black
BUTTON_COLOR = (255, 255, 255)  # White
BUTTON_TEXT_COLOR = (0,0,0)  
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Create a class for the calibration overlay


class CalibrationOverlay:
    def __init__(self):
        self.steps = [
            "Show hands in the frame",
            "Position whole body in frame",
        ]
        self.descriptions = [
            "Keep your hands visible within the frame for hand gesture detection",
            "Keep ~2m distance from camera and ensure your full body is visible",
        ]
        self.current_step = 0
        self.setup_completed = False

    def create_overlay(self):
        # Create a transparent overlay
        overlay = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

        # Create a semi-transparent gray background
        if not self.setup_completed:
            # safe zone area
            cv2.rectangle(overlay, (10, 10),
                          (WINDOW_WIDTH-10, WINDOW_HEIGHT-10), (50, 50, 50), 10)

            # Draw the current instruction title
            # title = self.steps[self.current_step]
            # title_size = cv2.getTextSize(title, FONT, TITLE_FONT_SIZE, 2)[0]
            # title_x = (WINDOW_WIDTH - title_size[0]) // 2
            # cv2.putText(overlay, title, (title_x, 280),
            #             FONT, TITLE_FONT_SIZE, TEXT_COLOR, 2)

            # Draw the description if available
            if self.descriptions[self.current_step]:
                desc = self.descriptions[self.current_step]
                # Handle multi-line descriptions by splitting text
                desc_lines = [desc[i:i+40] for i in range(0, len(desc), 40)]
                y_offset = 240
                for line in desc_lines:
                    desc_size = cv2.getTextSize(
                        line, FONT, TEXT_FONT_SIZE, 1)[0]
                    desc_x = (WINDOW_WIDTH - desc_size[0]) // 2
                    cv2.putText(overlay, line, (desc_x, y_offset),
                                FONT, TEXT_FONT_SIZE, TEXT_COLOR, 1)
                    y_offset += 30

            # Draw progress steps at the bottom
            y_step = 350
            for i, step_text in enumerate(self.steps):
                button_x = 40
                button_width = WINDOW_WIDTH - 80
                button_height = 20
                button_y = y_step + i * (button_height + 10)

                # Draw step number circle
                circle_radius = 15
                circle_x = button_x + circle_radius + 5
                circle_y = button_y + button_height // 2

                if i == self.current_step:
                    # Selected step - white circle with black text
                    cv2.circle(overlay, (circle_x, circle_y),
                               circle_radius, SELECTED_BUTTON_COLOR, -1)
                    cv2.putText(overlay, str(
                        i+1), (circle_x - 5, circle_y + 5), FONT, TEXT_FONT_SIZE, BUTTON_COLOR, 1)

                else:
                    # Non-selected step - black circle with white text
                    cv2.circle(overlay, (circle_x, circle_y),
                               circle_radius, BUTTON_COLOR, -1)
                    cv2.putText(overlay, str(i+1), (circle_x - 5, circle_y + 5),
                                FONT, TEXT_FONT_SIZE, SELECTED_BUTTON_COLOR, 1)

                # Draw step text
                text_x = circle_x + circle_radius + 15
                text_y = circle_y + 5
                cv2.putText(overlay, step_text, (text_x, text_y),
                            FONT, TEXT_FONT_SIZE, TEXT_COLOR, 1)

            # "Continue Setup" button at the top - rounded rectangle
            button_x = 50
            button_y = 50
            button_width = WINDOW_WIDTH - 100
            button_height = 50
            button_radius = 15

            # Draw rounded rectangle for button
            # Top left corner
            cv2.circle(overlay, (button_x + button_radius, button_y +
                       button_radius), button_radius, BUTTON_COLOR, -1)
            # Top right corner
            cv2.circle(overlay, (button_x + button_width - button_radius,
                       button_y + button_radius), button_radius, BUTTON_COLOR, -1)
            # Bottom left corner
            cv2.circle(overlay, (button_x + button_radius, button_y +
                       button_height - button_radius), button_radius, BUTTON_COLOR, -1)
            # Bottom right corner
            cv2.circle(overlay, (button_x + button_width - button_radius, button_y +
                       button_height - button_radius), button_radius, BUTTON_COLOR, -1)
            # Rectangles to connect the circles
            cv2.rectangle(overlay, (button_x + button_radius, button_y), (button_x +
                          button_width - button_radius, button_y + button_height), BUTTON_COLOR, -1)
            cv2.rectangle(overlay, (button_x, button_y + button_radius), (button_x +
                          button_width, button_y + button_height - button_radius), BUTTON_COLOR, -1)

            # Add button text
            button_text = "Continue Setup"
            text_size = cv2.getTextSize(
                button_text, FONT, TEXT_FONT_SIZE, 1)[0]
            text_x = button_x + (button_width - text_size[0]) // 2
            text_y = button_y + (button_height + text_size[1]) // 2
            cv2.putText(overlay, button_text, (text_x, text_y),
                        FONT, TEXT_FONT_SIZE, BUTTON_TEXT_COLOR, 1)

        return overlay

    def next_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
        else:
            self.setup_completed = True

    def restart(self):
        self.current_step = 0
        self.setup_completed = False


def main():
    # Create a capture object for the webcam
    cap = cv2.VideoCapture(0)

    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

    # Create calibration overlay
    calibration = CalibrationOverlay()

    while True:
        # Capture frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture image from camera.")
            break

        # Create the overlay
        overlay = calibration.create_overlay()

        # Combine the frame and overlay with opacity
        result = cv2.addWeighted(frame, 1.0, overlay, OPACITY, 0)

        # Display the result
        cv2.imshow('Webcam Calibration', result)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') or key == 13:  # Space or Enter
            calibration.next_step()
        elif key == ord('r'):
            calibration.restart()

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
