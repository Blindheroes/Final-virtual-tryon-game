import cv2
import numpy as np
import time


class WebcamHandGestureUI:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        # Get webcam dimensions
        _, frame = self.cap.read()
        self.height, self.width = frame.shape[:2]

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (240, 240, 240)
        self.DARK_GRAY = (136, 136, 136)
        self.DARKER_GRAY = (68, 68, 68)
        self.LIGHT_GRAY = (245, 245, 245)

        # UI state
        self.current_step = 1
        self.total_steps = 3
        self.steps = [
            "Extend hands in the frame",
            "Position whole body in frame",
            "Keep ~2m distance from camera"
        ]

        # Track time for animation
        self.start_time = time.time()

    def draw_rounded_rectangle(self, img, top_left, bottom_right, radius=10, color=(255, 255, 255), thickness=-1):
        """Draw a rounded rectangle"""
        # Draw the main rectangle
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Draw main rectangle
        cv2.rectangle(img, (x1 + radius, y1),
                      (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius),
                      (x2, y2 - radius), color, thickness)

        # Draw the four corner circles
        if thickness < 0:  # Filled rectangle
            cv2.circle(img, (x1 + radius, y1 + radius),
                       radius, color, thickness)
            cv2.circle(img, (x2 - radius, y1 + radius),
                       radius, color, thickness)
            cv2.circle(img, (x1 + radius, y2 - radius),
                       radius, color, thickness)
            cv2.circle(img, (x2 - radius, y2 - radius),
                       radius, color, thickness)
        else:
            cv2.circle(img, (x1 + radius, y1 + radius),
                       radius, color, thickness)
            cv2.circle(img, (x2 - radius, y1 + radius),
                       radius, color, thickness)
            cv2.circle(img, (x1 + radius, y2 - radius),
                       radius, color, thickness)
            cv2.circle(img, (x2 - radius, y2 - radius),
                       radius, color, thickness)

    def draw_continue_button(self, img):
        """Draw the continue setup button"""
        button_width = int(self.width * 0.85)
        button_height = 50
        button_x = (self.width - button_width) // 2
        button_y = 20

        # Draw button background
        self.draw_rounded_rectangle(
            img,
            (button_x, button_y),
            (button_x + button_width, button_y + button_height),
            radius=8,
            color=self.BLACK
        )

        # Draw button text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Continue Setup"
        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
        text_x = button_x + (button_width - text_size[0]) // 2 + 10
        text_y = button_y + (button_height + text_size[1]) // 2

        # Draw icon (simplified as a small white rectangle)
        icon_size = 16
        icon_x = text_x - icon_size - 10
        icon_y = button_y + (button_height - icon_size) // 2

        # Draw diagonal arrow icon
        cv2.line(img, (icon_x, icon_y + icon_size),
                 (icon_x + icon_size, icon_y), (255, 255, 255), 2)
        cv2.line(img, (icon_x + icon_size - 6, icon_y),
                 (icon_x + icon_size, icon_y), (255, 255, 255), 2)
        cv2.line(img, (icon_x + icon_size, icon_y),
                 (icon_x + icon_size, icon_y + 6), (255, 255, 255), 2)

        cv2.putText(img, text, (text_x, text_y), font, 0.6, self.WHITE, 2)

    def draw_preview_container(self, img):
        """Draw the preview container with hand icon and instructions"""
        container_width = int(self.width * 0.85)
        container_height = 220
        container_x = (self.width - container_width) // 2
        container_y = 90

        # Draw container background
        self.draw_rounded_rectangle(
            img,
            (container_x, container_y),
            (container_x + container_width, container_y + container_height),
            radius=10,
            color=self.GRAY
        )

        # Draw hand icon (simplified)
        icon_center_x = container_x + container_width // 2
        icon_center_y = container_y + 80
        icon_size = 50

        # Animate hand (simple pulsing effect)
        pulse = 1.0 + 0.1 * np.sin(2 * (time.time() - self.start_time))
        icon_size = int(icon_size * pulse)

        # Drawing a simplified hand icon
        # Palm
        cv2.circle(img, (icon_center_x, icon_center_y),
                   int(icon_size * 0.4), self.BLACK, 2)

        # Fingers
        for i in range(5):
            angle = np.pi * 0.7 - i * np.pi * 0.35
            finger_length = icon_size * 0.8
            end_x = int(icon_center_x + finger_length * np.cos(angle))
            end_y = int(icon_center_y - finger_length * np.sin(angle))
            cv2.line(img, (icon_center_x, icon_center_y),
                     (end_x, end_y), self.BLACK, 2)

        # Draw instruction title
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "Extend hands in the frame"
        title_size = cv2.getTextSize(title, font, 0.7, 2)[0]
        title_x = container_x + (container_width - title_size[0]) // 2
        title_y = container_y + 140
        cv2.putText(img, title, (title_x, title_y), font, 0.7, self.BLACK, 2)

        # Draw instruction detail
        detail = "Keep your hands visible within the frame"
        detail2 = "for hand gesture detection"
        detail_size = cv2.getTextSize(detail, font, 0.5, 1)[0]
        detail_x = container_x + (container_width - detail_size[0]) // 2
        detail_y = title_y + 25

        detail2_size = cv2.getTextSize(detail2, font, 0.5, 1)[0]
        detail2_x = container_x + (container_width - detail2_size[0]) // 2
        detail2_y = detail_y + 20

        cv2.putText(img, detail, (detail_x, detail_y),
                    font, 0.5, self.DARKER_GRAY, 1)
        cv2.putText(img, detail2, (detail2_x, detail2_y),
                    font, 0.5, self.DARKER_GRAY, 1)

    def draw_steps(self, img):
        """Draw the step indicators"""
        steps_width = int(self.width * 0.85)
        step_height = 50
        steps_x = (self.width - steps_width) // 2
        steps_y = 330

        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(self.total_steps):
            step_y = steps_y + i * step_height

            # Draw step circle
            circle_x = steps_x + 15
            circle_y = step_y + step_height // 2
            circle_radius = 15

            if i + 1 == self.current_step:
                cv2.circle(img, (circle_x, circle_y),
                           circle_radius, self.BLACK, -1)
            else:
                cv2.circle(img, (circle_x, circle_y),
                           circle_radius, self.DARK_GRAY, -1)

            # Draw step number
            number_size = cv2.getTextSize(str(i + 1), font, 0.5, 1)[0]
            number_x = circle_x - number_size[0] // 2
            number_y = circle_y + number_size[1] // 2
            cv2.putText(img, str(i + 1), (number_x, number_y),
                        font, 0.5, self.WHITE, 1)

            # Draw step text
            text_x = circle_x + 30
            text_y = circle_y + 5
            cv2.putText(img, self.steps[i], (text_x,
                        text_y), font, 0.5, self.BLACK, 1)

            # Draw checkmark for current step
            if i + 1 == self.current_step:
                checkmark_x = steps_x + steps_width - 30
                checkmark_y = circle_y

                # Draw simple checkmark
                cv2.line(img,
                         (checkmark_x - 5, checkmark_y),
                         (checkmark_x, checkmark_y + 5),
                         self.BLACK, 2)
                cv2.line(img,
                         (checkmark_x, checkmark_y + 5),
                         (checkmark_x + 10, checkmark_y - 5),
                         self.BLACK, 2)

            # Draw separator line
            if i < self.total_steps - 1:
                cv2.line(img,
                         (steps_x, step_y + step_height),
                         (steps_x + steps_width, step_y + step_height),
                         (220, 220, 220), 1)

    def draw_ui(self, frame):
        """Draw the complete UI overlay on the frame"""
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width,
                      self.height), (0, 0, 0), -1)

        # Create a container for UI elements
        container_width = int(min(self.width * 0.95, 400))
        container_height = int(self.height * 0.9)
        container_x = (self.width - container_width) // 2
        container_y = (self.height - container_height) // 2

        # Draw main container
        self.draw_rounded_rectangle(
            overlay,
            (container_x, container_y),
            (container_x + container_width, container_y + container_height),
            radius=15,
            color=self.WHITE
        )

        # Draw UI elements
        self.draw_continue_button(overlay)
        self.draw_preview_container(overlay)
        self.draw_steps(overlay)

        # Apply the overlay with transparency
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def run(self):
        """Main loop to capture and display webcam with UI overlay"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)

            # Apply UI overlay
            result = self.draw_ui(frame)

            # Display the result
            cv2.imshow('Hand Gesture Detection Setup', result)

            # Check for key press
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
            elif key == ord(' '):  # Space key to advance steps
                self.current_step = min(
                    self.current_step + 1, self.total_steps)
            elif key == ord('r'):  # 'r' key to reset steps
                self.current_step = 1

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        app = WebcamHandGestureUI()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
