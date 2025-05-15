import cv2
import numpy as np


class MainMenuOverlay:
    def __init__(self):
        # Settings
        self.width, self.height = 640, 480
        self.title_font_scale = 0.7
        self.text_font_scale = 0.45
        self.text_color = (255, 255, 255)  # White
        self.text_color_button = (0, 0, 0)  # Black text on buttons
        self.opacity = 0.6
        self.button_color = (255, 255, 255)  # White
        self.button_selected_color = (0, 255, 0)  # Green
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # States
        self.selected_gender = None  # None, "Male", or "Female"

        # UI elements - [x, y, width, height]

        self.male_button = [100, 120, 150, 50]
        self.female_button = [390, 120, 150, 50]
        self.scan_body_button = [120, 200, 400, 50]
        self.voice_assistant_button = [120, 300, 400, 50]
        self.camera_calibration_button = [120, 370, 400, 50]

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def draw_round_button(self, img, rect, color, text, text_color):
        x, y, w, h = rect
        radius = h // 2

        # Draw circle at left edge
        cv2.circle(img, (x + radius, y + radius), radius, color, -1)

        # Draw circle at right edge
        cv2.circle(img, (x + w - radius, y + radius), radius, color, -1)

        # Draw rectangle in the middle
        cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, -1)

        # Draw text
        text_size = cv2.getTextSize(
            text, self.font, self.text_font_scale, 1)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), self.font,
                    self.text_font_scale, text_color, 1, cv2.LINE_AA)

    def draw_divider(self, img):
        # Draw "or try" divider
        line_y = 270
        cv2.line(img, (150, line_y), (270, line_y), (150, 150, 150), 1)
        cv2.line(img, (370, line_y), (490, line_y), (150, 150, 150), 1)

        cv2.putText(img, "or try", (self.width // 2 - 20, line_y + 5),
                    self.font, self.text_font_scale, (150, 150, 150), 1, cv2.LINE_AA)

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check male button
            if (self.male_button[0] <= x <= self.male_button[0] + self.male_button[2] and
                    self.male_button[1] <= y <= self.male_button[1] + self.male_button[3]):
                self.selected_gender = "Male"

            # Check female button
            elif (self.female_button[0] <= x <= self.female_button[0] + self.female_button[2] and
                  self.female_button[1] <= y <= self.female_button[1] + self.female_button[3]):
                self.selected_gender = "Female"

            # Check scan body button
            elif (self.scan_body_button[0] <= x <= self.scan_body_button[0] + self.scan_body_button[2] and
                  self.scan_body_button[1] <= y <= self.scan_body_button[1] + self.scan_body_button[3]):
                print("Scan Body clicked")
                # Add functionality here

            # Check voice assistant button
            elif (self.voice_assistant_button[0] <= x <= self.voice_assistant_button[0] + self.voice_assistant_button[2] and
                  self.voice_assistant_button[1] <= y <= self.voice_assistant_button[1] + self.voice_assistant_button[3]):
                print("Voice Assistant clicked")
                # Add functionality here

            # Check camera calibration button
            elif (self.camera_calibration_button[0] <= x <= self.camera_calibration_button[0] + self.camera_calibration_button[2] and
                  self.camera_calibration_button[1] <= y <= self.camera_calibration_button[1] + self.camera_calibration_button[3]):
                print("Camera Calibration clicked")
                # Add functionality here

    def run(self):
        cv2.namedWindow('Main Menu')
        cv2.setMouseCallback('Main Menu', self.handle_click)

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Create transparent overlay
            overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Draw title
            cv2.putText(overlay, "Choose your gender", (self.width // 2 - 100, 100),
                        self.font, self.title_font_scale, self.text_color, 1, cv2.LINE_AA)

            # Draw gender buttons (male and female)
            male_color = self.button_selected_color if self.selected_gender == "Male" else self.button_color
            female_color = self.button_selected_color if self.selected_gender == "Female" else self.button_color

            self.draw_round_button(
                overlay, self.male_button, male_color, "Male", self.text_color_button)
            self.draw_round_button(
                overlay, self.female_button, female_color, "Female", self.text_color_button)

            # Draw scan body button (black background)
            self.draw_round_button(
                overlay, self.scan_body_button, self.button_color, "Scan Body", self.text_color_button)

            # Draw divider
            self.draw_divider(overlay)

            # Draw voice assistant and camera calibration buttons
            self.draw_round_button(overlay, self.voice_assistant_button, self.button_color,
                                   "Voice Assistant", self.text_color_button)
            self.draw_round_button(overlay, self.camera_calibration_button, self.button_color,
                                   "Camera Calibration", self.text_color_button)

            # Blend overlay with webcam frame
            result = cv2.addWeighted(frame, 1, overlay, self.opacity, 0)

            cv2.imshow('Main Menu', result)

            if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = MainMenuOverlay()
    app.run()
