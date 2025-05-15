import cv2
import numpy as np


class ScanScreenOverlay:
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

        # UI elements - [x, y, width, height]
        self.capture_button = [80, 40, 480, 50]
        self.scan_frame = [80, 120, 480, 320]  # Rectangle for the scan area
        self.button_clicked = False

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

    def draw_instruction_text(self, img):
        # Draw the two instruction lines at the bottom
        cv2.putText(img, "Stand ~2m away from camera, Ensure your full body is visible", (80, 460),
                    self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check capture button
            if (self.capture_button[0] <= x <= self.capture_button[0] + self.capture_button[2] and
                    self.capture_button[1] <= y <= self.capture_button[1] + self.capture_button[3]):
                self.button_clicked = True
                print("Capture button clicked!")
                # Here you would add functionality to capture the image

    def run(self):
        cv2.namedWindow('Scan Screen')
        cv2.setMouseCallback('Scan Screen', self.handle_click)

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Create transparent overlay
            overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Fill background with semi-transparent black
            cv2.rectangle(overlay, (0, 0), (self.width,
                          self.height), (0, 0, 0), -1)

            # Draw capture button at the top
            button_color = self.button_selected_color if self.button_clicked else (
                255, 255, 255)  # Black button as in the image
            self.draw_round_button(
                overlay, self.capture_button, button_color, "Capture", (0, 0, 0))

            # Reset button clicked state after drawing
            if self.button_clicked:
                self.button_clicked = False

            # Draw scan frame (rectangle outline)
            cv2.rectangle(overlay,
                          (self.scan_frame[0], self.scan_frame[1]),
                          (self.scan_frame[0] + self.scan_frame[2],
                           self.scan_frame[1] + self.scan_frame[3]),
                          (100, 100, 100), -1)

            # Draw the instruction text
            self.draw_instruction_text(overlay)

            # Blend overlay with webcam frame - keep the scan area more visible
            result = frame.copy()

            # Apply overlay with specified opacity
            cv2.addWeighted(overlay, self.opacity, result,
                            1 - self.opacity, 0, result)

            cv2.imshow('Scan Screen', result)

            if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = ScanScreenOverlay()
    app.run()
