import cv2
import numpy as np


class VirtualTryonOverlay:
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

        # States for buttons
        self.button_states = {
            "exit": False,
            "voice": False,
            "rescan": False
        }

        # UI elements - [x, y, width, height]
        self.exit_button = [20, 20, 600, 50]
        self.voice_button = [20, 90, 300, 50]
        self.rescan_button = [320, 90, 300, 50]
        # Main black area for garment overlay
        self.garment_area = [0, 0, 640, 450]
        # Bottom recommendation bar
        self.recommendation_bar = [0, 450, 640, 50]

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

    def draw_garment_area(self, img):
        # Draw black rectangle for garment overlay area
        x, y, w, h = self.garment_area
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

        # Add title text in the garment area
        text = "2D Garment Overlay"
        text_size = cv2.getTextSize(
            text, self.font, self.title_font_scale, 1)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h // 2)
        cv2.putText(img, text, (text_x, text_y), self.font,
                    self.title_font_scale, self.text_color, 1, cv2.LINE_AA)

    def draw_recommendation_bar(self, img):
        # Draw white rectangle for recommendation bar
        x, y, w, h = self.recommendation_bar
        # Ensure bar is within frame height
        y_adjusted = min(y, self.height - h)
        cv2.rectangle(img, (x, y_adjusted),
                      (x + w, y_adjusted + h), (255, 255, 255), -1)

        # Add text to recommendation bar
        text = "Recommendation for ...."
        text_size = cv2.getTextSize(
            text, self.font, self.text_font_scale, 1)[0]
        text_x = x + 30
        text_y = y_adjusted + (h + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), self.font,
                    self.text_font_scale, (100, 100, 100), 1, cv2.LINE_AA)

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check exit button
            if (self.exit_button[0] <= x <= self.exit_button[0] + self.exit_button[2] and
                    self.exit_button[1] <= y <= self.exit_button[1] + self.exit_button[3]):
                self.button_states["exit"] = True
                print("Exit button clicked!")

            # Check voice assistant button
            elif (self.voice_button[0] <= x <= self.voice_button[0] + self.voice_button[2] and
                  self.voice_button[1] <= y <= self.voice_button[1] + self.voice_button[3]):
                self.button_states["voice"] = not self.button_states["voice"]
                print("Voice Assistant button clicked!")

            # Check rescan button
            elif (self.rescan_button[0] <= x <= self.rescan_button[0] + self.rescan_button[2] and
                  self.rescan_button[1] <= y <= self.rescan_button[1] + self.rescan_button[3]):
                self.button_states["rescan"] = not self.button_states["rescan"]
                print("Rescan Body button clicked!")

    def run(self):
        cv2.namedWindow('Virtual Try-on')
        cv2.setMouseCallback('Virtual Try-on', self.handle_click)

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Create transparent overlay
            overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Draw exit button (black)
            exit_color = self.button_selected_color if self.button_states["exit"] else (
                0,0,255)
            exit_text_color = self.text_color_button if self.button_states[
                "exit"] else self.text_color
            self.draw_round_button(
                overlay, self.exit_button, exit_color, "Exit", exit_text_color)

            # Draw voice assistant button
            voice_color = self.button_selected_color if self.button_states[
                "voice"] else self.button_color
            self.draw_round_button(
                overlay, self.voice_button, voice_color, "Use Voice Assistant", self.text_color_button)

            # Draw rescan body button
            rescan_color = self.button_selected_color if self.button_states[
                "rescan"] else self.button_color
            self.draw_round_button(
                overlay, self.rescan_button, rescan_color, "Rescan Body", self.text_color_button)

            # # Draw garment overlay area
            # self.draw_garment_area(overlay)

            # Draw recommendation bar (ensure it's visible)
            if self.recommendation_bar[1] < self.height:
                self.draw_recommendation_bar(overlay)

            # Blend overlay with webcam frame
            result = cv2.addWeighted(
                frame, 1 - self.opacity, overlay, self.opacity, 0)

            cv2.imshow('Virtual Try-on', result)

            # Exit on Esc or exit button
            if cv2.waitKey(1) & 0xFF == 27 or self.button_states["exit"]:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = VirtualTryonOverlay()
    app.run()
