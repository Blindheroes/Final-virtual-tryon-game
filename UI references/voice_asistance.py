import cv2
import numpy as np


class VoiceAssistantOverlay:
    def __init__(self):
        # Settings
        self.width, self.height = 640, 480
        self.title_font_scale = 0.7
        self.text_font_scale = 0.45
        self.text_color = (255, 255, 255)  # White
        self.text_color_button = (0, 0, 0)  # Black text on buttons
        self.opacity = 0.6
        self.button_color = (255, 255, 255)  # White
        self.button_color_dark = (30, 30, 30)  # Dark for the talk button
        self.button_selected_color = (0, 255, 0)  # Green
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # States
        self.is_listening = False
        self.talk_button_clicked = False

        # UI elements - [x, y, width, height] - moved higher
        self.listening_box = [120, 60, 400, 50]
        self.talk_button = [120, 130, 400, 50]
        self.example_title_pos = (16, 200)
        self.example_commands = [
            [20, 220, 600, 40, "\"Find clothes for asian male \""],
            [20, 270, 600, 40, "\"Show me casual outfits\""],
            [20, 320, 600, 40, "\"What's trending for summer 2025?\""]
        ]

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

    def draw_character(self, img):
        # Draw a simple character face at the top
        center_x = self.width // 2
        # Head outline - moved up to ensure visibility
        cv2.circle(img, (center_x - 80, 90), 35, (0, 0, 0), -1)

        # Face features (eyes, mouth, etc.)
        # Simple eyes
        cv2.circle(img, (center_x - 95, 80), 5, (255, 255, 255), -1)
        cv2.circle(img, (center_x - 65, 80), 5, (255, 255, 255), -1)

        # Simple mouth (smile)
        cv2.ellipse(img, (center_x - 80, 100), (15, 8),
                    0, 0, 180, (255, 255, 255), 2)

        # Hair (simplified)
        points = np.array([
            [center_x - 115, 70],
            [center_x - 95, 55],
            [center_x - 65, 55],
            [center_x - 45, 70]
        ], np.int32)
        cv2.fillPoly(img, [points], (0, 0, 0))

        # Simple neck/collar
        cv2.rectangle(img, (center_x - 90, 125),
                      (center_x - 70, 140), (0, 0, 0), -1)
        cv2.ellipse(img, (center_x - 80, 140),
                    (25, 10), 0, 0, 180, (0, 0, 0), 2)

    def draw_listening_box(self, img):
        # Draw the listening status box
        # cv2.rectangle(img,
        #               (self.listening_box[0], self.listening_box[1]),
        #               (self.listening_box[0] + self.listening_box[2],
        #                self.listening_box[1] + self.listening_box[3]),
        #               (255, 255, 255), -1)

        # Draw text inside the box
        text = "Listening..." if self.is_listening else "Listening..."
        text_size = cv2.getTextSize(
            text, self.font, self.text_font_scale, 1)[0]
        text_x = self.listening_box[0] + \
            (self.listening_box[2] - text_size[0]) // 2
        text_y = self.listening_box[1] + \
            (self.listening_box[3] + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), self.font,
                    self.text_font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_example_commands(self, img):
        # Draw "Example Commands" title
        cv2.putText(img, "Example Commands", self.example_title_pos,
                    self.font, self.text_font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw example command boxes
        for x, y, w, h, text in self.example_commands:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          (200, 200, 200), 1)  # Border

            # Draw the command text
            text_size=cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x=x + 10
            text_y=y + (h + text_size[1]) // 2
            cv2.putText(img, text, (text_x, text_y), self.font,
                        self.text_font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    def handle_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if Talk button was clicked
            if (self.talk_button[0] <= x <= self.talk_button[0] + self.talk_button[2] and
                    self.talk_button[1] <= y <= self.talk_button[1] + self.talk_button[3]):
                self.talk_button_clicked=True
                self.is_listening=not self.is_listening
                print("Talk button clicked! Listening mode:", self.is_listening)

    def run(self):
        cv2.namedWindow('Voice Assistant')
        cv2.setMouseCallback('Voice Assistant', self.handle_click)

        while True:
            ret, frame=self.cap.read()

            if not ret:
                break

            # Create transparent overlay
            overlay=np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # overlay[:] = (240, 240, 240)  # Light gray background

            # Draw listening box
            if self.is_listening:
                self.draw_listening_box(overlay)

            # Draw talk button
            button_color=self.button_selected_color if self.is_listening else self.button_color_dark
            button_text_color=self.text_color_button if self.is_listening else self.text_color
            self.draw_round_button(
                overlay, self.talk_button, button_color, "Click to Talk", button_text_color)

            # Reset button clicked state after drawing
            if self.talk_button_clicked:
                self.talk_button_clicked=False

            # Draw example commands
            self.draw_example_commands(overlay)

            # Blend overlay with webcam frame
            result=cv2.addWeighted(
                frame, 1 - self.opacity, overlay, self.opacity, 0)

            cv2.imshow('Voice Assistant', result)

            if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app=VoiceAssistantOverlay()
    app.run()
