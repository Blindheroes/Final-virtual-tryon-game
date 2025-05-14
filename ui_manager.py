"""
UI Manager module for Virtual Try-On Game
Handles UI elements, screen drawing, and button interactions
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import time


class UIManager:
    def __init__(self):
        """Initialize the UI Manager"""
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.ui_path = os.path.join(self.base_path, 'UI')
        self.font_path = os.path.join(
            self.base_path, 'fonts', 'static', 'Montserrat-SemiBold.ttf')

        # UI Settings according to requirements
        self.width = 640
        self.height = 480
        self.title_font_scale = 0.7
        self.text_font_scale = 0.45
        self.text_color = (255, 255, 255)  # White
        self.text_color_button = (0, 0, 0)  # Black text on buttons
        self.opacity = 0.6
        self.button_color = (255, 255, 255)  # White
        self.button_selected_color = (0, 255, 0)  # Green

        # Current step for calibration
        self.current_step = 0

        # Define button areas for different screens (x, y, width, height)
        self.buttons = {
            # Calibration screen
            "calibration_complete": (50, 50, 540, 50),

            # Main menu
            "body_scan": (120, 200, 400, 50),
            "voice_assistant": (120, 300, 400, 50),
            "exit": (120, 400, 400, 50),

            # Gender select
            "male": (100, 120, 150, 50),
            "female": (390, 120, 150, 50),
            "back": (120, 400, 400, 50),

            # Body scan screen
            "continue": (450, 50, 100, 50),
            "back_from_scan": (50, 50, 100, 50),

            # Voice assistant screen
            "continue_voice": (120, 400, 400, 50),
            "back_voice": (120, 460, 400, 50),

            # Virtual try-on screen
            "main_menu": (20, 20, 295, 50),
            "recalibrate": (325, 20, 295, 50),
            "top_select": (20, 380, 150, 40),
            "bottom_select": (180, 380, 150, 40)
        }

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_thickness = 1

        # Animation and timing
        self.start_time = time.time()
        self.scan_animation_timer = 0

        # Try to load custom fonts
        self.custom_font_available = False

        # For voice assistant screen
        self.is_listening = False

    def _check_custom_font_available(self):
        """Check if custom fonts are available through PIL"""
        try:
            # Check if Montserrat font is available
            if os.path.exists(self.font_path):
                return True
            else:
                print("Custom font not found at:", self.font_path)
                return False
        except Exception as e:
            print("Error checking custom font:", e)
            return False

    def _pil_to_cv2(self, pil_image):
        """Convert PIL Image to OpenCV format"""
        # Convert PIL Image to numpy array
        return np.array(pil_image)[:, :, ::-1]  # RGB to BGR

    def _put_text_with_custom_font(self, img, text, position, font_size, color=(255, 255, 255)):
        """Add text to image using custom font through PIL"""
        if not self.custom_font_available:
            # Fall back to OpenCV font
            cv2.putText(img, text, position, self.font,
                        self.title_font_scale, color, self.font_thickness, cv2.LINE_AA)
            return img

        try:
            # Convert image from OpenCV to PIL
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # Load custom font
            font = ImageFont.truetype(self.font_path, font_size)

            # Convert BGR to RGB for PIL
            color_rgb = (color[2], color[1], color[0])

            # Add text
            draw.text(position, text, font=font, fill=color_rgb)

            # Convert back to OpenCV format
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"Error using custom font: {e}")
            # Fall back to OpenCV font
            cv2.putText(img, text, position, self.font,
                        self.title_font_scale, color, self.font_thickness, cv2.LINE_AA)
            return img

    def is_within_button(self, x, y, button_name):
        """Check if coordinates are within a specific button area"""
        if button_name not in self.buttons:
            return False

        btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
        return (btn_x <= x <= btn_x + btn_w) and (btn_y <= y <= btn_y + btn_h)

    def _draw_button(self, frame, button_name, text=None, active=False):
        """Draw a button on the frame"""
        if button_name not in self.buttons:
            return frame

        h, w = frame.shape[:2]
        btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
        color = self.button_selected_color if active else self.button_color

        # Check if button is completely outside the frame
        # Only return if the button is 100% outside the visible area
        if btn_x >= w or btn_y >= h or btn_x + btn_w <= 0 or btn_y + btn_h <= 0:
            return frame  # Button is completely outside, skip drawing

        # Adjust button coordinates if it's partially outside the frame
        x_start = max(0, btn_x)
        y_start = max(0, btn_y)
        x_end = min(w, btn_x + btn_w)
        y_end = min(h, btn_y + btn_h)

        # Calculate actual visible button dimensions
        visible_width = x_end - x_start
        visible_height = y_end - y_start

        if visible_width <= 0 or visible_height <= 0:
            return frame  # No visible part of the button

        # Create a mask for rounded corners (for the original button size)
        mask = np.zeros((btn_h, btn_w), dtype=np.uint8)
        radius = 15  # Corner radius

        # Draw filled rectangle on mask
        cv2.rectangle(mask, (radius, 0), (btn_w - radius, btn_h), 255, -1)
        cv2.rectangle(mask, (0, radius), (btn_w, btn_h - radius), 255, -1)

        # Draw the four corner circles on mask
        cv2.circle(mask, (radius, radius), radius, 255, -1)
        cv2.circle(mask, (btn_w - radius, radius), radius, 255, -1)
        cv2.circle(mask, (radius, btn_h - radius), radius, 255, -1)
        cv2.circle(mask, (btn_w - radius, btn_h - radius), radius, 255, -1)

        # Crop the mask to match the visible portion of the button
        mask_x_offset = x_start - btn_x
        mask_y_offset = y_start - btn_y
        visible_mask = mask[mask_y_offset:mask_y_offset + visible_height,
                            mask_x_offset:mask_x_offset + visible_width]

        # Create inverse mask for the visible portion
        mask_inv = cv2.bitwise_not(visible_mask)

        # Extract region of interest from the original image
        roi = frame[y_start:y_end, x_start:x_end]

        # Create colored button image for the visible portion
        button = np.ones((visible_height, visible_width, 3), dtype=np.uint8)
        button[:] = color

        # Apply mask to button
        masked_button = cv2.bitwise_and(button, button, mask=visible_mask)

        # Get background from ROI using inverse mask
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Add button and background together
        dst = cv2.add(roi_bg, masked_button)

        # Copy result back to original image
        frame[y_start:y_end, x_start:x_end] = dst

        # Add text if provided
        if text:
            # Calculate original text position (center of full button)
            if self.custom_font_available:
                text_size = 14
                # Rough estimate of text width for centering
                text_width = len(text) * text_size * 0.6
                text_x = btn_x + (btn_w - text_width) // 2
                text_y = btn_y + (btn_h - text_size) // 2

                # Check if text is in the visible area
                if x_start <= text_x < x_end and y_start <= text_y < y_end:
                    frame = self._put_text_with_custom_font(
                        frame, text, (text_x, text_y), text_size, self.text_color_button)
            else:
                text_size = cv2.getTextSize(
                    text, self.font, self.text_font_scale, self.font_thickness)[0]
                text_x = btn_x + (btn_w - text_size[0]) // 2
                text_y = btn_y + (btn_h + text_size[1]) // 2

                # Check if text is in the visible area
                if x_start <= text_x < x_end and y_start <= text_y < y_end:
                    cv2.putText(frame, text, (text_x, text_y), self.font,
                                self.text_font_scale, self.text_color_button, self.font_thickness, cv2.LINE_AA)

        return frame

    def _create_ui_background(self, h, w):
        """Create a stylish UI background"""
        # Create dark gradient background
        background = np.zeros((h, w, 3), dtype=np.uint8)

        # Add subtle gradient
        for y in range(h):
            color_value = int(30 + (y / h) * 20)
            background[y, :] = (color_value, color_value, color_value)

        # Add header bar
        header_height = 80
        cv2.rectangle(background, (0, 0), (w, header_height),
                      self.button_color, cv2.FILLED)

        # Add decorative elements
        cv2.line(background, (0, header_height),
                 (w, header_height), self.button_selected_color, 3)

        # Add a subtle pattern
        pattern_size = 20
        pattern_opacity = 0.05
        for x in range(0, w, pattern_size):
            for y in range(header_height, h, pattern_size):
                if (x + y) % (pattern_size * 2) == 0:
                    cv2.rectangle(background, (x, y), (x+pattern_size//2, y+pattern_size//2),
                                  (255, 255, 255), cv2.FILLED)

        return background

    def draw_calibration_screen(self, frame, pointer_pos=None):
        """Draw the calibration screen"""
        # Define UI settings according to requirements
        h, w = frame.shape[:2]

        # Create transparent overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Add title
        cv2.putText(overlay, "CALIBRATION", (w//2 - 100, 30),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Draw calibration instructions
        descriptions = [
            "Keep your hands visible within the frame for hand gesture detection",
            "Keep ~2m distance from camera and ensure your full body is visible"
        ]

        y_offset = 240
        for line in descriptions:
            # Handle multi-line descriptions by splitting text
            desc_lines = [line[i:i+40] for i in range(0, len(line), 40)]
            for desc in desc_lines:
                desc_size = cv2.getTextSize(
                    desc, self.font, self.text_font_scale, 1)[0]
                desc_x = (w - desc_size[0]) // 2
                cv2.putText(overlay, desc, (desc_x, y_offset),
                            self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)
                y_offset += 30

        # Draw progress steps
        steps = [
            #     "Extend hands in the frame",
            #     "Position whole body in frame",
            #     "Keep ~2m distance from camera"
        ]

        # y_step = 350
        # for i, step_text in enumerate(steps):
        #     button_x = 40
        #     button_width = w - 80
        #     button_height = 50
        #     button_y = y_step + i * (button_height + 10)

        #     # Draw rounded rectangle for step
        #     button_radius = 10

        #     # Button color based on selection
        #     button_color = (0, 255, 0) if i == self.current_step else (
        #         255, 255, 255)

        #     # Draw rounded rectangle background
        #     # Top left corner
        #     cv2.circle(overlay, (button_x + button_radius, button_y + button_radius),
        #                button_radius, button_color, -1)
        #     # Top right corner
        #     cv2.circle(overlay, (button_x + button_width - button_radius, button_y + button_radius),
        #                button_radius, button_color, -1)
        #     # Bottom left corner
        #     cv2.circle(overlay, (button_x + button_radius, button_y + button_height - button_radius),
        #                button_radius, button_color, -1)
        #     # Bottom right corner
        #     cv2.circle(overlay, (button_x + button_width - button_radius, button_y + button_height - button_radius),
        #                button_radius, button_color, -1)
        #     # Rectangles to connect the circles
        #     cv2.rectangle(overlay, (button_x + button_radius, button_y),
        #                   (button_x + button_width - button_radius, button_y + button_height), button_color, -1)
        #     cv2.rectangle(overlay, (button_x, button_y + button_radius),
        #                   (button_x + button_width, button_y + button_height - button_radius), button_color, -1)

        #     # Add step text
        #     text_x = button_x + 20
        #     text_y = button_y + (button_height + 8) // 2
        #     cv2.putText(overlay, step_text, (text_x, text_y),
        #                 self.font, self.text_font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        # "Continue Setup" button at the top - rounded rectangle
        button_x = self.buttons["calibration_complete"][0]
        button_y = self.buttons["calibration_complete"][1]
        button_width = self.buttons["calibration_complete"][2]
        button_height = self.buttons["calibration_complete"][3]
        button_radius = 10

        # Check if pointer is over the button
        button_active = False
        if pointer_pos:
            x, y = pointer_pos
            button_active = self.is_within_button(x, y, "calibration_complete")

            # Draw a cursor at pointer position
            cv2.circle(overlay, pointer_pos, 10, (255, 255, 255), 2)
            cv2.circle(overlay, pointer_pos, 2, (255, 255, 255), -1)

        # Use white for button color
        button_color = (255, 255, 255)

        # Draw rounded rectangle for button
        # Top left corner
        cv2.circle(overlay, (button_x + button_radius, button_y + button_radius),
                   button_radius, button_color, -1)
        # Top right corner
        cv2.circle(overlay, (button_x + button_width - button_radius, button_y + button_radius),
                   button_radius, button_color, -1)
        # Bottom left corner
        cv2.circle(overlay, (button_x + button_radius, button_y + button_height - button_radius),
                   button_radius, button_color, -1)
        # Bottom right corner
        cv2.circle(overlay, (button_x + button_width - button_radius, button_y + button_height - button_radius),
                   button_radius, button_color, -1)
        # Rectangles to connect the circles
        cv2.rectangle(overlay, (button_x + button_radius, button_y),
                      (button_x + button_width - button_radius, button_y + button_height), button_color, -1)
        cv2.rectangle(overlay, (button_x, button_y + button_radius),
                      (button_x + button_width, button_y + button_height - button_radius), button_color, -1)
        # Blend overlay with webcam frame
        result = cv2.addWeighted(frame, 1.0, overlay, self.opacity, 0)

        # Add button text
        button_text = "Continue Setup"
        text_size = cv2.getTextSize(
            button_text, self.font, self.text_font_scale, 1)[0]
        text_x = button_x + (button_width - text_size[0]) // 2
        text_y = button_y + (button_height + text_size[1]) // 2
        cv2.putText(result, button_text, (text_x, text_y),
                    self.font, self.text_font_scale, self.text_color_button, self.font_thickness, cv2.LINE_AA)

        return result

    def draw_main_menu(self, frame, pointer_pos=None):
        """Draw the main menu screen"""
        h, w = frame.shape[:2]

        # Create transparent overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Add title
        cv2.putText(overlay, "VIRTUAL TRY-ON", (w//2 - 120, 30),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Add subtitle
        cv2.putText(overlay, "Choose your option", (w//2 - 120, 100),
                    self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        # Check which button is active
        body_scan_active = False
        voice_active = False
        exit_active = False

        if pointer_pos:
            x, y = pointer_pos
            body_scan_active = self.is_within_button(x, y, "body_scan")
            voice_active = self.is_within_button(x, y, "voice_assistant")
            exit_active = self.is_within_button(x, y, "exit")

            # Draw cursor at pointer position
            cv2.circle(overlay, pointer_pos, 10, self.text_color, 2)
            cv2.circle(overlay, pointer_pos, 2, self.text_color, -1)

        # Draw buttons using rounded rectangle style
        buttons = [
            ("body_scan", "Body Scan", body_scan_active),
            ("voice_assistant", "Voice Assistant", voice_active),
            ("exit", "Exit", exit_active)
        ]

        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            button_radius = 25  # Corner radius (half of button height)

            # Button color based on active state
            button_color = self.button_selected_color if active else self.button_color

            # Draw the rounded button
            # Left semicircle
            cv2.circle(overlay, (btn_x + button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Right semicircle
            cv2.circle(overlay, (btn_x + btn_w - button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Center rectangle
            cv2.rectangle(overlay, (btn_x + button_radius, btn_y),
                          (btn_x + btn_w - button_radius, btn_y + btn_h), button_color, -1)

            # Add button text
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            text_y = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(overlay, text, (text_x, text_y),
                        self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)

        # Blend overlay with webcam frame
        result = cv2.addWeighted(frame, 1, overlay, self.opacity, 0)

        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]

            # Add button text
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            text_y = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(result, text, (text_x, text_y),
                        self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)

        return result

    def draw_gender_select(self, frame, pointer_pos=None):
        """Draw the gender selection screen"""
        h, w = frame.shape[:2]

        # Create transparent overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Add title
        cv2.putText(overlay, "SELECT GENDER", (w//2 - 120, 30),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Add subtitle
        cv2.putText(overlay, "Choose your gender for accurate fit", (w//2 - 180, 80),
                    self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        # Check which button is active
        male_active = False
        female_active = False
        back_active = False

        if pointer_pos:
            x, y = pointer_pos
            male_active = self.is_within_button(x, y, "male")
            female_active = self.is_within_button(x, y, "female")
            back_active = self.is_within_button(x, y, "back")

            # Draw cursor at pointer position
            cv2.circle(overlay, pointer_pos, 10, self.text_color, 2)
            cv2.circle(overlay, pointer_pos, 2, self.text_color, -1)

        # Draw gender buttons (shapes only, not text)
        buttons = [
            ("male", "Male", male_active),
            ("female", "Female", female_active)
        ]

        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            button_radius = 25  # Corner radius (half of button height)

            # Button color based on active state
            button_color = self.button_selected_color if active else self.button_color

            # Draw the rounded button
            # Left semicircle
            cv2.circle(overlay, (btn_x + button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Right semicircle
            cv2.circle(overlay, (btn_x + btn_w - button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Center rectangle
            cv2.rectangle(overlay, (btn_x + button_radius, btn_y),
                          (btn_x + btn_w - button_radius, btn_y + btn_h), button_color, -1)

        # Draw back button (shape only, not text)
        btn_x, btn_y, btn_w, btn_h = self.buttons["back"]
        button_radius = 25  # Corner radius (half of button height)

        # Button color based on active state
        button_color = self.button_selected_color if back_active else self.button_color

        # Draw the rounded button
        # Left semicircle
        cv2.circle(overlay, (btn_x + button_radius, btn_y + btn_h//2),
                   button_radius, button_color, -1)
        # Right semicircle
        cv2.circle(overlay, (btn_x + btn_w - button_radius, btn_y + btn_h//2),
                   button_radius, button_color, -1)
        # Center rectangle
        cv2.rectangle(overlay, (btn_x + button_radius, btn_y),
                      (btn_x + btn_w - button_radius, btn_y + btn_h), button_color, -1)

        # Blend overlay with webcam frame
        result = cv2.addWeighted(frame, 1, overlay, self.opacity, 0)

        # Now add all text after blending
        # Add button text for gender buttons
        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            text_y = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(result, text, (text_x, text_y),
                        self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)

        # Add back button text
        btn_x, btn_y, btn_w, btn_h = self.buttons["back"]
        text = "Back"
        text_size = cv2.getTextSize(
            text, self.font, self.text_font_scale, 1)[0]
        text_x = btn_x + (btn_w - text_size[0]) // 2
        text_y = btn_y + (btn_h + text_size[1]) // 2
        cv2.putText(result, text, (text_x, text_y),
                    self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)

        return result

    def draw_body_scan(self, frame, pointer_pos=None, body_type=None):
        """Draw the body scan screen"""
        h, w = frame.shape[:2]

        # Create transparent overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Add title
        cv2.putText(overlay, "BODY SCAN", (w//2 - 80, 30),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Drawing the scanning area with a rectangle frame
        scan_x = 80
        scan_y = 120
        scan_w = 480
        scan_h = 280
        cv2.rectangle(overlay, (scan_x, scan_y), (scan_x+scan_w, scan_y+scan_h),
                      (100, 100, 100), 2)

        # Calculate animation time
        current_time = time.time()
        self.scan_animation_timer = (current_time - self.start_time) * 2

        # Show body type if available
        if body_type:
            # Display body type info
            text = f"Body Type: {body_type}"
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x = (w - text_size[0]) // 2
            cv2.rectangle(overlay, (text_x - 10, 80), (text_x + text_size[0] + 10, 110),
                          self.button_selected_color, -1)
        else:
            # Draw scanning line animation (horizontal line moving down)
            scan_y_pos = 120 + int((self.scan_animation_timer * 30) % 280)
            cv2.line(overlay, (80, scan_y_pos), (560, scan_y_pos),
                     (0, 255, 255), 2)

        # Check which button is active
        continue_active = False
        back_active = False

        if pointer_pos:
            x, y = pointer_pos
            continue_active = self.is_within_button(x, y, "continue")
            back_active = self.is_within_button(x, y, "back_from_scan")

        # Draw buttons
        buttons = [
            ("continue", "Continue", continue_active),
            ("back_from_scan", "Back", back_active)
        ]

        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            button_radius = 25  # Corner radius (half of button height)

            # Button color based on active state
            button_color = self.button_selected_color if active else self.button_color

            # Draw the rounded button
            # Left semicircle
            cv2.circle(overlay, (btn_x + button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Right semicircle
            cv2.circle(overlay, (btn_x + btn_w - button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Center rectangle
            cv2.rectangle(overlay, (btn_x + button_radius, btn_y),
                          (btn_x + btn_w - button_radius, btn_y + btn_h), button_color, -1)

        # Blend overlay with webcam frame
        result = cv2.addWeighted(frame, 1, overlay, self.opacity, 0)

        # Add text elements AFTER blending

        # Add title again (on the blended result)
        cv2.putText(result, "BODY SCAN", (w//2 - 80, 30),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Add instruction text
        instruction = "Stand ~2m away from camera, Ensure your full body is visible"

        # Handle multi-line text for instructions
        max_chars = 40
        lines = []
        for i in range(0, len(instruction), max_chars):
            lines.append(instruction[i:i+max_chars])

        y_offset = 420
        for line in lines:
            text_size = cv2.getTextSize(
                line, self.font, self.text_font_scale, 1)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(result, line, (text_x, y_offset),
                        self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)
            y_offset += 30

        # Show body type text if available
        if body_type:
            # Display body type info
            text = f"Body Type: {body_type}"
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(result, text, (text_x, 100),
                        self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)
        else:
            # Scanning animation text
            scan_time = int(self.scan_animation_timer) % 4
            scan_msg = "Scanning" + "." * (scan_time + 1)

            text_size = cv2.getTextSize(
                scan_msg, self.font, self.text_font_scale, 1)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(result, scan_msg, (text_x, 100),
                        self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        # Add button text after blending
        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            text_y = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(result, text, (text_x, text_y),
                        self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)

        # Draw cursor at pointer position (after blending)
        if pointer_pos:
            cv2.circle(result, pointer_pos, 10, self.text_color, 2)
            cv2.circle(result, pointer_pos, 2, self.text_color, -1)

        return result

    def draw_voice_assistant(self, frame, pointer_pos=None):
        """Draw the voice assistant screen"""
        h, w = frame.shape[:2]

        # Create transparent overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw talk button
        button_x = 120
        button_y = 130
        button_w = 400
        button_h = 50
        button_radius = 25  # Corner radius (half of button height)

        # Button color based on listening state
        button_color = self.button_selected_color if self.is_listening else self.button_color

        # Draw the rounded button
        # Left semicircle
        cv2.circle(overlay, (button_x + button_radius, button_y + button_h//2),
                   button_radius, button_color, -1)
        # Right semicircle
        cv2.circle(overlay, (button_x + button_w - button_radius, button_y + button_h//2),
                   button_radius, button_color, -1)
        # Center rectangle
        cv2.rectangle(overlay, (button_x + button_radius, button_y),
                      (button_x + button_w - button_radius, button_y + button_h), button_color, -1)

        # Add example commands as white rectangles with black text
        example_commands = [
            "\"Find clothes for asian male\"",
            "\"Show me casual outfits\"",
            "\"What's trending for summer 2025?\""
        ]

        y_pos = 220
        for cmd in example_commands:
            # Draw white rectangle background
            rect_x = 20
            rect_y = y_pos
            rect_w = 600
            rect_h = 40

            # Draw rounded rectangle for commands
            cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h),
                          self.button_color, -1)
            y_pos += 50

        # Check which button is active
        continue_active = False
        back_active = False

        if pointer_pos:
            x, y = pointer_pos
            continue_active = self.is_within_button(x, y, "continue_voice")
            # back_active = self.is_within_button(x, y, "back_voice")

        # Draw navigation buttons
        buttons = [
            ("continue_voice", "Continue", continue_active),
            # ("back_voice", "Back", back_active)
        ]

        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            button_radius = 25  # Corner radius (half of button height)

            # Button color based on active state
            button_color = self.button_selected_color if active else self.button_color

            # Draw the rounded button
            # Left semicircle
            cv2.circle(overlay, (btn_x + button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Right semicircle
            cv2.circle(overlay, (btn_x + btn_w - button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Center rectangle
            cv2.rectangle(overlay, (btn_x + button_radius, btn_y),
                          (btn_x + btn_w - button_radius, btn_y + btn_h), button_color, -1)

        # Blend overlay with webcam frame
        result = cv2.addWeighted(frame, 1, overlay, self.opacity, 0)

        # ---- Add all text after blending ----

        # Add title
        cv2.putText(result, "VOICE ASSISTANT", (w//2 - 120, 30),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Add listening status
        text = "Listening..." if self.is_listening else "Click to Talk"
        text_size = cv2.getTextSize(
            text, self.font, self.text_font_scale, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(result, text, (text_x, 100),
                    self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        # Add button text
        text = "Click to Talk"
        text_size = cv2.getTextSize(
            text, self.font, self.text_font_scale, 1)[0]
        text_x = button_x + (button_w - text_size[0]) // 2
        text_y = button_y + (button_h + text_size[1]) // 2
        cv2.putText(result, text, (text_x, text_y),
                    self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)

        # Add example commands title
        cv2.putText(result, "Example Commands:", (20, 200),
                    self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        # Add example command texts
        y_pos = 220
        for cmd in example_commands:
            rect_x = 20
            rect_y = y_pos
            rect_w = 600
            rect_h = 40

            # Add command text
            text_size = cv2.getTextSize(
                cmd, self.font, self.text_font_scale, 1)[0]
            text_x = rect_x + 10
            text_y = rect_y + (rect_h + text_size[1]) // 2
            cv2.putText(result, cmd, (text_x, text_y),
                        self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)
            y_pos += 50

        # Add navigation button texts
        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            text_y = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(result, text, (text_x, text_y),
                        self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)

        # Draw cursor at pointer position after blending
        if pointer_pos:
            cv2.circle(result, pointer_pos, 10, self.text_color, 2)
            cv2.circle(result, pointer_pos, 2, self.text_color, -1)

        return result

    def draw_virtual_tryon(self, frame, pointer_pos=None, active_clothing_type='top'):
        """Draw the virtual try-on screen"""
        h, w = frame.shape[:2]

        # Create transparent overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Add a recommendation bar at the bottom
        # cv2.rectangle(overlay, (0, 430), (w, 480), self.button_color, -1)

        # Check which button is active
        main_menu_active = False
        recalibrate_active = False
        exit_active = False

        if pointer_pos:
            x, y = pointer_pos
            main_menu_active = self.is_within_button(x, y, "main_menu")
            recalibrate_active = self.is_within_button(x, y, "recalibrate")
            exit_active = self.is_within_button(x, y, "exit")

        # Draw navigation buttons (shapes only)
        buttons = [
            ("main_menu", "Main Menu", main_menu_active),
            # ("recalibrate", "Rescan", recalibrate_active),
            # ("exit", "Exit", exit_active)
        ]

        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            button_radius = 25  # Corner radius (half of button height)

            # Button color based on active state
            button_color = self.button_selected_color if active else self.button_color

            # Draw the rounded button
            # Left semicircle
            cv2.circle(overlay, (btn_x + button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Right semicircle
            cv2.circle(overlay, (btn_x + btn_w - button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Center rectangle
            cv2.rectangle(overlay, (btn_x + button_radius, btn_y),
                          (btn_x + btn_w - button_radius, btn_y + btn_h), button_color, -1)

        # Blend overlay with webcam frame
        result = cv2.addWeighted(frame, 1, overlay, self.opacity, 0)

        # ----- Add all text after blending -----

        # Add title
        cv2.putText(result, "VIRTUAL TRY-ON", (w//2 - 120, 300),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Add button texts
        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            text_y = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(result, text, (text_x, text_y),
                        self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)

        # Add label above clothing buttons
        # cv2.putText(result, "Select clothing type:", (20, 360),
        #             self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        # Add recommendation text (now after blending)
        # cv2.putText(result, "Use hand gestures to cycle clothing", (20, 460),
        #             self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)

        # Draw cursor at pointer position after blending
        if pointer_pos:
            cv2.circle(result, pointer_pos, 10, self.text_color, 2)
            cv2.circle(result, pointer_pos, 2, self.text_color, -1)

        return result
