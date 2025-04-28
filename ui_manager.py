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

        # UI Colors - Monochrome theme
        self.primary_color = (220, 220, 220)     # Light gray
        self.secondary_color = (150, 150, 150)   # Medium gray
        self.text_color = (255, 255, 255)        # White
        self.text_highlight_color = (255, 255, 255)  # White
        self.highlight_color = (200, 200, 200)   # Highlighted light gray
        self.background_color = (30, 30, 30)     # Dark gray
        self.accent_color = (100, 100, 100)      # Accent gray

        # Define button areas for different screens (x, y, width, height)
        self.buttons = {
            # Calibration screen
            "calibration_complete": (30, 300, 200, 25),

            # Main menu
            "body_scan": (20, 300, 280, 80),
            "voice_assistant": (20, 400, 280, 80),
            "exit": (20, 500, 280, 80),

            # Gender select
            "male": (175, 400, 200, 80),
            "female": (405, 400, 200, 80),
            "back": (290, 550, 200, 80),

            # Body scan screen
            "continue": (290, 550, 200, 80),

            # Voice assistant screen

            # Virtual try-on screen
            "main_menu": (175, 650, 200, 80),
            "recalibrate": (405, 650, 200, 80)
        }

        # Font settings for OpenCV fallback
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 1.0
        self.font_color = (255, 255, 255)
        self.font_thickness = 2
        self.title_font_size = 20
        self.text_font_size = 12
        self.text_left_margin_divider = 8

        # Animation and timing
        self.start_time = time.time()
        self.scan_animation_timer = 0

        # Try to load custom fonts
        self.custom_font_available = self._check_custom_font_available()

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
                        self.font_scale, color, self.font_thickness, cv2.LINE_AA)
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
                        self.font_scale, color, self.font_thickness, cv2.LINE_AA)
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
        color = self.highlight_color if active else self.secondary_color

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
                        frame, text, (text_x, text_y), text_size, self.text_color)
            else:
                text_size = cv2.getTextSize(
                    text, self.font, self.font_scale, self.font_thickness)[0]
                text_x = btn_x + (btn_w - text_size[0]) // 2
                text_y = btn_y + (btn_h + text_size[1]) // 2

                # Check if text is in the visible area
                if x_start <= text_x < x_end and y_start <= text_y < y_end:
                    cv2.putText(frame, text, (text_x, text_y), self.font,
                                self.font_scale, self.font_color, self.font_thickness, cv2.LINE_AA)

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
                      self.secondary_color, cv2.FILLED)

        # Add decorative elements
        cv2.line(background, (0, header_height),
                 (w, header_height), self.accent_color, 3)

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
        h, w = frame.shape[:2]

        # Create UI background
        ui_frame = self._create_ui_background(h, w)

        # Add camera feed with transparency
        alpha = 0.7
        ui_frame = cv2.addWeighted(ui_frame, 1-alpha, frame, alpha, 0)

        # Add title
        if self.custom_font_available:
            ui_frame = self._put_text_with_custom_font(
                ui_frame, "CALIBRATION", (w//4, 30), self.title_font_size)
        else:
            cv2.putText(ui_frame, "CALIBRATION", (w//4, 50), self.font,
                        1.5, self.font_color, self.font_thickness, cv2.LINE_AA)

        # Draw calibration instructions
        instructions = [
            "1. Stand in front of the camera",
            "2. Move your hand to verify tracking",
            "3. Practice gestures:"
        ]

        y_pos = 150
        for line in instructions:
            if self.custom_font_available:
                ui_frame = self._put_text_with_custom_font(
                    ui_frame, line, (w//self.text_left_margin_divider, y_pos), self.text_font_size)
            else:
                cv2.putText(ui_frame, line, (100, y_pos), self.font,
                            self.font_scale, self.font_color, self.font_thickness, cv2.LINE_AA)
            y_pos += 12

        # Add gesture instructions
        gesture_instructions = [
            "- Point with index finger",
            "- Select with index + pinky finger"
        ]

        y_pos += 20
        for line in gesture_instructions:
            if self.custom_font_available:
                ui_frame = self._put_text_with_custom_font(
                    ui_frame, line, (w//self.text_left_margin_divider, y_pos), self.text_font_size)
            else:
                cv2.putText(ui_frame, line, (150, y_pos), self.font,
                            self.font_scale-0.2, self.font_color, self.font_thickness-1, cv2.LINE_AA)
            y_pos += 12

        # Draw progress steps
        steps = [
            "Extend hands in the frame",
            "Position whole body in frame",
            "Keep ~2m distance from camera"
        ]

        y_pos = 250
        for i, step in enumerate(steps):
            # Draw step indicator
            cv2.circle(ui_frame, (w//self.text_left_margin_divider-10, y_pos), self.text_font_size - 8,
                       self.secondary_color, cv2.FILLED)

            if self.custom_font_available:
                ui_frame = self._put_text_with_custom_font(
                    ui_frame, str(i+1), (w//self.text_left_margin_divider, y_pos+5), self.text_font_size)
                ui_frame = self._put_text_with_custom_font(
                    ui_frame, step, (w//self.text_left_margin_divider, y_pos+5), self.text_font_size)
            else:
                cv2.putText(ui_frame, str(i+1), (75, y_pos+8),
                            self.font, 0.8, self.font_color, 2)
                cv2.putText(ui_frame, step, (110, y_pos+8),
                            self.font, 0.7, self.font_color, 1)
            y_pos += 12

        # Check if pointer is over the button
        button_active = False
        if pointer_pos:
            x, y = pointer_pos
            button_active = self.is_within_button(x, y, "calibration_complete")

            # Draw a cursor
            cv2.circle(ui_frame, pointer_pos, 2,
                       (255, 255, 255), -1)  # White center dot
            if button_active:
                # Draw a larger cursor for active button (red)
                cv2.circle(ui_frame, pointer_pos, 10, (0, 0, 255), 2)
            else:
                cv2.circle(ui_frame, pointer_pos, 10,
                           (255, 255, 255), 2)  # White circle

        # Draw calibration complete button
        self._draw_button(ui_frame, "calibration_complete",
                          "Continue Setup", button_active)

        return ui_frame

    def draw_main_menu(self, frame, pointer_pos=None):
        """Draw the main menu screen"""
        h, w = frame.shape[:2]

        # Create UI background
        ui_frame = self._create_ui_background(h, w)

        # Add camera feed with transparency
        alpha = 0.5
        ui_frame = cv2.addWeighted(ui_frame, 1-alpha, frame, alpha, 0)

        # Add title
        if self.custom_font_available:
            ui_frame = self._put_text_with_custom_font(
                ui_frame, "VIRTUAL TRY-ON", (w//2-170, 30), 40)
            ui_frame = self._put_text_with_custom_font(
                ui_frame, "Choose an option", (w//2-150, 130), 30)
        else:
            cv2.putText(ui_frame, "VIRTUAL TRY-ON", (w//2-170, 50), self.font,
                        1.5, self.font_color, self.font_thickness, cv2.LINE_AA)
            cv2.putText(ui_frame, "Choose an option", (w//2-150, 150), self.font,
                        1.2, self.font_color, self.font_thickness, cv2.LINE_AA)

        # Check which button is active
        body_scan_active = False
        voice_active = False
        exit_active = False

        if pointer_pos:
            x, y = pointer_pos
            body_scan_active = self.is_within_button(x, y, "body_scan")
            voice_active = self.is_within_button(x, y, "voice_assistant")
            exit_active = self.is_within_button(x, y, "exit")

            # Draw a cursor with white color
            cv2.circle(ui_frame, pointer_pos, 10, (255, 255, 255), 2)
            cv2.circle(ui_frame, pointer_pos, 2, (255, 255, 255), -1)

        # Draw buttons
        self._draw_button(ui_frame, "body_scan", "Body Scan", body_scan_active)
        self._draw_button(ui_frame, "voice_assistant",
                          "Voice Assistant", voice_active)
        self._draw_button(ui_frame, "exit", "Exit", exit_active)

        # Add some decorative icons
        # Body scan icon
        icon_x = 200
        icon_y = 330
        cv2.rectangle(ui_frame, (icon_x, icon_y),
                      (icon_x+30, icon_y+60), (255, 255, 255), 1)
        cv2.line(ui_frame, (icon_x, icon_y+30),
                 (icon_x+30, icon_y+30), (255, 255, 255), 1)

        # Voice assistant icon
        icon_x = 200
        icon_y = 430
        cv2.circle(ui_frame, (icon_x+15, icon_y+15), 15, (255, 255, 255), 1)
        cv2.rectangle(ui_frame, (icon_x+13, icon_y+15),
                      (icon_x+17, icon_y+40), (255, 255, 255), cv2.FILLED)

        return ui_frame

    def draw_gender_select(self, frame, pointer_pos=None):
        """Draw the gender selection screen"""
        h, w = frame.shape[:2]

        # Create UI background
        ui_frame = self._create_ui_background(h, w)

        # Add camera feed with transparency
        alpha = 0.5
        ui_frame = cv2.addWeighted(ui_frame, 1-alpha, frame, alpha, 0)

        # Add title
        if self.custom_font_available:
            ui_frame = self._put_text_with_custom_font(
                ui_frame, "SELECT GENDER", (w//2-150, 30), 40)
            ui_frame = self._put_text_with_custom_font(
                ui_frame, "Choose your gender for accurate fit", (w//2-220, 130), 28)
        else:
            cv2.putText(ui_frame, "SELECT GENDER", (w//2-150, 50), self.font,
                        1.5, self.font_color, self.font_thickness, cv2.LINE_AA)
            cv2.putText(ui_frame, "Choose your gender for accurate fit",
                        (w//2-220, 150), self.font, 0.9, self.font_color, self.font_thickness, cv2.LINE_AA)

        # Check which button is active
        male_active = False
        female_active = False
        back_active = False

        if pointer_pos:
            x, y = pointer_pos
            male_active = self.is_within_button(x, y, "male")
            female_active = self.is_within_button(x, y, "female")
            back_active = self.is_within_button(x, y, "back")

            # Draw a cursor
            cv2.circle(ui_frame, pointer_pos, 2,
                       (255, 255, 255), -1)  # White center dot
            if male_active or female_active or back_active:
                # Draw a larger cursor for active button (red)
                cv2.circle(ui_frame, pointer_pos, 10, (0, 0, 255), 2)
            else:
                cv2.circle(ui_frame, pointer_pos, 10,
                           (255, 255, 255), 2)  # White circle

        # Draw buttons
        self._draw_button(ui_frame, "male", "Male", male_active)
        self._draw_button(ui_frame, "female", "Female", female_active)
        self._draw_button(ui_frame, "back", "Back", back_active)

        # Add gender icons
        # Male icon
        cv2.circle(ui_frame, (275, 330), 30, (255, 255, 255), 2)
        # Arrow pointing up-right
        cv2.line(ui_frame, (290, 315), (310, 295), (255, 255, 255), 2)

        # Female icon
        cv2.circle(ui_frame, (505, 330), 30, (255, 255, 255), 2)
        # Cross at bottom
        cv2.line(ui_frame, (505, 360), (505, 380), (255, 255, 255), 2)
        cv2.line(ui_frame, (490, 370), (520, 370), (255, 255, 255), 2)

        return ui_frame

    def draw_body_scan(self, frame, pointer_pos=None, body_type=None):
        """Draw the body scan screen"""
        h, w = frame.shape[:2]

        # Create UI background
        ui_frame = self._create_ui_background(h, w)

        # Add camera feed with higher transparency
        alpha = 0.8  # More camera visibility for body scan
        ui_frame = cv2.addWeighted(ui_frame, 1-alpha, frame, alpha, 0)

        # Add title
        if self.custom_font_available:
            ui_frame = self._put_text_with_custom_font(
                ui_frame, "BODY SCAN", (w//2-120, 30), 40)
        else:
            cv2.putText(ui_frame, "BODY SCAN", (w//2-120, 50), self.font,
                        1.5, self.font_color, self.font_thickness, cv2.LINE_AA)

        # Add scanning overlay
        # Draw T-pose silhouette guide (simplified)
        center_x = w // 2
        cv2.line(ui_frame, (center_x, 150), (center_x, 400),
                 (0, 255, 255), 1, cv2.LINE_AA)  # Body line
        cv2.line(ui_frame, (center_x - 100, 200), (center_x +
                 100, 200), (0, 255, 255), 1, cv2.LINE_AA)  # Arms

        # Calculate animation time
        current_time = time.time()
        self.scan_animation_timer = (current_time - self.start_time) * 2

        # Show body type if available
        if body_type:
            # Highlight box
            cv2.rectangle(ui_frame, (center_x-140, 120),
                          (center_x+140, 180), (0, 100, 0), cv2.FILLED)
            cv2.rectangle(ui_frame, (center_x-140, 120),
                          (center_x+140, 180), (0, 255, 0), 2)

            if self.custom_font_available:
                ui_frame = self._put_text_with_custom_font(
                    ui_frame, f"Body Type: {body_type}", (center_x-120, 150), 30)
                ui_frame = self._put_text_with_custom_font(
                    ui_frame, "Stand in T-pose for measurements", (center_x-220, 210), 24)
            else:
                cv2.putText(ui_frame, f"Body Type: {body_type}",
                            (center_x-120, 170), self.font, 0.9, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
                cv2.putText(ui_frame, "Stand in T-pose for accurate measurements",
                            (center_x-240, 230), self.font, 0.7, self.font_color, 1, cv2.LINE_AA)
        else:
            # Scanning animation
            scan_time = int(self.scan_animation_timer) % 4
            scan_msg = "Scanning" + "." * (scan_time + 1)

            if self.custom_font_available:
                ui_frame = self._put_text_with_custom_font(
                    ui_frame, scan_msg, (center_x-80, 150), 32)
                ui_frame = self._put_text_with_custom_font(
                    ui_frame, "Stand in T-pose for accurate measurements", (center_x-240, 200), 22)
            else:
                cv2.putText(ui_frame, scan_msg,
                            (center_x-80, 150), self.font, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(ui_frame, "Stand in T-pose for accurate measurements",
                            (center_x-240, 200), self.font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw scanning line animation
            scan_y = 150 + int((self.scan_animation_timer * 50) % 300)
            cv2.line(ui_frame, (center_x-150, scan_y),
                     (center_x+150, scan_y), (0, 255, 255), 2)

        # Check which button is active
        continue_active = False
        back_active = False

        if pointer_pos:
            x, y = pointer_pos
            continue_active = self.is_within_button(x, y, "continue")
            back_active = self.is_within_button(x, y, "back")

            # Draw a cursor
            cv2.circle(ui_frame, pointer_pos, 2,
                       (255, 255, 255), -1)  # White center dot
            if continue_active or back_active:
                # Draw a larger cursor for active button (red)
                cv2.circle(ui_frame, pointer_pos, 10, (0, 0, 255), 2)
            else:
                cv2.circle(ui_frame, pointer_pos, 10,
                           (255, 255, 255), 2)  # White circle

        # Draw buttons
        self._draw_button(ui_frame, "continue", "Continue", continue_active)
        self._draw_button(ui_frame, "back", "Back", back_active)

        return ui_frame

    def draw_voice_assistant(self, frame, pointer_pos=None):
        """Draw the voice assistant screen"""
        h, w = frame.shape[:2]

        # Create UI background
        ui_frame = self._create_ui_background(h, w)

        # Add camera feed with transparency
        alpha = 0.6
        ui_frame = cv2.addWeighted(ui_frame, 1-alpha, frame, alpha, 0)

        # Add title
        if self.custom_font_available:
            ui_frame = self._put_text_with_custom_font(
                ui_frame, "VOICE ASSISTANT", (w//2-170, 30), 38)
        else:
            cv2.putText(ui_frame, "VOICE ASSISTANT", (w//2-170, 50), self.font,
                        1.5, self.font_color, self.font_thickness, cv2.LINE_AA)

        # Add voice visualization
        center_x = w // 2
        # Voice animation based on time
        current_time = time.time()
        animation_factor = (np.sin(current_time * 6) + 1) / \
            2  # 0 to 1 animation factor
        radius = 40 + int(15 * animation_factor)  # Pulsating circle

        # Draw voice button with animation
        cv2.circle(ui_frame, (center_x, 200), radius,
                   self.secondary_color, cv2.FILLED)
        cv2.circle(ui_frame, (center_x, 200), radius, (255, 255, 255), 2)

        # Add mic icon
        cv2.rectangle(ui_frame, (center_x-10, 180),
                      (center_x+10, 220), (255, 255, 255), cv2.FILLED)
        cv2.circle(ui_frame, (center_x, 180), 15, (255, 255, 255), cv2.FILLED)

        # Add voice assistant prompts
        prompts = [
            "What type of clothing are you looking for?",
            "What is your preferred style?",
            "Any specific color preferences?"
        ]

        y_pos = 300
        for prompt in prompts:
            if self.custom_font_available:
                ui_frame = self._put_text_with_custom_font(
                    ui_frame, prompt, (100, y_pos), 26)
            else:
                cv2.putText(ui_frame, prompt, (100, y_pos), self.font,
                            0.8, self.font_color, self.font_thickness, cv2.LINE_AA)
            y_pos += 50

        # Voice assistant functionality would typically go here
        # For now, just add a placeholder message
        if self.custom_font_available:
            ui_frame = self._put_text_with_custom_font(
                ui_frame, "Voice assistant feature in development", (100, 480), 24, (0, 255, 255))
        else:
            cv2.putText(ui_frame, "Voice assistant feature in development",
                        (100, 480), self.font, 0.8, (0, 255, 255), self.font_thickness, cv2.LINE_AA)

        # Check which button is active
        continue_active = False
        back_active = False

        if pointer_pos:
            x, y = pointer_pos
            continue_active = self.is_within_button(x, y, "continue")
            back_active = self.is_within_button(x, y, "back")

            # Draw a cursor with white color
            cv2.circle(ui_frame, pointer_pos, 2,
                       (255, 255, 255), -1)  # White center dot
            if continue_active or back_active:
                # Draw a larger cursor for active button (red)
                cv2.circle(ui_frame, pointer_pos, 10, (0, 0, 255), 2)
            else:
                cv2.circle(ui_frame, pointer_pos, 10,
                           (255, 255, 255), 2)  # White circle

        # Draw buttons
        self._draw_button(ui_frame, "continue", "Continue", continue_active)
        self._draw_button(ui_frame, "back", "Back", back_active)

        return ui_frame

    def draw_virtual_tryon(self, frame, pointer_pos=None):
        """Draw the virtual try-on screen"""
        h, w = frame.shape[:2]

        # Create UI background with minimal elements
        ui_frame = frame.copy()

        # Add semi-transparent header
        header_height = 80
        header = np.zeros((header_height, w, 3), dtype=np.uint8)
        header[:] = (30, 30, 30)

        # Apply the header with transparency
        for y in range(header_height):
            alpha = 0.7
            ui_frame[y, :] = cv2.addWeighted(
                ui_frame[y, :], 1-alpha, header[y, :], alpha, 0)

        # Add title
        if self.custom_font_available:
            ui_frame = self._put_text_with_custom_font(
                ui_frame, "VIRTUAL TRY-ON", (w//2-150, 30), 38)
            ui_frame = self._put_text_with_custom_font(
                ui_frame, "Use selection gesture to switch outfits", (w//2-220, 80), 22)
        else:
            cv2.putText(ui_frame, "VIRTUAL TRY-ON", (w//2-150, 50), self.font,
                        1.5, self.font_color, self.font_thickness, cv2.LINE_AA)
            cv2.putText(ui_frame, "Use selection gesture to switch outfits",
                        (w//2-220, 100), self.font, 0.8, self.font_color, 1, cv2.LINE_AA)

        # Check which button is active
        main_menu_active = False
        recalibrate_active = False
        exit_active = False

        if pointer_pos:
            x, y = pointer_pos
            main_menu_active = self.is_within_button(x, y, "main_menu")
            recalibrate_active = self.is_within_button(x, y, "recalibrate")
            exit_active = self.is_within_button(x, y, "exit")

            # Draw a cursor
            cv2.circle(ui_frame, pointer_pos, 10, (0, 255, 0), 2)
            cv2.circle(ui_frame, pointer_pos, 2, (0, 255, 0), -1)

        # Draw buttons
        self._draw_button(ui_frame, "main_menu", "Main Menu", main_menu_active)
        self._draw_button(ui_frame, "recalibrate",
                          "Recalibrate", recalibrate_active)

        return ui_frame
