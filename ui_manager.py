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
        self.width = 1282
        self.height = 752
        self.title_font_scale = 1.1  # Increased from 0.7
        self.text_font_scale = 0.7   # Increased from 0.45
        self.text_color = (255, 255, 255)  # White
        self.text_color_button = (0, 0, 0)  # Black text on buttons
        self.opacity = 0.6
        self.button_color = (255, 255, 255)  # White
        self.button_selected_color = (0, 255, 0)  # Green

        # Button style settings
        self.button_radius_ratio = 0.4  # Radius as a proportion of button heightttons

        # Current step for calibration
        self.current_step = 0

        # Define button areas for different screens (x, y, width, height)
        self.buttons = {
            # Calibration screen
            "calibration_complete": (100, 100, 1082, 78),

            # Main menu - adjust positions to match screenshot
            "body_scan": (240, 313, 800, 78),
            "voice_assistant": (240, 470, 800, 78),
            "exit": (240, 626, 800, 78),

            # Gender select
            "male": (200, 188, 300, 78),
            "female": (780, 188, 300, 78),
            "back": (240, 626, 800, 78),

            # Body scan screen
            "continue": (900, 78, 200, 78),
            "back_from_scan": (100, 78, 200, 78),

            # Voice assistant screen
            "continue_voice": (240, 626, 800, 78),
            "back_voice": (240, 720, 800, 78),

            # Virtual try-on screen
            "main_menu": (40, 31, 590, 78),
            "recalibrate": (650, 31, 590, 78),
            "top_select": (40, 595, 300, 63),
            "bottom_select": (360, 595, 300, 63)
        }

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_thickness = 2  # Increased from 1

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

        # Calculate radius based on button height
        button_radius = int(btn_h * self.button_radius_ratio)

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

        # Draw filled rectangle on mask
        cv2.rectangle(mask, (button_radius, 0),
                      (btn_w - button_radius, btn_h), 255, -1)
        cv2.rectangle(mask, (0, button_radius),
                      (btn_w, btn_h - button_radius), 255, -1)

        # Draw the four corner circles on mask
        cv2.circle(mask, (button_radius, button_radius),
                   button_radius, 255, -1)
        cv2.circle(mask, (btn_w - button_radius, button_radius),
                   button_radius, 255, -1)
        cv2.circle(mask, (button_radius, btn_h - button_radius),
                   button_radius, 255, -1)
        cv2.circle(mask, (btn_w - button_radius, btn_h - button_radius),
                   button_radius, 255, -1)

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

        # "Continue Setup" button at the top - rounded rectangle
        button_x = self.buttons["calibration_complete"][0]
        button_y = self.buttons["calibration_complete"][1]
        button_width = self.buttons["calibration_complete"][2]
        button_height = self.buttons["calibration_complete"][3]
        button_radius = int(button_height * self.button_radius_ratio)

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
        cv2.putText(overlay, "VIRTUAL TRY-ON", (w//2 - 200, 50),
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

        # Draw buttons using rounded rectangle style - more pill-shaped
        buttons = [
            ("body_scan", "Body Scan", body_scan_active),
            ("voice_assistant", "Voice Assistant", voice_active),
            ("exit", "Exit", exit_active)
        ]

        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            # Full pill shape: radius should be half of height
            button_radius = btn_h // 2

            # Button color based on active state
            button_color = self.button_selected_color if active else self.button_color

            # Increase opacity for better visibility
            alpha = 0.8 if active else 0.7

            # Draw the pill-shaped button
            # Left semicircle
            cv2.circle(overlay, (btn_x + button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Right semicircle
            cv2.circle(overlay, (btn_x + btn_w - button_radius, btn_y + btn_h//2),
                       button_radius, button_color, -1)
            # Center rectangle
            cv2.rectangle(overlay, (btn_x + button_radius, btn_y),
                          (btn_x + btn_w - button_radius, btn_y + btn_h), button_color, -1)

        # Blend overlay with webcam frame with slightly higher opacity for better contrast
        result = cv2.addWeighted(frame, 1, overlay, 0.7, 0)

        # Add text to buttons after blending (cleaner appearance)
        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, self.font_thickness)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            text_y = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(result, text, (text_x, text_y),
                        self.font, self.text_font_scale, self.text_color_button, self.font_thickness, cv2.LINE_AA)

        # Add title and subtitle again after blending for better visibility
        cv2.putText(result, "VIRTUAL TRY-ON", (w//2 - 200, 50),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)
        cv2.putText(result, "Choose your option", (w//2 - 120, 100),
                    self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        return result

    def draw_gender_select(self, frame, pointer_pos=None):
        """Draw the gender selection screen"""
        h, w = frame.shape[:2]

        # Create transparent overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Add title
        cv2.putText(overlay, "SELECT GENDER", (w//2 - 200, 50),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Add subtitle
        cv2.putText(overlay, "Choose your gender for accurate fit", (w//2 - 220, 100),
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

        # Draw gender buttons with full pill shape (like in the screenshot)
        buttons = [
            ("male", "Male", male_active),
            ("female", "Female", female_active),
            ("back", "Back", back_active)
        ]

        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            # Full pill shape: radius should be half of button height
            button_radius = btn_h // 2

            # Button color based on active state
            button_color = self.button_selected_color if active else self.button_color

            # Draw the pill-shaped button
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
        result = cv2.addWeighted(frame, 1, overlay, 0.7, 0)

        # Add text after blending for cleaner appearance
        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, self.font_thickness)[0]
            text_x = btn_x + (btn_w - text_size[0]) // 2
            text_y = btn_y + (btn_h + text_size[1]) // 2
            cv2.putText(result, text, (text_x, text_y),
                        self.font, self.text_font_scale, self.text_color_button, self.font_thickness, cv2.LINE_AA)

        # Add title and subtitle again after blending for better visibility
        cv2.putText(result, "SELECT GENDER", (w//2 - 200, 50),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)
        cv2.putText(result, "Choose your gender for accurate fit", (w//2 - 220, 100),
                    self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        return result

    def draw_body_scan(self, frame, pointer_pos=None, body_type=None):
        """Draw the body scan screen"""
        h, w = frame.shape[:2]

        # Create transparent overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Add title
        cv2.putText(overlay, "BODY SCAN", (w//2 - 120, 50),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Drawing the scanning area with a rectangle frame - update for new resolution
        # Make the scan area larger for the higher resolution
        scan_x = w//6
        scan_y = h//6
        scan_w = int(w * 2/3)
        scan_h = int(h * 2/3)
        cv2.rectangle(overlay, (scan_x, scan_y), (scan_x+scan_w, scan_y+scan_h),
                      (100, 100, 100), 2)

        # Calculate animation time
        current_time = time.time()
        self.scan_animation_timer = (current_time - self.start_time) * 2

        # Show body type if available
        if body_type:
            # Display body type info - green background with body type text
            text = f"Body Type: {body_type}"
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x = (w - text_size[0]) // 2
            # Position the body type indicator better - move down from title
            body_type_y = 120

            # Draw a green background box for the body type
            rect_radius = 10
            cv2.rectangle(overlay, (text_x - 20, body_type_y - 20),
                          (text_x + text_size[0] + 20, body_type_y + 10),
                          self.button_selected_color, -1)
        else:
            # Draw scanning line animation (horizontal line moving down)
            scan_y_pos = scan_y + \
                int((self.scan_animation_timer * 30) % scan_h)
            cv2.line(overlay, (scan_x, scan_y_pos), (scan_x + scan_w, scan_y_pos),
                     (0, 255, 255), 2)

        # Check which button is active
        continue_active = False
        back_active = False

        if pointer_pos:
            x, y = pointer_pos
            continue_active = self.is_within_button(x, y, "continue")
            back_active = self.is_within_button(x, y, "back_from_scan")

        # Draw buttons - use the pill-shaped rounded buttons like in screenshot
        buttons = [
            ("continue", "Continue", continue_active),
            ("back_from_scan", "Back", back_active)
        ]

        for button_name, text, active in buttons:
            if button_name not in self.buttons:
                continue

            btn_x, btn_y, btn_w, btn_h = self.buttons[button_name]
            # Make buttons more rounded (pill shape)
            button_radius = int(btn_h * 0.5)

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
        cv2.putText(result, "BODY SCAN", (w//2 - 120, 50),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Show body type text if available
        if body_type:
            # Display body type info
            text = f"Body Type: {body_type}"
            text_size = cv2.getTextSize(
                text, self.font, self.text_font_scale, 1)[0]
            text_x = (w - text_size[0]) // 2
            body_type_y = 120
            cv2.putText(result, text, (text_x, body_type_y),
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

        # Add instruction text - position at bottom of screen
        instruction = "Stand ~2m away from camera, Ensure your full body is visible"
        instruction_y = h - 100  # Position near bottom of screen

        # Center the instruction text
        text_size = cv2.getTextSize(
            instruction, self.font, self.text_font_scale, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(result, instruction, (text_x, instruction_y),
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

        # Add title - make it match the screenshot exactly
        title_text = "VOICE ASSISTANT"
        title_size = cv2.getTextSize(
            title_text, self.font, self.title_font_scale, 2)[0]
        title_x = (w - title_size[0]) // 2  # Center the title
        cv2.putText(overlay, title_text, (title_x, 50),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Draw talk button - center it better as shown in screenshot
        button_x = (w - 400) // 2  # Center the button (400px width)
        button_y = 130
        button_w = 400
        button_h = 50
        # Use full pill shape for buttons
        button_radius = button_h // 2

        # Button color based on listening state
        button_color = self.button_selected_color if self.is_listening else self.button_color

        # Draw the rounded button with full pill shape
        # Left semicircle
        cv2.circle(overlay, (button_x + button_radius, button_y + button_h//2),
                   button_radius, button_color, -1)
        # Right semicircle
        cv2.circle(overlay, (button_x + button_w - button_radius, button_y + button_h//2),
                   button_radius, button_color, -1)
        # Center rectangle
        cv2.rectangle(overlay, (button_x + button_radius, button_y),
                      (button_x + button_w - button_radius, button_y + button_h), button_color, -1)

        # Add example commands with full pill shape
        example_commands = [
            "\"Find clothes for asian male\"",
            "\"Show me casual outfits\"",
            "\"What's trending for summer 2025?\""
        ]

        y_pos = 220
        for cmd in example_commands:
            # Draw white rectangle background with full pill shape
            # Center the example commands (600px width)
            rect_x = (w - 600) // 2
            rect_y = y_pos
            rect_w = 600
            rect_h = 40
            # Use full pill shape
            rect_radius = rect_h // 2

            # Draw rounded rectangle for commands with full pill shape
            # Left semicircle
            cv2.circle(overlay, (rect_x + rect_radius, rect_y + rect_h//2),
                       rect_radius, self.button_color, -1)
            # Right semicircle
            cv2.circle(overlay, (rect_x + rect_w - rect_radius, rect_y + rect_h//2),
                       rect_radius, self.button_color, -1)
            # Center rectangle
            cv2.rectangle(overlay, (rect_x + rect_radius, rect_y),
                          (rect_x + rect_w - rect_radius, rect_y + rect_h), self.button_color, -1)

            y_pos += 50

        # Check which button is active
        continue_active = False
        if pointer_pos:
            x, y = pointer_pos
            continue_active = self.is_within_button(x, y, "continue_voice")
            # Draw cursor at pointer position
            cv2.circle(overlay, pointer_pos, 10, self.text_color, 2)
            cv2.circle(overlay, pointer_pos, 2, self.text_color, -1)

        # Draw Continue button with full pill shape as shown in screenshot
        btn_x, btn_y, btn_w, btn_h = self.buttons["continue_voice"]
        button_radius = btn_h // 2  # Full pill shape

        # Button color based on active state - make it match the green in screenshot
        button_color = self.button_selected_color if continue_active else self.button_color

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
        # Increased opacity to 0.7 for better visibility
        result = cv2.addWeighted(frame, 1, overlay, 0.7, 0)

        # ---- Add all text after blending ----

        # Add title again for better visibility
        cv2.putText(result, title_text, (title_x, 50),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Add listening status
        text = "Listening..." if self.is_listening else "Click to Talk"
        text_size = cv2.getTextSize(
            text, self.font, self.text_font_scale, 1)[0]
        text_x = (w - text_size[0]) // 2  # Center the text
        cv2.putText(result, text, (text_x, 100),
                    self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        # Add talk button text
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

        # Add example command texts after blending
        y_pos = 220
        for cmd in example_commands:
            rect_x = (w - 600) // 2  # Center the example commands
            rect_y = y_pos
            rect_h = 40

            # Add command text
            text_size = cv2.getTextSize(
                cmd, self.font, self.text_font_scale, 1)[0]
            # Center the text in the button
            text_x = rect_x + 20  # Add left padding
            text_y = rect_y + (rect_h + text_size[1]) // 2
            cv2.putText(result, cmd, (text_x, text_y),
                        self.font, self.text_font_scale, self.text_color_button, 1, cv2.LINE_AA)
            y_pos += 50

        # Add continue button text
        btn_x, btn_y, btn_w, btn_h = self.buttons["continue_voice"]
        text = "Continue"
        text_size = cv2.getTextSize(
            text, self.font, self.text_font_scale, self.font_thickness)[0]
        text_x = btn_x + (btn_w - text_size[0]) // 2
        text_y = btn_y + (btn_h + text_size[1]) // 2
        cv2.putText(result, text, (text_x, text_y),
                    self.font, self.text_font_scale, self.text_color_button, self.font_thickness, cv2.LINE_AA)

        return result

    def draw_virtual_tryon(self, frame, pointer_pos=None, active_clothing_type='top', gender=None, body_type=None, recommendation_text=None):
        """Draw the virtual try-on screen"""
        h, w = frame.shape[:2]

        # Create transparent overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Add title at top centered
        title_text = "VIRTUAL TRY-ON"
        title_size = cv2.getTextSize(
            title_text, self.font, self.title_font_scale, 2)[0]
        title_x = (w - title_size[0]) // 2
        cv2.putText(overlay, title_text, (title_x, 50),
                    self.font, self.title_font_scale, self.text_color, 2, cv2.LINE_AA)

        # Check if Main Menu button is active
        main_menu_active = False
        if pointer_pos:
            x, y = pointer_pos
            main_menu_active = self.is_within_button(x, y, "main_menu")
            # Draw cursor at pointer position
            cv2.circle(overlay, pointer_pos, 10, self.text_color, 2)
            cv2.circle(overlay, pointer_pos, 2, self.text_color, -1)

        # Draw Main Menu button as a pill shape with reduced width
        btn_x, btn_y, btn_w, btn_h = self.buttons["main_menu"]
        # Reduce button width from 590 to 300
        original_width = btn_w
        reduced_width = 300

        # Adjust x position to keep button centered
        btn_x = btn_x + ((original_width - reduced_width) // 2)
        btn_w = reduced_width

        button_radius = btn_h // 2  # Full pill shape
        button_color = self.button_selected_color if main_menu_active else self.button_color

        # Left semicircle
        cv2.circle(overlay, (btn_x + button_radius, btn_y + btn_h//2),
                   button_radius, button_color, -1)
        # Right semicircle
        cv2.circle(overlay, (btn_x + btn_w - button_radius, btn_y + btn_h//2),
                   button_radius, button_color, -1)
        # Center rectangle
        cv2.rectangle(overlay, (btn_x + button_radius, btn_y),
                      (btn_x + btn_w - button_radius, btn_y + btn_h), button_color, -1)

        # Add recommendation text area at the bottom (like in screenshot)
        if recommendation_text:
            # Create semi-transparent dark background for text
            rec_x = w // 4
            rec_y = h - 150
            rec_w = w // 2
            rec_h = 100
            rec_radius = 15

            # Draw rounded rectangle with darker background
            cv2.rectangle(overlay, (rec_x, rec_y), (rec_x + rec_w, rec_y + rec_h),
                          (30, 30, 30), -1)

            # We'll add the text after blending

        # Blend overlay with webcam frame
        result = cv2.addWeighted(frame, 1, overlay, 0.7, 0)

        # Add Main Menu button text
        text = "Main Menu"
        text_size = cv2.getTextSize(
            text, self.font, self.text_font_scale, self.font_thickness)[0]
        text_x = btn_x + (btn_w - text_size[0]) // 2
        text_y = btn_y + (btn_h + text_size[1]) // 2
        cv2.putText(result, text, (text_x, text_y),
                    self.font, self.text_font_scale, self.text_color_button, self.font_thickness, cv2.LINE_AA)

        # Add recommendation text if provided (after blending)
        if recommendation_text:
            # Split text into lines if it's long
            max_chars_per_line = 50
            lines = []
            words = recommendation_text.split()
            current_line = ""

            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars_per_line:
                    if current_line:
                        current_line += " "
                    current_line += word
                else:
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            # Draw the text lines
            text_y = h - 130
            for line in lines:
                text_size = cv2.getTextSize(
                    line, self.font, self.text_font_scale, 1)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(result, line, (text_x, text_y),
                            self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)
                text_y += 30

        # # Add "Active: Top/Bottom" indicator in the bottom right (as shown in screenshot)
        # if active_clothing_type:
        #     clothing_text = f"Active: {active_clothing_type.capitalize()}"
        #     text_size = cv2.getTextSize(
        #         clothing_text, self.font, self.text_font_scale, 1)[0]
        #     cv2.putText(result, clothing_text, (w - text_size[0] - 20, h - 20),
        #                 self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        # Add gender and body type info (if needed)
        if gender or body_type:
            info_y = 100
            if gender:
                gender_text = f"Gender: {gender}"
                cv2.putText(result, gender_text, (20, info_y),
                            self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)
                info_y += 30
            if body_type:
                body_type_text = f"Body Type: {body_type}"
                cv2.putText(result, body_type_text, (20, info_y),
                            self.font, self.text_font_scale, self.text_color, 1, cv2.LINE_AA)

        return result
