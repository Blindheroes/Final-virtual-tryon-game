"""
User Interface Module for Virtual Try-On Game
This module handles the UI elements and screens for the virtual try-on application.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import pygame  # For font rendering


class Button:
    def __init__(self, x, y, width, height, text, action=None, icon=None, color_scheme="blue", style="default"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.action = action
        self.icon = icon
        self.style = style  # default, bordered

        # Color schemes for different button types
        self.color_schemes = {
            "blue": {
                # Blue with border in BGR
                "default": {"start": (255, 128, 0), "end": (255, 80, 0)},
                "hover": {"start": (0, 200, 50), "end": (0, 150, 50)},
                "active": {"start": (0, 50, 255), "end": (0, 0, 200)},
                # Navy blue for bordered buttons
                "bordered": {"bg": (120, 60, 0), "border": (255, 150, 0)}
            },
            "purple": {
                # Purple for SAVE button
                "default": {"start": (255, 0, 255), "end": (200, 0, 200)},
                "hover": {"start": (255, 100, 255), "end": (220, 50, 220)},
                "active": {"start": (255, 0, 128), "end": (200, 0, 100)},
                "bordered": {"bg": (120, 0, 120), "border": (255, 100, 255)}
            },
            "dark-blue": {
                # Dark blue for RESCAN button
                "default": {"start": (200, 50, 0), "end": (150, 20, 0)},
                "hover": {"start": (220, 70, 0), "end": (170, 40, 0)},
                "active": {"start": (240, 90, 0), "end": (190, 60, 0)},
                "bordered": {"bg": (120, 30, 0), "border": (180, 60, 0)}
            },
            "red": {
                # Red for EXIT button
                "default": {"start": (0, 0, 200), "end": (0, 0, 150)},
                "hover": {"start": (0, 0, 255), "end": (0, 0, 180)},
                "active": {"start": (80, 0, 255), "end": (50, 0, 200)},
                "bordered": {"bg": (0, 0, 120), "border": (0, 0, 180)}
            },
            "accent": {
                # Magenta/Pink for female button (BGR format)
                "default": {"start": (255, 0, 255), "end": (200, 0, 200)},
                "hover": {"start": (255, 50, 255), "end": (230, 20, 230)},
                "active": {"start": (180, 0, 180), "end": (150, 0, 150)},
                "bordered": {"bg": (100, 0, 100), "border": (255, 0, 255)}
            }
        }
        self.color_scheme = color_scheme
        self.corner_radius = min(10, height // 3)  # Rounded corners
        self.border_thickness = 2

        # Animation properties
        self.animation_state = 0.0  # 0.0 to 1.0
        self.is_hovered = False
        self.is_active = False

    def is_over(self, pos):
        """Check if position is over the button"""
        if pos is None:
            return False

        x, y = pos
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def _draw_rounded_rectangle(self, frame, color_start, color_end, thickness=-1):
        """Draw a rounded rectangle with gradient fill"""
        # Create gradient
        gradient = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for i in range(self.height):
            ratio = i / self.height
            gradient[i, :] = [
                int(color_start[0] + ratio * (color_end[0] - color_start[0])),
                int(color_start[1] + ratio * (color_end[1] - color_start[1])),
                int(color_start[2] + ratio * (color_end[2] - color_start[2]))
            ]

        # Create a mask for rounded corners
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.rectangle(mask, (self.corner_radius, 0),
                      (self.width - self.corner_radius, self.height), 255, -1)
        cv2.rectangle(mask, (0, self.corner_radius),
                      (self.width, self.height - self.corner_radius), 255, -1)
        cv2.circle(mask, (self.corner_radius, self.corner_radius),
                   self.corner_radius, 255, -1)
        cv2.circle(mask, (self.width - self.corner_radius, self.corner_radius),
                   self.corner_radius, 255, -1)
        cv2.circle(mask, (self.width - self.corner_radius, self.height - self.corner_radius),
                   self.corner_radius, 255, -1)
        cv2.circle(mask, (self.corner_radius, self.height - self.corner_radius),
                   self.corner_radius, 255, -1)

        # Apply the mask to the gradient
        masked_gradient = cv2.bitwise_and(gradient, gradient, mask=mask)

        # Create ROI in the frame
        roi = frame[self.y:self.y+self.height, self.x:self.x+self.width]

        # Blend the masked gradient with the ROI
        if thickness == -1:  # Fill
            # Blend with alpha
            alpha = 0.9
            cv2.addWeighted(masked_gradient, alpha, roi, 1.0-alpha, 0, roi)

            # Add border for better definition
            border_mask = np.zeros_like(mask)
            cv2.rectangle(border_mask, (self.corner_radius, 0),
                          (self.width - self.corner_radius, self.height), 255, self.border_thickness)
            cv2.rectangle(border_mask, (0, self.corner_radius),
                          (self.width, self.height - self.corner_radius), 255, self.border_thickness)
            cv2.circle(border_mask, (self.corner_radius, self.corner_radius),
                       self.corner_radius, 255, self.border_thickness)
            cv2.circle(border_mask, (self.width - self.corner_radius, self.corner_radius),
                       self.corner_radius, 255, self.border_thickness)
            cv2.circle(border_mask, (self.width - self.corner_radius, self.height - self.corner_radius),
                       self.corner_radius, 255, self.border_thickness)
            cv2.circle(border_mask, (self.corner_radius, self.height - self.corner_radius),
                       self.corner_radius, 255, self.border_thickness)

            # Add white border
            border_color = (255, 255, 255)
            border_roi = np.zeros_like(roi)
            border_roi[border_mask > 0] = border_color
            alpha_border = 0.7
            cv2.addWeighted(border_roi, alpha_border,
                            roi, 1.0-alpha_border, 0, roi)

    def draw(self, frame, pointer_pos=None, is_selecting=False):
        """Draw the button on the frame"""
        # Update hover state
        prev_hover = self.is_hovered
        self.is_hovered = pointer_pos and self.is_over(pointer_pos)

        # Update active state
        prev_active = self.is_active
        self.is_active = self.is_hovered and is_selecting

        # Handle animation state
        if not prev_hover and self.is_hovered:
            self.animation_state = 0.3  # Start hover animation
        elif prev_hover and not self.is_hovered:
            self.animation_state = 0.0  # Reset animation

        if not prev_active and self.is_active:
            self.animation_state = 0.6  # Start active animation
        elif prev_active and not self.is_active:
            self.animation_state = 0.3 if self.is_hovered else 0.0  # Return to hover or default

        if self.style == "bordered":
            # Draw bordered button style (like in the image)
            bg_color = self.color_schemes[self.color_scheme]["bordered"]["bg"]
            border_color = self.color_schemes[self.color_scheme]["bordered"]["border"]

            # Draw background rectangle
            cv2.rectangle(frame, (self.x, self.y), (self.x + self.width, self.y + self.height),
                          bg_color, -1)

            # Draw decorative corners (like in the image)
            corner_size = 10
            line_thickness = 2

            # Top-left corner
            cv2.line(frame, (self.x, self.y + corner_size),
                     (self.x, self.y), border_color, line_thickness)
            cv2.line(frame, (self.x, self.y),
                     (self.x + corner_size, self.y), border_color, line_thickness)

            # Top-right corner
            cv2.line(frame, (self.x + self.width - corner_size, self.y),
                     (self.x + self.width, self.y), border_color, line_thickness)
            cv2.line(frame, (self.x + self.width, self.y),
                     (self.x + self.width, self.y + corner_size), border_color, line_thickness)

            # Bottom-left corner
            cv2.line(frame, (self.x, self.y + self.height - corner_size),
                     (self.x, self.y + self.height), border_color, line_thickness)
            cv2.line(frame, (self.x, self.y + self.height),
                     (self.x + corner_size, self.y + self.height), border_color, line_thickness)

            # Bottom-right corner
            cv2.line(frame, (self.x + self.width - corner_size, self.y + self.height),
                     (self.x + self.width, self.y + self.height), border_color, line_thickness)
            cv2.line(frame, (self.x + self.width, self.y + self.height),
                     (self.x + self.width, self.y + self.height - corner_size), border_color, line_thickness)

            # Text color is white for bordered buttons
            text_color = (255, 255, 255)

            # Calculate text position (centered)
            text_size = cv2.getTextSize(
                self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = self.x + (self.width - text_size[0]) // 2
            text_y = self.y + (self.height + text_size[1]) // 2

            # Draw text with anti-aliasing
            cv2.putText(frame, self.text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        else:
            # Original gradient style
            # Select color scheme based on state
            if self.is_active:
                color_start = self.color_schemes[self.color_scheme]["active"]["start"]
                color_end = self.color_schemes[self.color_scheme]["active"]["end"]
            elif self.is_hovered:
                color_start = self.color_schemes[self.color_scheme]["hover"]["start"]
                color_end = self.color_schemes[self.color_scheme]["hover"]["end"]
            else:
                color_start = self.color_schemes[self.color_scheme]["default"]["start"]
                color_end = self.color_schemes[self.color_scheme]["default"]["end"]

            # Apply animation effects
            pulse_effect = np.sin(
                cv2.getTickCount() * 0.0000025 + self.animation_state * 10) * 0.1 + 0.9
            color_start = tuple(int(c * pulse_effect) for c in color_start)
            color_end = tuple(int(c * pulse_effect) for c in color_end)

            # Draw the rounded rectangle button
            self._draw_rounded_rectangle(frame, color_start, color_end, -1)

            # Draw button text with better contrast
            text_size = cv2.getTextSize(
                self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = self.x + (self.width - text_size[0]) // 2
            text_y = self.y + (self.height + text_size[1]) // 2

            # Draw text shadow for better contrast
            cv2.putText(frame, self.text, (text_x+1, text_y+1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # Draw text
            cv2.putText(frame, self.text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


class UserInterface:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # UI theme colors
        self.colors = {
            "primary": (255, 140, 0),      # Orange in BGR
            "secondary": (0, 200, 100),    # Green in BGR
            "accent": (255, 0, 255),       # Magenta in BGR
            "danger": (0, 0, 200),         # Red in BGR
            "text_light": (255, 255, 255),  # White
            "text_dark": (50, 50, 50),     # Dark gray
            # Dark overlay - darker for better contrast
            "overlay": (20, 20, 30),
            "highlight": (0, 255, 255)     # Yellow in BGR
        }

        # Load UI assets if available
        self.ui_assets = {}
        try:
            self.ui_assets["btn_left"] = cv2.imread(
                "UI/button-left.png", cv2.IMREAD_UNCHANGED)
            self.ui_assets["btn_right"] = cv2.imread(
                "UI/button-right.png", cv2.IMREAD_UNCHANGED)
            self.ui_assets["btn_save"] = cv2.imread(
                "UI/button-save.png", cv2.IMREAD_UNCHANGED)
            self.ui_assets["btn_rescan"] = cv2.imread(
                "UI/button-rescan.png", cv2.IMREAD_UNCHANGED)
            self.ui_assets["guide"] = cv2.imread(
                "UI/guide.png", cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Warning: Could not load UI assets: {e}")
            self.ui_assets = {}

        # Hand gesture instruction images could be loaded here
        self.gesture_pointer = None
        self.gesture_select = None

        # Initialize UI elements for different screens
        self.init_welcome_screen()
        self.init_gender_selection()
        self.init_try_on_screen()

        # For animation effects
        self.animation_time = 0

        # Initialize pygame for font rendering
        pygame.init()

        # Load custom Montserrat font
        self.font_regular = None
        self.font_bold = None
        self.font_light = None
        self.use_custom_font = False

        # Path to font files
        font_dir = Path("fonts/static")
        if font_dir.exists():
            try:
                # Load different font weights
                self.font_regular = pygame.font.Font(
                    str(font_dir / "Montserrat-Regular.ttf"), 24)
                self.font_bold = pygame.font.Font(
                    str(font_dir / "Montserrat-Bold.ttf"), 24)
                self.font_light = pygame.font.Font(
                    str(font_dir / "Montserrat-Light.ttf"), 24)
                self.use_custom_font = True
                print("Montserrat font loaded successfully")
            except Exception as e:
                print(f"Error loading Montserrat font: {e}")
        else:
            print(f"Font directory not found at {font_dir}")

    def render_text(self, text, size, color=(255, 255, 255), bold=False):
        """Render text using Montserrat font and return as OpenCV image"""
        if not self.use_custom_font:
            return None

        # Select font weight based on parameters
        font = self.font_bold if bold else self.font_regular

        # Resize font if needed (pygame doesn't support dynamic sizing, so we create a new font)
        if size != 24:  # Default size
            font_path = "fonts/static/Montserrat-Bold.ttf" if bold else "fonts/static/Montserrat-Regular.ttf"
            try:
                font = pygame.font.Font(font_path, size)
            except:
                pass

        # Render text to a pygame surface (white text)
        text_surface = font.render(
            text, True, (color[2], color[1], color[0]))  # RGB to BGR

        # Convert the pygame surface to a numpy array for OpenCV
        text_data = pygame.surfarray.array3d(text_surface)
        # Transpose because pygame and OpenCV have different coordinate systems
        text_data = np.transpose(text_data, (1, 0, 2))

        # Create alpha channel mask from the original surface
        alpha_surface = pygame.surfarray.array_alpha(
            text_surface) if text_surface.get_alpha() else None
        if alpha_surface is not None:
            alpha_surface = np.transpose(alpha_surface, (1, 0))
            # Stack the RGB and alpha channels
            text_data = np.dstack((text_data, alpha_surface))

        return text_data

    def draw_text_with_font(self, frame, text, position, size=24, color=(255, 255, 255), bold=False):
        """Draw text on frame using Montserrat font"""
        if self.use_custom_font:
            # Render text using Pygame
            text_image = self.render_text(text, size, color, bold)
            if text_image is not None:
                # Get text dimensions
                h, w = text_image.shape[:2]
                x, y = position

                # Create a region of interest on the frame
                roi = frame[y:y+h, x:x+w]

                # Check if ROI is valid
                if roi.shape[0] > 0 and roi.shape[1] > 0 and roi.shape[:2] == text_image.shape[:2]:
                    # If we have alpha channel, use it for blending
                    if text_image.shape[2] == 4:
                        # Normalize alpha channel to range 0..1
                        alpha = text_image[:, :, 3] / 255.0
                        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

                        # Blend text with frame using alpha
                        blended = (
                            text_image[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
                        frame[y:y+h, x:x+w] = blended
                    else:
                        # Just copy the text image to the frame
                        frame[y:y+h, x:x+w] = text_image
                return

        # Fall back to OpenCV text rendering if custom font fails
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    size/30, color, 2 if bold else 1)

    def _create_panel(self, frame, x, y, width, height, alpha=0.7):
        """Create a semi-transparent panel for better text readability"""
        panel = frame[y:y+height, x:x+width].copy()
        overlay = np.zeros_like(panel)
        overlay[:] = self.colors["overlay"]
        cv2.addWeighted(overlay, alpha, panel, 1-alpha, 0, panel)

        # Add a subtle border
        border_thickness = 2
        cv2.rectangle(panel, (0, 0), (width-1, height-1),
                      (255, 255, 255), border_thickness)

        frame[y:y+height, x:x+width] = panel
        return (x, y, width, height)

    def _draw_hand_gesture_help(self, frame, x, y):
        """Draw visual indicators for hand gestures"""
        # Background panel
        panel_width, panel_height = 200, 80
        self._create_panel(frame, x, y, panel_width, panel_height)

        # Draw hand position for pointer (index finger)
        cv2.putText(frame, "Point:", (x + 10, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text_light"], 1)

        # Draw finger illustration for pointer
        finger_x = x + 80
        finger_y = y + 25
        cv2.circle(frame, (finger_x, finger_y), 8,
                   self.colors["text_light"], -1)
        cv2.line(frame, (finger_x, finger_y), (finger_x,
                 finger_y - 20), self.colors["text_light"], 3)

        # Draw hand position for select (index + pinky)
        cv2.putText(frame, "Select:", (x + 10, y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text_light"], 1)

        # Draw finger illustration for select gesture
        finger_x = x + 80
        finger_y = y + 60
        # Index finger
        cv2.circle(frame, (finger_x, finger_y), 8,
                   self.colors["text_light"], -1)
        cv2.line(frame, (finger_x, finger_y), (finger_x,
                 finger_y - 20), self.colors["text_light"], 3)
        # Pinky finger
        cv2.circle(frame, (finger_x + 50, finger_y),
                   8, self.colors["text_light"], -1)
        cv2.line(frame, (finger_x + 50, finger_y), (finger_x +
                 50, finger_y - 20), self.colors["text_light"], 3)

    def _draw_clothing_preview(self, frame, clothing_type, item, x, y, width=120, height=100):
        """Draw a preview of the clothing item"""
        # Create preview panel
        self._create_panel(frame, x, y, width, height, 0.5)

        # Draw clothing name
        cv2.putText(frame, f"{clothing_type.title()}: {item}",
                    (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text_light"], 1)

        # Draw a simple clothing icon based on type
        if clothing_type.lower() == "top":
            # Simple shirt shape
            pts = np.array([
                [x + width//2 - 30, y + 45],
                [x + width//2 + 30, y + 45],
                [x + width//2 + 40, y + 55],
                [x + width//2 + 30, y + height - 20],
                [x + width//2 - 30, y + height - 20],
                [x + width//2 - 40, y + 55]
            ], np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Color based on item
            if "polo" in item.lower():
                color = (255, 128, 0)  # Orange
            elif "shirt" in item.lower():
                color = (0, 128, 255)  # Red
            else:
                color = (128, 255, 0)  # Green

            cv2.fillPoly(frame, [pts], color)
            # Add collar
            cv2.line(frame, (x + width//2 - 10, y + 45),
                     (x + width//2, y + 60), color, 2)
            cv2.line(frame, (x + width//2 + 10, y + 45),
                     (x + width//2, y + 60), color, 2)

        else:  # Bottom
            # Simple pants/skirt shape
            if "skirt" in item.lower():
                # A-line skirt
                pts = np.array([
                    [x + width//2 - 25, y + 45],
                    [x + width//2 + 25, y + 45],
                    [x + width//2 + 40, y + height - 20],
                    [x + width//2 - 40, y + height - 20]
                ], np.int32)
                pts = pts.reshape((-1, 1, 2))
                color = (255, 0, 255)  # Pink
                cv2.fillPoly(frame, [pts], color)
            else:
                # Pants
                cv2.rectangle(frame, (x + width//2 - 25, y + 45),
                              (x + width//2 - 5, y + height - 20), (0, 0, 128), -1)
                cv2.rectangle(frame, (x + width//2 + 5, y + 45),
                              (x + width//2 + 25, y + height - 20), (0, 0, 128), -1)
                cv2.rectangle(frame, (x + width//2 - 25, y + 45),
                              (x + width//2 + 25, y + 55), (0, 0, 128), -1)

    def init_welcome_screen(self):
        """Initialize welcome screen UI elements"""
        self.welcome_title = "Welcome to Virtual Try-On"
        self.welcome_subtitle = "Use hand gestures to control the interface"
        self.start_button = Button(
            self.width // 2 - 100,
            self.height // 2 + 50,
            200, 60, "Start", "start", color_scheme="blue"
        )

    def init_gender_selection(self):
        """Initialize gender selection screen UI elements"""
        self.gender_title = "Select Your Gender"
        # Male button with better position and styling
        self.male_button = Button(
            self.width // 4 - 100,
            self.height // 2,
            200, 70, "Male", "male", color_scheme="blue"
        )
        # Female button with better position and styling
        self.female_button = Button(
            3 * self.width // 4 - 100,
            self.height // 2,
            200, 70, "Female", "female", color_scheme="accent"
        )

    def init_try_on_screen(self):
        """Initialize try-on screen UI elements with bordered style buttons"""
        button_height = 50

        # Navigation buttons with better sizing and style matching the image
        nav_button_width = 120

        # Top navigation buttons - positioned on left and right sides
        self.prev_top_button = Button(
            30, self.height // 3 - 30,
            nav_button_width, button_height, "Previous", "prev_top",
            color_scheme="blue", style="bordered"
        )
        self.next_top_button = Button(
            self.width - nav_button_width - 30, self.height // 3 - 30,
            nav_button_width, button_height, "Next", "next_top",
            color_scheme="blue", style="bordered"
        )

        # Bottom navigation buttons - positioned on left and right sides
        self.prev_bottom_button = Button(
            30, self.height // 2 + 30,
            nav_button_width, button_height, "Previous", "prev_bottom",
            color_scheme="blue", style="bordered"
        )
        self.next_bottom_button = Button(
            self.width - nav_button_width - 30, self.height // 2 + 30,
            nav_button_width, button_height, "Next", "next_bottom",
            color_scheme="blue", style="bordered"
        )

        # Action buttons at the bottom with equal spacing
        action_button_width = 120
        action_button_height = 50
        button_spacing = (self.width - (3 * action_button_width)) // 4
        bottom_margin = 30

        # Save, Rescan, and Exit buttons with appropriate colors
        self.save_button = Button(
            button_spacing,
            self.height - action_button_height - bottom_margin,
            action_button_width, action_button_height, "SAVE", "save",
            color_scheme="purple", style="bordered"
        )
        self.rescan_button = Button(
            button_spacing * 2 + action_button_width,
            self.height - action_button_height - bottom_margin,
            action_button_width, action_button_height, "RESCAN", "rescan",
            color_scheme="dark-blue", style="bordered"
        )
        self.exit_button = Button(
            button_spacing * 3 + action_button_width * 2,
            self.height - action_button_height - bottom_margin,
            action_button_width, action_button_height, "EXIT", "exit",
            color_scheme="red", style="bordered"
        )

    def draw_welcome_screen(self, frame):
        """Draw the welcome screen with improved visuals"""
        self.animation_time = cv2.getTickCount() * 0.0000025

        # Add semi-transparent overlay with gradient
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Create gradient background
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            ratio = i / h
            gradient[i, :] = [
                int(30 + ratio * 20),  # Subtle dark gradient
                int(30 + ratio * 20),
                int(50 + ratio * 30)
            ]

        # Apply gradient with transparency
        cv2.addWeighted(gradient, 0.7, overlay, 0.3, 0, overlay)
        frame[:] = overlay

        # Create title panel
        panel_y = self.height // 4
        panel_height = 120
        panel_width = 500
        panel_x = (self.width - panel_width) // 2

        self._create_panel(frame, panel_x, panel_y, panel_width, panel_height)

        # Draw title with shadow for depth
        title_x = self.width // 2 - 200
        title_y = panel_y + 50
        cv2.putText(frame, self.welcome_title,
                    (title_x + 2, title_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(frame, self.welcome_title,
                    (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Draw subtitle
        subtitle_y = panel_y + 90
        cv2.putText(frame, self.welcome_subtitle,
                    (title_x + 30, subtitle_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)

        # Add visual hand gesture help
        self._draw_hand_gesture_help(
            frame, self.width // 2 - 100, panel_y + panel_height + 20)

        # Draw pulsing start button with animation
        self.start_button.draw(frame)

    def draw_gender_selection(self, frame, pointer_pos, is_selecting):
        """Draw the gender selection screen with improved visuals"""
        # Add semi-transparent overlay with gradient
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Create gradient background
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            ratio = i / h
            gradient[i, :] = [
                int(30 + ratio * 20),  # Subtle dark gradient
                int(30 + ratio * 20),
                int(50 + ratio * 30)
            ]

        # Apply gradient with transparency
        cv2.addWeighted(gradient, 0.7, overlay, 0.3, 0, overlay)
        frame[:] = overlay

        # Create title panel
        panel_y = self.height // 5
        panel_height = 80
        panel_width = 400
        panel_x = (self.width - panel_width) // 2

        self._create_panel(frame, panel_x, panel_y, panel_width, panel_height)

        # Draw title with shadow for depth
        title_x = self.width // 2 - 150
        title_y = panel_y + 50
        cv2.putText(frame, self.gender_title,
                    (title_x + 2, title_y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(frame, self.gender_title,
                    (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Add icons or silhouettes for male/female
        male_icon_x = self.width // 4
        female_icon_x = 3 * self.width // 4
        icon_y = self.height // 2 - 120
        icon_size = 100

        # Male icon (simplified)
        cv2.rectangle(frame, (male_icon_x - icon_size//2, icon_y),
                      (male_icon_x + icon_size//2, icon_y + icon_size), (255, 0, 0), 2)

        # Female icon (simplified)
        cv2.circle(frame, (female_icon_x, icon_y + icon_size//2),
                   icon_size//2, (255, 0, 255), 2)

        # Draw gender buttons
        self.male_button.draw(frame, pointer_pos, is_selecting)
        self.female_button.draw(frame, pointer_pos, is_selecting)

        # Add visual hand gesture help in the bottom corner
        self._draw_hand_gesture_help(frame, 20, self.height - 100)

    def draw_scanning_screen(self, frame, scan_complete):
        """Draw the body scanning screen with improved animation"""
        # Add semi-transparent darkening overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width,
                      self.height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        animation_time = cv2.getTickCount() * 0.0000025

        if not scan_complete:
            # Create scanning effect panel
            panel_width = 400
            panel_height = 120
            panel_x = (self.width - panel_width) // 2
            panel_y = (self.height - panel_height) // 2

            self._create_panel(frame, panel_x, panel_y,
                               panel_width, panel_height, 0.6)

            # Draw scanning text with animation
            scan_text = "Scanning Body..."
            cv2.putText(frame, scan_text,
                        (panel_x + 50, panel_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Loading animation (dots)
            dots = int((animation_time % 1.0) * 6)  # 0 to 5 dots
            for i in range(dots):
                cv2.circle(frame, (panel_x + 300 + i*15,
                           panel_y + 50), 5, (0, 255, 0), -1)

            # Add scanning effect (horizontal line moving up and down)
            scan_line_y = int((self.height // 2) +
                              np.sin(animation_time * 2) * (self.height // 4))

            # Add glow to scan line
            for i in range(5):
                alpha = 0.2 - i * 0.04
                thickness = 6 - i
                color = (0, 200 - i*30, 0)
                y_offset = scan_line_y + i*2
                cv2.line(frame, (0, y_offset),
                         (self.width, y_offset), color, thickness)

            # Main scan line
            cv2.line(frame, (0, scan_line_y),
                     (self.width, scan_line_y), (0, 255, 50), 2)

            # Add body outline visualization
            body_points = []
            for i in range(10):
                angle = animation_time * 3 + i * 0.63
                noise = np.sin(i * 8.7) * 5
                x = int(self.width // 2 + np.sin(angle) * 10)
                y = int(scan_line_y + noise)
                body_points.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            # Connect points with lines
            for i in range(len(body_points)-1):
                cv2.line(frame, body_points[i],
                         body_points[i+1], (0, 255, 255), 1)

        else:
            # Success panel
            panel_width = 400
            panel_height = 100
            panel_x = (self.width - panel_width) // 2
            panel_y = (self.height - panel_height) // 2

            self._create_panel(frame, panel_x, panel_y,
                               panel_width, panel_height, 0.6)

            # Draw completion message with animation
            cv2.putText(frame, "Scan Complete!",
                        (panel_x + 80, panel_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # Add checkmark animation
            check_x = panel_x + 50
            check_y = panel_y + 60
            check_size = 40

            # Animated checkmark
            check_progress = min(1.0, (animation_time % 1.0) * 2)
            if check_progress > 0:
                pt1 = (int(check_x - check_size * 0.3), int(check_y))
                pt2 = (int(check_x), int(
                    check_y + check_size * 0.5 * check_progress))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

                if check_progress > 0.5:
                    pt3_prog = min(1.0, (check_progress - 0.5) * 2)
                    pt3 = (int(check_x + check_size * 0.7 * pt3_prog),
                           int(check_y - check_size * 0.5 * pt3_prog))
                    cv2.line(frame, pt2, pt3, (0, 255, 0), 3)

    def draw_try_on_screen(self, frame, pointer_pos, is_selecting):
        """Draw the try-on screen with improved clothing navigation controls"""
        # Add dark semi-transparent overlay for better contrast
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width,
                      self.height), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw "TRY-ON" title at the top with larger font - positioned exactly as in the image
        title_text = "TRY-ON"
        if self.use_custom_font:
            # Use custom Montserrat font for the title
            self.draw_text_with_font(
                frame, title_text, (self.width//2-150, 140), 92, (255, 255, 255), True)
        else:
            # Fall back to OpenCV text
            text_size = cv2.getTextSize(
                title_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 5)[0]
            title_x = (self.width - text_size[0]) // 2
            # Draw with shadow for better visibility
            cv2.putText(frame, title_text, (title_x, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)

        # Draw hand gesture guide in the top-right corner as shown in the image
        guide_x = self.width - 150
        guide_y = 100
        guide_width = 120
        guide_height = 80

        # Create background for the guide
        cv2.rectangle(frame, (guide_x, guide_y), (guide_x + guide_width, guide_y + guide_height),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (guide_x, guide_y), (guide_x + guide_width, guide_y + guide_height),
                      (255, 255, 255), 1)

        # Add text for the guide
        if self.use_custom_font:
            self.draw_text_with_font(
                frame, "Point:", (guide_x + 10, guide_y + 20), 18, (255, 255, 255), True)
            self.draw_text_with_font(
                frame, "Select:", (guide_x + 10, guide_y + 50), 18, (255, 255, 255), True)
        else:
            cv2.putText(frame, "Point:", (guide_x + 10, guide_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Select:", (guide_x + 10, guide_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw the gesture indicators (circles)
        cv2.circle(frame, (guide_x + 90, guide_y + 15), 8,
                   (255, 255, 255), -1)  # Point indicator
        cv2.circle(frame, (guide_x + 90, guide_y + 45), 8,
                   (255, 255, 255), -1)  # Select indicator

        # Draw clothing navigation buttons
        self.prev_top_button.draw(frame, pointer_pos, is_selecting)
        self.next_top_button.draw(frame, pointer_pos, is_selecting)
        self.prev_bottom_button.draw(frame, pointer_pos, is_selecting)
        self.next_bottom_button.draw(frame, pointer_pos, is_selecting)

        # Add labels for navigation - positioned as in the image
        if self.use_custom_font:
            # Use larger font for TOPS and BOTTOMS labels
            self.draw_text_with_font(frame, "TOPS",
                                     (self.width // 2 - 50, self.height // 3 - 10),
                                     36, (255, 255, 255), True)

            self.draw_text_with_font(frame, "BOTTOMS",
                                     (self.width // 2 - 80, self.height // 2 + 50),
                                     36, (255, 255, 255), True)
        else:
            cv2.putText(frame, "TOPS",
                        (self.width // 2 - 50, self.height // 3 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.putText(frame, "BOTTOMS",
                        (self.width // 2 - 80, self.height // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Draw action buttons at the bottom
        self.save_button.draw(frame, pointer_pos, is_selecting)
        self.rescan_button.draw(frame, pointer_pos, is_selecting)
        self.exit_button.draw(frame, pointer_pos, is_selecting)

    def get_gender_selection(self, pointer_pos):
        """Check if a gender button is selected"""
        if self.male_button.is_over(pointer_pos):
            return "male"
        elif self.female_button.is_over(pointer_pos):
            return "female"
        return None

    def get_try_on_action(self, pointer_pos):
        """Check if any button in try-on screen is selected"""
        buttons = [
            self.prev_top_button, self.next_top_button,
            self.prev_bottom_button, self.next_bottom_button,
            self.save_button, self.rescan_button, self.exit_button
        ]

        for button in buttons:
            if button.is_over(pointer_pos):
                return button.action

        return None
