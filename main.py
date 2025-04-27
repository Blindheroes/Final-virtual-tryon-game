"""
Virtual Try-On Game with Hand Gesture Control
Main application entry point
"""

import os
import cv2
import numpy as np
import time
import pygame

# Import custom modules
from ui_manager import UIManager
from hand_tracking import HandTracker
from body_classifier import BodyClassifier
from clothing_overlay import ClothingOverlay
from display_utils import create_vertical_display

# Initialize pygame for sound effects
pygame.init()
pygame.mixer.init()


class VirtualTryOnGame:
    """Main application class that controls the flow of the virtual try-on game"""

    def __init__(self):
        """Initialize the application"""
        self.camera_index = 0
        self.cap = None
        self.running = False
        self.current_screen = "calibration"
        self.gender = None
        self.body_type = None

        # Initialize component modules
        self.ui_manager = UIManager()
        self.hand_tracker = HandTracker()
        self.body_classifier = BodyClassifier()
        self.clothing_overlay = ClothingOverlay()

        # Window settings
        self.window_name = "Virtual Try-On"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Get screen dimensions for better positioning
        screen_width = 1280  # Default fallback width
        screen_height = 720  # Default fallback height

        try:
            # Try to get actual screen resolution
            from screeninfo import get_monitors
            screen = get_monitors()[0]
            screen_width = screen.width
            screen_height = screen.height
        except:
            pass

        # Set a reasonable window size (can be adjusted based on screen size)
        window_height = int(screen_height * 0.9)  # 90% of screen height
        # Maintain 9:16 aspect ratio
        window_width = int(window_height * (9/16))

        # cv2.resizeWindow(self.window_name, window_width, window_height)
        # Center horizontally
        cv2.moveWindow(self.window_name, (screen_width - window_width) // 2, 0)

        # Toggle fullscreen if needed
        self.fullscreen = False
        if self.fullscreen:
            cv2.setWindowProperty(
                self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Load audio
        self.sound_select = pygame.mixer.Sound(
            'UI/sounds/select.wav') if os.path.exists('UI/sounds/select.wav') else None

        # Selection cooldown to prevent multiple selections
        self.last_select_time = 0
        self.select_cooldown = 0.5  # seconds

        # Store window dimensions for coordinate mapping
        self.display_width = window_width
        self.display_height = window_height

    def initialize_camera(self):
        """Initialize the camera with vertical display ratio"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return False

        # Try to set camera to HD resolution if available
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Get the actual camera resolution
        cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized successfully at {cam_width}x{cam_height}")

        return True

    def process_hand_gestures(self, frame):
        """Process hand gestures and return pointer position and gesture state"""
        processed_frame = self.hand_tracker.process_frame(frame)

        # Get pointer position (index finger tip)
        pointer_pos = self.hand_tracker.get_pointer_position()

        # Scale pointer position to match display dimensions if needed
        if pointer_pos:
            h, w = frame.shape[:2]
            x, y = pointer_pos

            # # Map from camera coordinates to display coordinates
            # x = int(x * (self.display_width / w))
            # y = int(y * (self.display_height / h))
            # pointer_pos = (x, y)

        # Determine if selection gesture is being made
        is_selecting = self.hand_tracker.is_selecting()
        is_pointing = self.hand_tracker.is_pointing()
        is_open_palm = self.hand_tracker.is_open_palm()

        return processed_frame, pointer_pos, is_selecting, is_pointing, is_open_palm

    def handle_calibration_screen(self, frame, pointer_pos, is_selecting):
        """Handle interactions on the calibration screen"""
        ui_frame = self.ui_manager.draw_calibration_screen(frame, pointer_pos)

        # Check for calibration completion button press
        if is_selecting and pointer_pos:
            x, y = pointer_pos
            if self.ui_manager.is_within_button(x, y, "calibration_complete"):
                # Cooldown check to prevent multiple selections
                current_time = time.time()
                if current_time - self.last_select_time >= self.select_cooldown:
                    self.last_select_time = current_time
                    print("Calibration complete - moving to main menu")

                    # Play selection sound if available
                    if self.sound_select:
                        self.sound_select.play()

                    self.current_screen = "main_menu"

        return ui_frame

    def handle_main_menu(self, frame, pointer_pos, is_selecting):
        """Handle interactions on the main menu screen"""
        ui_frame = self.ui_manager.draw_main_menu(frame, pointer_pos)

        # Check for button presses
        if is_selecting and pointer_pos:
            x, y = pointer_pos
            current_time = time.time()

            if current_time - self.last_select_time >= self.select_cooldown:
                # Check which button was selected
                if self.ui_manager.is_within_button(x, y, "body_scan"):
                    self.last_select_time = current_time
                    print("Body Scan selected")
                    if self.sound_select:
                        self.sound_select.play()
                    self.current_screen = "gender_select"

                elif self.ui_manager.is_within_button(x, y, "voice_assistant"):
                    self.last_select_time = current_time
                    print("Voice Assistant selected")
                    if self.sound_select:
                        self.sound_select.play()
                    self.current_screen = "voice_assistant"

                elif self.ui_manager.is_within_button(x, y, "exit"):
                    self.last_select_time = current_time
                    print("Exit selected")
                    if self.sound_select:
                        self.sound_select.play()
                    self.running = False

        return ui_frame

    def handle_gender_select(self, frame, pointer_pos, is_selecting):
        """Handle gender selection screen"""
        ui_frame = self.ui_manager.draw_gender_select(frame, pointer_pos)

        # Check for button presses
        if is_selecting and pointer_pos:
            x, y = pointer_pos
            current_time = time.time()

            if current_time - self.last_select_time >= self.select_cooldown:
                # Check which gender was selected
                if self.ui_manager.is_within_button(x, y, "male"):
                    self.last_select_time = current_time
                    print("Male selected")
                    if self.sound_select:
                        self.sound_select.play()
                    self.gender = "male"
                    self.current_screen = "body_scan"

                elif self.ui_manager.is_within_button(x, y, "female"):
                    self.last_select_time = current_time
                    print("Female selected")
                    if self.sound_select:
                        self.sound_select.play()
                    self.gender = "female"
                    self.current_screen = "body_scan"

                elif self.ui_manager.is_within_button(x, y, "back"):
                    self.last_select_time = current_time
                    print("Back to main menu")
                    if self.sound_select:
                        self.sound_select.play()
                    self.current_screen = "main_menu"

        return ui_frame

    def handle_body_scan(self, frame, pointer_pos, is_selecting):
        """Handle body scan screen and classification"""
        # Classify body type using body classifier
        self.body_type = self.body_classifier.process_frame(frame)

        ui_frame = self.ui_manager.draw_body_scan(
            frame, pointer_pos, self.body_type)

        # Check for button presses
        if is_selecting and pointer_pos:
            x, y = pointer_pos
            current_time = time.time()

            if current_time - self.last_select_time >= self.select_cooldown:
                if self.ui_manager.is_within_button(x, y, "continue"):
                    self.last_select_time = current_time
                    print(f"Body type detected: {self.body_type}")
                    if self.sound_select:
                        self.sound_select.play()
                    self.current_screen = "virtual_tryon"

                elif self.ui_manager.is_within_button(x, y, "back"):
                    self.last_select_time = current_time
                    print("Back to gender select")
                    if self.sound_select:
                        self.sound_select.play()
                    self.current_screen = "gender_select"

        return ui_frame

    def handle_voice_assistant(self, frame, pointer_pos, is_selecting):
        """Handle voice assistant screen"""
        ui_frame = self.ui_manager.draw_voice_assistant(frame, pointer_pos)

        # Voice assistant functionality would be implemented here
        # For now, we'll just provide a UI to proceed to virtual try-on

        # Check for button presses
        if is_selecting and pointer_pos:
            x, y = pointer_pos
            current_time = time.time()

            if current_time - self.last_select_time >= self.select_cooldown:
                if self.ui_manager.is_within_button(x, y, "continue"):
                    self.last_select_time = current_time
                    print("Voice assistant completed")
                    if self.sound_select:
                        self.sound_select.play()
                    # For demo purposes, set default values
                    self.body_type = "ideal"
                    self.gender = "male" if not self.gender else self.gender
                    self.current_screen = "virtual_tryon"

                elif self.ui_manager.is_within_button(x, y, "back"):
                    self.last_select_time = current_time
                    print("Back to main menu")
                    if self.sound_select:
                        self.sound_select.play()
                    self.current_screen = "main_menu"

        return ui_frame

    def handle_virtual_tryon(self, frame, pointer_pos, is_selecting):
        """Handle virtual try-on screen"""
        # Apply clothing overlay based on body type and gender
        tryon_frame = self.clothing_overlay.apply_clothing(
            frame,
            self.body_type,
            self.gender
        )

        ui_frame = self.ui_manager.draw_virtual_tryon(tryon_frame, pointer_pos)

        # Check for button presses
        if is_selecting and pointer_pos:
            x, y = pointer_pos
            current_time = time.time()

            if current_time - self.last_select_time >= self.select_cooldown:
                if self.ui_manager.is_within_button(x, y, "main_menu"):
                    self.last_select_time = current_time
                    print("Back to main menu")
                    if self.sound_select:
                        self.sound_select.play()
                    self.current_screen = "main_menu"

                elif self.ui_manager.is_within_button(x, y, "recalibrate"):
                    self.last_select_time = current_time
                    print("Back to calibration")
                    if self.sound_select:
                        self.sound_select.play()
                    self.current_screen = "calibration"

                elif self.ui_manager.is_within_button(x, y, "exit"):
                    self.last_select_time = current_time
                    print("Exit selected")
                    if self.sound_select:
                        self.sound_select.play()
                    self.running = False

        return ui_frame

    def run(self):
        """Main application loop"""
        if not self.initialize_camera():
            return

        self.running = True
        last_time = time.time()
        fps_values = []

        while self.running:
            # Calculate FPS
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            fps = 1.0 / delta_time if delta_time > 0 else 0
            fps_values.append(fps)
            if len(fps_values) > 30:  # Average over 30 frames
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)

            # Capture frame from camera
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame from camera")
                break

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for processing (MediaPipe requires RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create vertical display (9:16 aspect ratio)
            vertical_frame = create_vertical_display(frame)
            # vertical_frame = frame

            # Save original dimensions before processing for coordinate mapping
            frame_h, frame_w = vertical_frame.shape[:2]

            # Process hand gestures
            processed_frame, pointer_pos, is_selecting, is_pointing, is_open_palm = self.process_hand_gestures(
                vertical_frame)

            # Handle current screen
            if self.current_screen == "calibration":
                ui_frame = self.handle_calibration_screen(
                    vertical_frame, pointer_pos, is_selecting)
            elif self.current_screen == "main_menu":
                ui_frame = self.handle_main_menu(
                    vertical_frame, pointer_pos, is_selecting)
            elif self.current_screen == "gender_select":
                ui_frame = self.handle_gender_select(
                    vertical_frame, pointer_pos, is_selecting)
            elif self.current_screen == "body_scan":
                ui_frame = self.handle_body_scan(
                    vertical_frame, pointer_pos, is_selecting)
            elif self.current_screen == "voice_assistant":
                ui_frame = self.handle_voice_assistant(
                    vertical_frame, pointer_pos, is_selecting)
            elif self.current_screen == "virtual_tryon":
                ui_frame = self.handle_virtual_tryon(
                    vertical_frame, pointer_pos, is_selecting)
            else:
                ui_frame = vertical_frame

            # Add FPS counter for debugging (optional)
            cv2.putText(ui_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display hand position (optional)
            if pointer_pos:
                cv2.circle(ui_frame, pointer_pos, 10, (0, 255, 0), 2)

            # Display the frame without resizing
            cv2.imshow(self.window_name, ui_frame)

            # Toggle fullscreen with 'f' key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('f'):
                pass
                # self.fullscreen = not self.fullscreen
                # if self.fullscreen:
                #     cv2.setWindowProperty(
                #         self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                # else:
                #     cv2.setWindowProperty(
                #         self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            # Exit on 'q' key press
            elif key == ord('q'):
                break

        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    app = VirtualTryOnGame()
    app.run()
