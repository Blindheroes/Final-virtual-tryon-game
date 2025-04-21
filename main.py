"""
Virtual Try-On Game with Hand Gesture Control - Main Application
This is the entry point for the virtual try-on application.
"""

import cv2
import pygame
import sys
import time
from modules.hand_tracking import HandTracker
from modules.user_interface import UserInterface
from modules.clothing_overlay import ClothingOverlay
from modules.body_scanner import BodyScanner

class VirtualTryOnGame:
    def __init__(self):
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Error: Could not open camera.")
            sys.exit()
            
        # Get camera dimensions
        _, frame = self.camera.read()
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.frame_width, self.frame_height))
        pygame.display.set_caption("Virtual Try-On Smart Mirror")
        
        # Initialize modules
        self.hand_tracker = HandTracker()
        self.ui = UserInterface(self.frame_width, self.frame_height)
        self.clothing_overlay = ClothingOverlay()
        self.body_scanner = BodyScanner()
        
        # Game state variables
        self.current_screen = "welcome"  # welcome, gender_selection, scanning, try_on
        self.gender = None
        self.body_type = None
        self.selected_clothing = {"top": None, "bottom": None}
        self.is_running = True
        
        # Finger positions
        self.pointer_pos = (0, 0)
        self.is_selecting = False
        
        # For tracking transitions
        self.last_action_time = time.time()
        self.action_cooldown = 0.5  # seconds

    def process_frame(self, frame):
        """Process camera frame and detect hand gestures"""
        # Convert frame to RGB for hand tracking
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Track hands and get gesture info
        self.hand_tracker.process_frame(rgb_frame)
        self.pointer_pos = self.hand_tracker.get_pointer_position()
        self.is_selecting = self.hand_tracker.is_selecting()
        
        # Process based on current screen
        if self.current_screen == "welcome":
            self.ui.draw_welcome_screen(frame)
            
        elif self.current_screen == "gender_selection":
            self.ui.draw_gender_selection(frame, self.pointer_pos, self.is_selecting)
            
        elif self.current_screen == "scanning":
            scan_complete = self.body_scanner.scan(frame)
            self.ui.draw_scanning_screen(frame, scan_complete)
            
            if scan_complete:
                self.body_type = self.body_scanner.get_body_type()
                self.current_screen = "try_on"
                
        elif self.current_screen == "try_on":
            # Apply clothing overlay
            frame = self.clothing_overlay.apply_clothing(
                frame, 
                self.gender, 
                self.body_type, 
                self.selected_clothing
            )
            
            # Draw UI elements for try-on screen
            self.ui.draw_try_on_screen(frame, self.pointer_pos, self.is_selecting)
        
        # Draw pointer
        if self.pointer_pos:
            cv2.circle(frame, self.pointer_pos, 10, (0, 255, 0), -1)
            if self.is_selecting:
                cv2.circle(frame, self.pointer_pos, 15, (0, 0, 255), 2)
        
        return frame
            
    def check_ui_interactions(self):
        """Check for UI button interactions"""
        current_time = time.time()
        if self.is_selecting and (current_time - self.last_action_time) > self.action_cooldown:
            self.last_action_time = current_time
            
            if self.current_screen == "welcome":
                self.current_screen = "gender_selection"
                
            elif self.current_screen == "gender_selection":
                selected = self.ui.get_gender_selection(self.pointer_pos)
                if selected:
                    self.gender = selected
                    self.current_screen = "scanning"
                    self.body_scanner.start_scan()
                    
            elif self.current_screen == "try_on":
                action = self.ui.get_try_on_action(self.pointer_pos)
                if action == "next_top":
                    self.clothing_overlay.next_top()
                elif action == "prev_top":
                    self.clothing_overlay.prev_top()
                elif action == "next_bottom":
                    self.clothing_overlay.next_bottom()
                elif action == "prev_bottom":
                    self.clothing_overlay.prev_bottom()
                elif action == "save":
                    # Save functionality would go here
                    pass
                elif action == "rescan":
                    self.current_screen = "scanning"
                    self.body_scanner.start_scan()
                elif action == "exit":
                    self.is_running = False
    
    def convert_frame_to_pygame(self, frame):
        """Convert OpenCV frame to PyGame surface"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)  # Mirror image for "mirror" effect
        return pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    
    def run(self):
        """Main game loop"""
        while self.is_running:
            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False
                    
            # Read camera frame
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
                
            # Process frame (hand tracking, UI updates)
            processed_frame = self.process_frame(frame)
            
            # Check for UI interactions
            self.check_ui_interactions()
            
            # Convert OpenCV frame to PyGame surface and display
            pygame_frame = self.convert_frame_to_pygame(processed_frame)
            self.screen.blit(pygame_frame, (0, 0))
            pygame.display.flip()
            
        # Clean up
        self.camera.release()
        pygame.quit()

if __name__ == "__main__":
    try:
        game = VirtualTryOnGame()
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit(1)
