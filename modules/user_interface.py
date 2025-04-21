"""
User Interface Module for Virtual Try-On Game
This module handles the UI elements and screens for the virtual try-on application.
"""

import cv2
import numpy as np

class Button:
    def __init__(self, x, y, width, height, text, action=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.action = action
        
    def is_over(self, pos):
        """Check if position is over the button"""
        if pos is None:
            return False
            
        x, y = pos
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
                
    def draw(self, frame, pointer_pos=None, is_selecting=False):
        """Draw the button on the frame"""
        # Change color if pointer is over the button
        color = (0, 120, 255)  # Default color
        if pointer_pos and self.is_over(pointer_pos):
            color = (0, 255, 0)  # Highlight color
            if is_selecting:
                color = (255, 0, 0)  # Selection color
                
        # Draw button rectangle
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     color, -1)
        
        # Draw button text
        text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        cv2.putText(frame, self.text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

class UserInterface:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Initialize UI elements for different screens
        self.init_welcome_screen()
        self.init_gender_selection()
        self.init_try_on_screen()
        
    def init_welcome_screen(self):
        """Initialize welcome screen UI elements"""
        self.welcome_title = "Welcome to Virtual Try-On"
        self.welcome_subtitle = "Point your index finger and select with pinky+index to begin"
        self.start_button = Button(
            self.width // 2 - 100, 
            self.height // 2 + 50, 
            200, 60, "Start", "start"
        )
        
    def init_gender_selection(self):
        """Initialize gender selection screen UI elements"""
        self.gender_title = "Select Your Gender"
        self.male_button = Button(
            self.width // 4 - 75, 
            self.height // 2, 
            150, 60, "Male", "male"
        )
        self.female_button = Button(
            3 * self.width // 4 - 75, 
            self.height // 2, 
            150, 60, "Female", "female"
        )
        
    def init_try_on_screen(self):
        """Initialize try-on screen UI elements"""
        # Navigation buttons
        self.prev_top_button = Button(50, self.height - 150, 100, 50, "< Top", "prev_top")
        self.next_top_button = Button(170, self.height - 150, 100, 50, "Top >", "next_top")
        
        self.prev_bottom_button = Button(50, self.height - 80, 100, 50, "< Bottom", "prev_bottom")
        self.next_bottom_button = Button(170, self.height - 80, 100, 50, "Bottom >", "next_bottom")
        
        # Action buttons
        button_width = 120
        button_spacing = 20
        total_width = 3 * button_width + 2 * button_spacing
        start_x = self.width - total_width - 50
        
        self.save_button = Button(
            start_x, 
            self.height - 80, 
            button_width, 50, "Save", "save"
        )
        self.rescan_button = Button(
            start_x + button_width + button_spacing, 
            self.height - 80, 
            button_width, 50, "Rescan", "rescan"
        )
        self.exit_button = Button(
            start_x + 2 * (button_width + button_spacing), 
            self.height - 80, 
            button_width, 50, "Exit", "exit"
        )
        
    def draw_welcome_screen(self, frame):
        """Draw the welcome screen"""
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw title
        cv2.putText(frame, self.welcome_title, 
                   (self.width // 2 - 200, self.height // 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Draw subtitle
        cv2.putText(frame, self.welcome_subtitle, 
                   (self.width // 2 - 300, self.height // 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Draw start button
        self.start_button.draw(frame)
        
    def draw_gender_selection(self, frame, pointer_pos, is_selecting):
        """Draw the gender selection screen"""
        # Add semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw title
        cv2.putText(frame, self.gender_title, 
                   (self.width // 2 - 150, self.height // 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Draw gender buttons
        self.male_button.draw(frame, pointer_pos, is_selecting)
        self.female_button.draw(frame, pointer_pos, is_selecting)
        
    def draw_scanning_screen(self, frame, scan_complete):
        """Draw the body scanning screen"""
        # Add progress indicator
        if not scan_complete:
            # Draw scanning animation
            cv2.putText(frame, "Scanning Body...", 
                       (self.width // 2 - 150, self.height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Add scanning effect (horizontal line moving up and down)
            scan_line_y = int((self.height // 2) + np.sin(cv2.getTickCount() * 0.0000025) * (self.height // 4))
            cv2.line(frame, (0, scan_line_y), (self.width, scan_line_y), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Scan Complete!", 
                       (self.width // 2 - 150, self.height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
    def draw_try_on_screen(self, frame, pointer_pos, is_selecting):
        """Draw the try-on screen with clothing navigation controls"""
        # Draw clothing navigation buttons
        self.prev_top_button.draw(frame, pointer_pos, is_selecting)
        self.next_top_button.draw(frame, pointer_pos, is_selecting)
        self.prev_bottom_button.draw(frame, pointer_pos, is_selecting)
        self.next_bottom_button.draw(frame, pointer_pos, is_selecting)
        
        # Draw action buttons
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