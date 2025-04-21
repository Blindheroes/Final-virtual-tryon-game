"""
Email Service Module for Virtual Try-On Game
This module handles saving try-on results to email.
For MVP, this is just a placeholder implementation.
"""

import cv2
import os
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from modules.config import EMAIL_SERVER, EMAIL_PORT, EMAIL_FROM

class EmailService:
    def __init__(self):
        """Initialize email service"""
        self.server = EMAIL_SERVER
        self.port = EMAIL_PORT
        self.sender = EMAIL_FROM
        
        # Create temp directory for saved images if it doesn't exist
        self.temp_dir = "temp_images"
        if not os.path.exists(self.temp_dir):
            try:
                os.makedirs(self.temp_dir)
            except Exception as e:
                print(f"Error creating temp directory: {e}")
                self.temp_dir = ""
    
    def save_image(self, frame):
        """
        Save the current try-on image to a file
        
        Args:
            frame: The current frame with clothing overlay
            
        Returns:
            Path to the saved image or None if saving failed
        """
        if not self.temp_dir:
            return None
            
        try:
            # Generate a unique filename using timestamp
            filename = f"tryon_{int(time.time())}.jpg"
            filepath = os.path.join(self.temp_dir, filename)
            
            # Save the image
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            print(f"Error saving image: {e}")
            return None
    
    def send_email(self, email_address, image_path, clothing_info):
        """
        Send email with try-on image
        For MVP, just simulate sending (print to console)
        
        Args:
            email_address: Recipient email address
            image_path: Path to the saved image
            clothing_info: Dict with clothing information
            
        Returns:
            Boolean indicating success
        """
        # For MVP, just print the info (simulate sending)
        print("\n--- EMAIL SIMULATION ---")
        print(f"To: {email_address}")
        print(f"From: {self.sender}")
        print(f"Subject: Your Virtual Try-On Results")
        print(f"Image: {image_path}")
        print(f"Clothing: {clothing_info}")
        print("--- END EMAIL SIMULATION ---\n")
        
        return True
        
        # In a real implementation, would use smtplib like this:
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = email_address
            msg['Subject'] = "Your Virtual Try-On Results"
            
            # Add text
            text = f"Thank you for using our Virtual Try-On Service!\n\n"
            text += f"You tried on: {clothing_info['top']} (top) and {clothing_info['bottom']} (bottom)\n\n"
            text += "We hope you enjoyed the experience!"
            msg.attach(MIMEText(text))
            
            # Add image
            with open(image_path, 'rb') as f:
                img_data = f.read()
                image = MIMEImage(img_data, name=os.path.basename(image_path))
                msg.attach(image)
            
            # Connect to server and send
            with smtplib.SMTP(self.server, self.port) as server:
                server.starttls()
                # server.login(username, password)  # Would need credentials in a real app
                server.send_message(msg)
                
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
        """
    
    def get_email(self, frame):
        """
        Display email input screen and get email address
        For MVP, just return a placeholder email
        
        Args:
            frame: Current camera frame for displaying UI
            
        Returns:
            Email address string
        """
        # For MVP, just return a placeholder email
        return "user@example.com"
        
        # In a real implementation, would show a keyboard UI for input
