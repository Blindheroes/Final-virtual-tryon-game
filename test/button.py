import cv2
import numpy as np

class Button:
    def __init__(self, x, y, width, height, text, radius=15, color=(0, 120, 255), text_color=(255, 255, 255)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.radius = radius
        self.color = color
        self.text_color = text_color
        self.hover = False
        self.clicked = False

    def draw(self, img):
        # Create button background with rounded corners
        color = (self.color[0] + 30, self.color[1] + 30, self.color[2] + 30) if self.hover else self.color
        
        # Create a mask for rounded corners
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Draw filled white rectangle on mask
        cv2.rectangle(mask, (self.radius, 0), (self.width - self.radius, self.height), 255, -1)
        cv2.rectangle(mask, (0, self.radius), (self.width, self.height - self.radius), 255, -1)
        
        # Draw the four corner circles on mask
        cv2.circle(mask, (self.radius, self.radius), self.radius, 255, -1)
        cv2.circle(mask, (self.width - self.radius, self.radius), self.radius, 255, -1)
        cv2.circle(mask, (self.radius, self.height - self.radius), self.radius, 255, -1)
        cv2.circle(mask, (self.width - self.radius, self.height - self.radius), self.radius, 255, -1)
        
        # Extract region of interest from the original image
        roi = img[self.y:self.y + self.height, self.x:self.x + self.width]
        
        # Create colored button image
        button = np.ones((self.height, self.width, 3), dtype=np.uint8)
        button[:] = color
        
        # Apply mask to button
        masked_button = cv2.bitwise_and(button, button, mask=mask)
        
        # Create inverse mask
        mask_inv = cv2.bitwise_not(mask)
        
        # Get background from ROI using inverse mask
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        
        # Add button and background together
        dst = cv2.add(roi_bg, masked_button)
        
        # Copy result back to original image
        img[self.y:self.y + self.height, self.x:self.x + self.width] = dst
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_size = cv2.getTextSize(self.text, font, font_scale, 2)[0]
        text_x = int((self.width - text_size[0]) / 2)
        text_y = int((self.height + text_size[1]) / 2)
        
        cv2.putText(img, self.text, (self.x + text_x, self.y + text_y), 
                   font, font_scale, self.text_color, 2)
        
    def check_hover(self, mouse_pos):
        # Check if mouse is over button
        if (self.x < mouse_pos[0] < self.x + self.width and
            self.y < mouse_pos[1] < self.y + self.height):
            self.hover = True
            return True
        else:
            self.hover = False
            return False
    
    def check_click(self, mouse_pos, mouse_click):
        # Check if button is clicked
        if (self.x < mouse_pos[0] < self.x + self.width and
            self.y < mouse_pos[1] < self.y + self.height and mouse_click):
            self.clicked = True
            return True
        else:
            self.clicked = False
            return False

# Demo code to test the button
def main():
    # Create a window
    window_name = "OpenCV Rounded Button Demo"
    cv2.namedWindow(window_name)
    
    # Initialize mouse variables
    mouse_pos = (0, 0)
    mouse_clicked = False
    
    # Create a button
    button1 = Button(50, 50, 200, 60, "Click Me!")
    button2 = Button(50, 150, 200, 60, "Exit", color=(220, 50, 50))
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_pos, mouse_clicked
        mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_clicked = True
        else:
            mouse_clicked = False
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Main loop
    while True:
        # Create a blank image
        img = np.ones((300, 300, 3), dtype=np.uint8) * 240  # light gray background
        
        # Check button state
        button1.check_hover(mouse_pos)
        button2.check_hover(mouse_pos)
        
        if button1.check_click(mouse_pos, mouse_clicked):
            print("Button 1 clicked!")
            
        if button2.check_click(mouse_pos, mouse_clicked):
            print("Exit button clicked!")
            break
        
        # Draw buttons
        button1.draw(img)
        button2.draw(img)
        
        # Reset mouse_clicked
        mouse_clicked = False
        
        # Show the image
        cv2.imshow(window_name, img)
        
        # Break the loop if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()