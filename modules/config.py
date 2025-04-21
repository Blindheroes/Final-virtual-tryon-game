"""
Configuration Settings for Virtual Try-On Game
"""

# Application settings
APP_TITLE = "Virtual Try-On Smart Mirror"
VERSION = "0.1.0 (MVP)"

# Camera settings
CAMERA_INDEX = 0  # Default camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIRROR_IMAGE = True  # Flip image horizontally to create mirror effect

# UI settings
UI_FONT = "Arial"
UI_FONT_SIZE = 24
UI_BUTTON_COLOR = (0, 120, 255)      # Default button color (BGR)
UI_BUTTON_HOVER = (0, 255, 0)        # Button hover color
UI_BUTTON_SELECT = (255, 0, 0)       # Button selection color
UI_TEXT_COLOR = (255, 255, 255)      # White text color
UI_BACKGROUND = (30, 30, 30)         # Dark gray background for overlay

# Scanning settings
SCAN_DURATION = 3.0  # seconds

# Hand tracking settings
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
MAX_HANDS = 1  # Track only one hand for simplicity
GESTURE_COOLDOWN = 0.5  # seconds before registering another gesture

# Clothing overlay settings
SHIRT_SCALE = 0.55
PANTS_SCALE = 2.0
VERTICAL_OFFSET = -50  # To position shirt properly

# Path settings (for future implementation with real clothing images)
CLOTHING_PATH = "clothing/"
TOPS_PATH = CLOTHING_PATH + "tops/"
BOTTOMS_PATH = CLOTHING_PATH + "bottoms/"

# Available clothing items (for future implementation)
CLOTHING_ITEMS = {
    "tops": {
        "male": ["polo.png", "tshirt.png", "shirt.png"],
        "female": ["blouse.png", "tshirt.png", "top.png"]
    },
    "bottoms": {
        "male": ["jeans.png", "shorts.png", "pants.png"],
        "female": ["skirt.png", "jeans.png", "pants.png"]
    }
}

# Email settings (for future implementation)
EMAIL_SERVER = "smtp.example.com"
EMAIL_PORT = 587
EMAIL_FROM = "virtual-tryon@example.com"
