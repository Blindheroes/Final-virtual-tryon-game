import cv2
import numpy as np

# Load webcam
cap = cv2.VideoCapture(0)

# UI Settings
window_width = 640
window_height = 480
background_color = (245, 245, 245)  # light gray


def draw_ui(frame):
    # Create transparent overlay
    overlay = frame.copy()

    # Semi-transparent background rectangle
    cv2.rectangle(overlay, (50, 30), (350, 670), background_color, -1)

    # Profile Circle
    center = (200, 100)
    radius = 40
    cv2.circle(overlay, center, radius, (200, 200, 200), -1)  # gray circle

    # "Choose your gender"
    cv2.putText(overlay, 'Choose your gender', (110, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Gender Buttons
    cv2.rectangle(overlay, (90, 210), (180, 250),
                  (0, 0, 0), -1)  # Male button (black)
    cv2.putText(overlay, '♂ Male', (100, 238),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(overlay, (210, 210), (310, 250),
                  (255, 255, 255), -1)  # Female button (white)
    cv2.putText(overlay, '♀ Female', (220, 238),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.rectangle(overlay, (210, 210), (310, 250), (0, 0, 0), 2)  # border

    # Main action button
    cv2.rectangle(overlay, (90, 290), (310, 340), (0, 0, 0), -1)
    cv2.putText(overlay, '📷 Scan Body', (110, 325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Divider "or try"
    cv2.line(overlay, (90, 380), (180, 380), (150, 150, 150), 2)
    cv2.putText(overlay, 'or try', (190, 385),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.line(overlay, (260, 380), (310, 380), (150, 150, 150), 2)

    # Option buttons
    cv2.rectangle(overlay, (90, 410), (310, 460), (255, 255, 255), -1)
    cv2.putText(overlay, '🎤 Voice Assistant', (110, 445),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.rectangle(overlay, (90, 480), (310, 530), (255, 255, 255), -1)
    cv2.putText(overlay, '✋ Camera Calibration', (100, 515),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Make overlay semi-transparent
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to consistent size
    frame = cv2.resize(frame, (window_width, window_height))

    # Draw UI elements
    frame = draw_ui(frame)

    # Show frame
    cv2.imshow('Main Menu Overlay', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
