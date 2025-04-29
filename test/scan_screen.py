import cv2
import numpy as np


def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness, radius=20):
    """Gambar rounded rectangle manual."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    if thickness < 0:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, -1)
        cv2.circle(img, (x1+radius, y1+radius), radius, color, -1)
        cv2.circle(img, (x2-radius, y1+radius), radius, color, -1)
        cv2.circle(img, (x1+radius, y2-radius), radius, color, -1)
        cv2.circle(img, (x2-radius, y2-radius), radius, color, -1)
    else:
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius),
                    180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius),
                    270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+radius, y2-radius),
                    (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y2-radius),
                    (radius, radius), 0, 0, 90, color, thickness)


# Buka webcam
cap = cv2.VideoCapture(0)

# Set ukuran frame
frame_width = 640
frame_height = 480

# Warna
white = (255, 255, 255)
black = (0, 0, 0)
dark_bg = (30, 30, 30)
gray = (150, 150, 150)

# Font
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    # Overlay
    overlay = frame.copy()

    # Background area gelap (scan area)
    scan_top = 80
    scan_bottom = 380
    cv2.rectangle(overlay, (0, scan_top),
                  (frame_width, scan_bottom), dark_bg, -1)

    # Frame deteksi badan (kotak abu-abu)
    scan_w = 250
    scan_h = 300
    center_x = frame_width // 2
    center_y = (scan_top + scan_bottom) // 2
    scan_top_left = (center_x - scan_w//2, center_y - scan_h//2)
    scan_bottom_right = (center_x + scan_w//2, center_y + scan_h//2)
    cv2.rectangle(overlay, scan_top_left, scan_bottom_right, gray, 2)

    # Tombol Capture di atas (rounded rectangle)
    button_top_left = (20, 20)
    button_bottom_right = (frame_width - 20, 60)
    draw_rounded_rectangle(overlay, button_top_left,
                           button_bottom_right, black, thickness=-1, radius=20)
    cv2.putText(overlay, "Capture", (frame_width//2 - 50, 50),
                font, 0.7, white, 2, cv2.LINE_AA)

    # Panel bawah untuk instruksi
    panel_height = 80
    cv2.rectangle(overlay, (0, frame_height - panel_height),
                  (frame_width, frame_height), black, -1)

    # Text instruksi
    instruction_1 = "Stand 6 feet away from camera"
    instruction_2 = "Ensure your full body is visible"
    cv2.putText(overlay, instruction_1, (30, frame_height - panel_height + 30),
                font, 0.45, white, 1, cv2.LINE_AA)
    cv2.putText(overlay, instruction_2, (30, frame_height - panel_height + 60),
                font, 0.45, white, 1, cv2.LINE_AA)

    # Gabungkan overlay ke frame asli dengan transparansi
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Tampilkan frame
    cv2.imshow("Scan Screen", frame)

    # Tombol keluar (ESC)
    key = cv2.waitKey(1)
    if key == 27:
        break

# Bersih-bersih
cap.release()
cv2.destroyAllWindows()
