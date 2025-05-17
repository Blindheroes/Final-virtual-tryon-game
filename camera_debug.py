import cv2

# Open the default camera
camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Error: Could not open the camera.")
else:
    # Get the resolution of the camera
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {width}x{height}")

    # Get additional camera details
    fps = camera.get(cv2.CAP_PROP_FPS)
    brightness = camera.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = camera.get(cv2.CAP_PROP_CONTRAST)
    saturation = camera.get(cv2.CAP_PROP_SATURATION)
    hue = camera.get(cv2.CAP_PROP_HUE)

    print(f"Frames per second: {fps}")
    print(f"Brightness: {brightness}")
    print(f"Contrast: {contrast}")
    print(f"Saturation: {saturation}")
    print(f"Hue: {hue}")

    # video capture loop
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera
camera.release()