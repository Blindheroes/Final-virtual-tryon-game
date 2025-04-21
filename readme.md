# Virtual Try-On Game with Hand Gesture Control

A smart mirror application that allows users to virtually try on different clothing items using hand gestures as a control interface.

## Overview

This project implements a virtual try-on system that uses a camera for input, hand tracking for gesture control, and displays the virtual try-on results on a mirror-like interface. Users can navigate through the application using simple hand gestures and try on different clothing items.

## Features

- **Hand Gesture Control**:
  - Index finger as pointer
  - Index + little finger for selection

- **User Flow**:
  - Welcome screen
  - Gender selection
  - Body type scanning (simulated in MVP)
  - Virtual clothing try-on
  - Navigation through different clothing options
  - Save results to email (simulated in MVP)
  - Option to rescan or exit

- **Clothing Options**:
  - Multiple tops and bottoms for both male and female
  - Real-time overlay on user's body

## Technologies Used

- **Python**: Main programming language
- **OpenCV**: Computer vision for video processing and image manipulation
- **MediaPipe**: Hand tracking and gesture recognition
- **PyGame**: GUI and interactive elements

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/virtual-tryon-game.git
   cd virtual-tryon-game
   ```

2. Install required packages:
   ```
   pip install opencv-python mediapipe pygame numpy
   ```

3. Run the application:
   ```
   python main.py
   ```

## Project Structure

- `main.py`: Application entry point and main loop
- `modules/`:
  - `hand_tracking.py`: Hand gesture detection and interaction
  - `body_scanner.py`: Body type recognition (simulated)
  - `clothing_overlay.py`: Virtual try-on functionality
  - `user_interface.py`: UI components and screens
  - `email_service.py`: Email functionality (simulated)
  - `config.py`: Configuration settings

## How to Use

1. Launch the application and stand in front of the camera
2. Use your index finger as a pointer to navigate
3. Extend both index and little fingers to select an option
4. Follow the on-screen prompts to select gender and try on different clothing items
5. Use the navigation buttons to cycle through different clothing options
6. Save your results or exit when finished

## Notes for MVP

This is a Minimum Viable Product (MVP) with the following limitations:
- Body scanning is simulated (no actual measurements)
- Clothing items are simplified placeholders
- Email functionality is simulated (not actually sending emails)
- Limited clothing options

## Future Enhancements

- Advanced body measurements using depth cameras
- Realistic clothing textures and physics
- Multiple hand gesture support
- User profiles and preferences storage
- Social media sharing
- Expanded clothing catalog
- Voice control integration

## Requirements

- Camera (webcam or built-in camera)
- Display (monitor or smart mirror setup)
- Python 3.7+
- Sufficient lighting for hand gesture recognition
