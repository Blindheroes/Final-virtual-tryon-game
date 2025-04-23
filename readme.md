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
  - Body type scanning (simulated in MVP so just random [thin, fat, obesity])
  - Virtual clothing try-on
  - Save results to email (simulated in MVP)
  - Option to rescan or exit

- **Clothing Options**:
  - Real-time overlay on user's body

## Technologies Used

- **Python**: Main programming language
- **OpenCV**: Computer vision for video processing and image manipulation
- **MediaPipe**: Hand tracking and gesture recognition
- **PyGame**: GUI and interactive elements

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/username/virtual-tryon-game.git
   cd virtual-tryon-game
   ```

2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Ensure you have a camera (webcam) connected to your computer

4. Run the application:
   ```
   python main.py
   ```

## Project Structure

```
|-- main.py                  # Entry point for the application
|-- requirements.txt         # Dependencies list
|-- game_flow.mermaid        # Game flow diagram
|-- project_structure.mermaid # Project structure diagram
|-- readme.md                # This documentation
|-- clothing/                # Clothing asset directory
|   |-- tops/                # Top clothing assets
|   |   |-- male/            # Male tops
|   |   |-- female/          # Female tops
|   |-- bottoms/             # Bottom clothing assets
|       |-- male/            # Male bottoms
|       |-- female/          # Female bottoms
|-- fonts/                   # Custom Montserrat font files
|-- modules/                 # Core modules
    |-- __init__.py          # Module initialization
    |-- body_scanner.py      # Body scanning functionality
    |-- clothing_overlay.py  # Virtual try-on overlay
    |-- config.py            # Configuration settings
    |-- email_service.py     # Email service for saving results
    |-- hand_tracking.py     # Hand gesture control
    |-- user_interface.py    # User interface elements
```

## How to Use

1. Launch the application and stand in front of the camera, be sure your hand and body are visible
2. Use your index finger as a pointer to navigate
3. Extend both index and little fingers to select (like a left click on mouse) an option
4. Follow the on-screen prompts to select gender and wait sytem to scan your body type
5. rescan if you want to
6. Save your results, back to menu, or exit when finished

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
