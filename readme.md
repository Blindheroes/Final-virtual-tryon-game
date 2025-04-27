# Virtual Try-On Game with Hand Gesture Control

A smart mirror application that allows users to virtually try on clothing based on body type or voice asistent using hand gestures as a control interface.

## Overview

This project implements a virtual try-on system that uses a camera for input, hand tracking for gesture control, and displays the virtual try-on results on a mirror-like interface. Users can navigate through the application using simple hand gestures. user can try clothing based on recomendation

## Features

- **Hand Gesture Control**:
  - Index finger as pointer
  - Index + little finger for selection

- **Vertical Display**:
  - display captured ration 9:16 by changing the width

- **body type recomendation**:
  - before start scan chose gender first
  - scan body

- **voice asistent recomendation**:
  - 

- **User Flow**:
  - calibration
  - main menu
  - body scan or voice asistent 
  - scan screen or voice screen
  - virtual tryon recomended apparel based on body type (scan) or voice asistent
  - back to main menu, calibration, or exit


## Technologies Used

- **Python**: Main programming language
- **OpenCV**: Computer vision for video processing, image manipulation, and GUI
- **MediaPipe**: Hand tracking and gesture recognition

## Reference
 - display ration: reference\display ratio.py
 - body type: reference\Body Classification.py
 - virtual tryon: reference\Body Classification.py
 - hand gesture controll: reference\hand_tracking.py

## Project Structure


## How to Use


## Notes for MVP


## Requirements

- Camera (webcam or built-in camera)
- Display (monitor or smart mirror setup)
- Python 3.7+
- Sufficient lighting for hand gesture recognition