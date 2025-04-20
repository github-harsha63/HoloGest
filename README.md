Hand Gesture-Controlled Automation System
This project is a Python-based system that uses hand gestures captured through a webcam to control various system functions like mouse movement, clicks, scrolling, screenshots, and more.

Features
Real-time hand detection and tracking using MediaPipe
Custom gesture data collection system
Machine learning-based gesture recognition
System automation using PyAutoGUI
Supports 15+ predefined gestures
Easy to extend with new gestures
Requirements
Python 3.7+
Webcam
Required libraries:
mediapipe
opencv-python
numpy
tensorflow
pyautogui
pandas
scikit-learn
matplotlib
joblib
Installation
Clone or download this repository
Install the required dependencies:
bash
pip install mediapipe opencv-python numpy tensorflow pyautogui pandas scikit-learn matplotlib joblib
Project Structure
hand-gesture-control/
│
├── hand_detector.py       # Hand detection and landmark extraction module
├── gesture_collector.py   # Data collection system for gathering gesture samples
├── gesture_trainer.py     # Machine learning model for gesture recognition
├── gesture_controller.py  # Real-time gesture controller with system actions
├── main.py                # Main application with menu interface
│
├── gesture_data/          # Directory for collected gesture data
│   └── gesture_data.csv   # CSV file with collected gesture landmarks
│
├── models/                # Directory for trained models
│   ├── gesture_model.h5   # Trained TensorFlow model
│   └── label_encoder.pkl  # Label encoder for gesture names
│
└── screenshots/           # Directory for saved screenshots
Usage
Step 1: Run the main application
bash
python main.py
Step 2: Collect Gesture Data
Select option 1 from the main menu
You can collect data for a specific gesture or all gestures
Position your hand in front of the camera
Press 's' to start collecting samples
Perform the gesture multiple times (50-100 samples recommended per gesture)
Step 3: Train Recognition Model
Select option 2 from the main menu
Set training parameters (or use defaults)
Wait for the model to train
The model will be saved automatically
Step 4: Run Gesture Control System
Select option 3 from the main menu
Position your hand in front of the camera
Perform gestures to control your system
Use the defined "exit_program" gesture or press 'q' to stop
Supported Gestures
move_right: Move mouse cursor right
move_left: Move mouse cursor left
move_up: Move mouse cursor up
move_down: Move mouse cursor down
left_click: Perform a left mouse click
right_click: Perform a right mouse click
select: Select all (Ctrl+A)
deselect: Deselect by clicking
scroll_up: Scroll up
scroll_down: Scroll down
screenshot: Take a screenshot and save it
switch_window: Switch between windows (Alt+Tab)
switch_tab: Switch between browser tabs (Ctrl+Tab)
zoom_in: Zoom in (Ctrl++)
zoom_out: Zoom out (Ctrl+-)
exit_program: Stop the gesture control system
idle: No action (default when no gesture is detected)
Adding New Gestures
Add the new gesture name to the self.gestures list in gesture_collector.py
Collect data for the new gesture
Add a corresponding action method in gesture_controller.py
Add the action to the self.actions dictionary
Retrain the model
Tips for Better Recognition
Ensure good lighting conditions
Keep a consistent distance from the camera
Make clear, distinct gestures
Collect a good amount of training data (at least 50-100 samples per gesture)
Try different model architectures or hyperparameters if recognition accuracy is low
Troubleshooting
Low recognition accuracy: Collect more data samples, try different gestures that are more distinct from each other
Slow performance: Reduce the webcam resolution or frame rate in the controller code
Hand not detected: Improve lighting conditions, ensure your hand is clearly visible in the camera frame
