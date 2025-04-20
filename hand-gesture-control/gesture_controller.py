# gesture_controller.py

import cv2
import numpy as np
import time
import os
import pyautogui
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
import joblib
import threading
from hand_detector import HandDetector

# Configure PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

class GestureController:
    def __init__(self, model_dir="models", confidence_threshold=0.7):
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        self.detector = HandDetector(min_detection_confidence=0.7)
        
        # Load model and label encoder
        self.model_path = os.path.join(model_dir, "gesture_model.h5")
        self.encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Model or label encoder not found in {model_dir}")
        
        self.model = load_model(self.model_path)
        self.label_encoder = joblib.load(self.encoder_path)
        
        self.actions = {
            "move_right": self._move_mouse_right,
            "move_left": self._move_mouse_left,
            "move_up": self._move_mouse_up,
            "move_down": self._move_mouse_down,
            "left_click": self._left_click,
            "right_click": self._right_click,
            "select": self._select,
            "deselect": self._deselect,
            "scroll_up": self._scroll_up,
            "scroll_down": self._scroll_down,
            "screenshot": self._take_screenshot,
            "switch_window": self._switch_window,
            "switch_tab": self._switch_tab,
            "zoom_in": self._zoom_in,
            "zoom_out": self._zoom_out,
            "exit_program": self._exit_program,
            "idle": self._idle
        }
        
        self.running = False
        self.exit_requested = False
        self.current_gesture = "idle"
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 0.5  # seconds between gesture actions
        
        # For smoothing: keep track of recent predictions
        self.gesture_history = []
        self.history_length = 5
        
        print(f"Loaded model with {len(self.label_encoder.classes_)} gestures:")
        print(self.label_encoder.classes_)
    
    def predict_gesture(self, landmarks):
        """
        Predict the gesture from landmarks
        """
        if not landmarks or len(landmarks) != 63:  # 21 landmarks * 3 coordinates
            return "unknown", 0.0
        
        # Reshape for model input
        input_data = np.array([landmarks])
        
        # Get prediction
        prediction = self.model.predict(input_data, verbose=0)[0]
        
        # Get the highest confidence prediction
        max_index = np.argmax(prediction)
        confidence = prediction[max_index]
        
        # Get the gesture name
        gesture = self.label_encoder.classes_[max_index]
        
        return gesture, confidence
    
    def get_smoothed_gesture(self, gesture, confidence):
        """
        Apply smoothing to avoid jittery predictions
        """
        # Add current prediction to history
        self.gesture_history.append((gesture, confidence))
        
        # Keep only the recent history
        if len(self.gesture_history) > self.history_length:
            self.gesture_history.pop(0)
        
        # Count occurrences of each gesture in history
        gesture_counts = {}
        for g, c in self.gesture_history:
            if c >= self.confidence_threshold:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        # Return the most common gesture
        if gesture_counts:
            most_common = max(gesture_counts.items(), key=lambda x: x[1])
            # Only return if it appears in at least half the history
            if most_common[1] >= self.history_length // 2:
                return most_common[0]
        
        # Default to idle if no clear gesture is detected
        return "idle"
    
    def execute_action(self, gesture):
        """
        Execute the action corresponding to the detected gesture
        """
        # Check if action exists for this gesture
        if gesture in self.actions:
            # Check cooldown for non-idle gestures
            if gesture != "idle" and gesture != self.current_gesture:
                current_time = time.time()
                if current_time - self.last_gesture_time >= self.gesture_cooldown:
                    # Update gesture and time
                    self.current_gesture = gesture
                    self.last_gesture_time = current_time
                    
                    # Execute action in a separate thread to avoid blocking
                    threading.Thread(target=self.actions[gesture]).start()
            
            # For continuous gestures like movement
            elif gesture in ["move_right", "move_left", "move_up", "move_down", 
                           "scroll_up", "scroll_down"]:
                self.actions[gesture]()
                
        return gesture
    
    def run(self):
        """
        Main loop for gesture control
        """
        self.running = True
        self.exit_requested = False
        
        cap = cv2.VideoCapture(0)
        print("Starting gesture controller. Press 'q' to quit.")
        
        while self.running:
            if self.exit_requested:
                break
                
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Find hands and landmarks
            img = self.detector.find_hands(img)
            landmarks = self.detector.get_normalized_landmarks()
            
            if landmarks:
                # Preprocess landmarks
                processed_landmarks = self.detector.preprocess_landmarks(landmarks)
                
                if processed_landmarks and len(processed_landmarks) == 63:
                    # Predict gesture
                    gesture, confidence = self.predict_gesture(processed_landmarks)
                    
                    # Apply smoothing
                    smoothed_gesture = self.get_smoothed_gesture(gesture, confidence)
                    
                    # Execute action
                    active_gesture = self.execute_action(smoothed_gesture)
                    
                    # Display prediction on image
                    conf_text = f"Confidence: {confidence:.2f}"
                    cv2.putText(img, f"Gesture: {active_gesture}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, conf_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the image
            cv2.imshow("Gesture Controller", img)
            
            # Check for key press
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Gesture controller stopped")
    
    def stop(self):
        """
        Stop the controller
        """
        self.running = False
    
    # Action implementations
    def _move_mouse_right(self):
        x, y = pyautogui.position()
        pyautogui.moveTo(x + 30, y, duration=0.1)
    
    def _move_mouse_left(self):
        x, y = pyautogui.position()
        pyautogui.moveTo(x - 30, y, duration=0.1)
    
    def _move_mouse_up(self):
        x, y = pyautogui.position()
        pyautogui.moveTo(x, y - 30, duration=0.1)
    
    def _move_mouse_down(self):
        x, y = pyautogui.position()
        pyautogui.moveTo(x, y + 30, duration=0.1)
    
    def _left_click(self):
        pyautogui.click()
        print("Left click")
    
    def _right_click(self):
        pyautogui.rightClick()
        print("Right click")
    
    def _select(self):
        pyautogui.hotkey('ctrl', 'a')
        print("Select all")
    
    def _deselect(self):
        pyautogui.click()  # Click to deselect
        print("Deselect")
    
    def _scroll_up(self):
        pyautogui.scroll(5)  # Positive value scrolls up
    
    def _scroll_down(self):
        pyautogui.scroll(-5)  # Negative value scrolls down
    
    def _take_screenshot(self):
        # Generate a filename with timestamp
        screenshot_dir = "screenshots"
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
            
        filename = os.path.join(screenshot_dir, f"screenshot_{int(time.time())}.png")
        
        # Take the screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        print(f"Screenshot saved as {filename}")
    
    def _switch_window(self):
        pyautogui.hotkey('alt', 'tab')
        print("Switch window")
    
    def _switch_tab(self):
        pyautogui.hotkey('ctrl', 'tab')
        print("Switch tab")
    
    def _zoom_in(self):
        pyautogui.hotkey('ctrl', '+')
        print("Zoom in")
    
    def _zoom_out(self):
        pyautogui.hotkey('ctrl', '-')
        print("Zoom out")
    
    def _exit_program(self):
        print("Exit gesture detected, stopping...")
        self.exit_requested = True
    
    def _idle(self):
        # Do nothing
        pass

if __name__ == "__main__":
    try:
        controller = GestureController()
        controller.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nYou need to collect gesture data and train the model first.")
        print("Run 'gesture_collector.py' to collect data, then 'gesture_trainer.py' to train the model.")