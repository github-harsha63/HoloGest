# gesture_collector.py

import cv2
import os
import numpy as np
import pandas as pd
import time
from hand_detector import HandDetector

class GestureCollector:
    def __init__(self, data_dir="gesture_data"):
        self.data_dir = data_dir
        self.detector = HandDetector(min_detection_confidence=0.7)
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Define gestures
        self.gestures = [
            "move_right", "move_left", "move_up", "move_down",
            "left_click", "right_click", "select", "deselect",
            "scroll_up", "scroll_down", "screenshot", "switch_window",
            "switch_tab", "zoom_in", "zoom_out", "exit_program",
            "idle"  # No action
        ]
        
        # Create CSV file for collected data if it doesn't exist
        self.csv_path = os.path.join(data_dir, "gesture_data.csv")
        if not os.path.exists(self.csv_path):
            # Create columns for all landmark features (21 landmarks * 3 coordinates)
            columns = ['label']
            for i in range(21):
                for coord in ['x', 'y', 'z']:
                    columns.append(f'landmark_{i}_{coord}')
            
            # Create empty DataFrame with columns
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_path, index=False)
    
    def collect_gesture_data(self, gesture_name, num_samples=100):
        """
        Collect samples for a specific gesture
        """
        if gesture_name not in self.gestures:
            print(f"Unknown gesture: {gesture_name}")
            print(f"Available gestures: {self.gestures}")
            return
        
        print(f"Collecting data for gesture: {gesture_name}")
        print("Prepare your hand and press 's' to start collecting...")
        
        cap = cv2.VideoCapture(0)
        samples_collected = 0
        collecting = False
        
        while samples_collected < num_samples:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                break
                
            img = self.detector.find_hands(img)
            
            # Add text showing current status
            status_text = f"Gesture: {gesture_name} | "
            status_text += f"Collecting: {'Yes' if collecting else 'No (press s)'} | "
            status_text += f"Samples: {samples_collected}/{num_samples}"
            
            cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Show countdown when collecting
            if collecting:
                # Add small delay between samples to allow for hand movement variation
                time.sleep(0.1)
                
                # Get landmarks
                landmarks = self.detector.get_normalized_landmarks()
                
                if landmarks and len(landmarks) == 63:  # 21 landmarks * 3 coordinates
                    # Preprocess landmarks
                    processed_landmarks = self.detector.preprocess_landmarks(landmarks)
                    
                    if processed_landmarks and len(processed_landmarks) == 63:
                        # Save data
                        self._save_sample(gesture_name, processed_landmarks)
                        samples_collected += 1
                        
                        # Visual feedback
                        cv2.putText(img, f"Sample {samples_collected} collected!", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2)
            
            cv2.imshow("Gesture Collection", img)
            key = cv2.waitKey(1)
            
            if key & 0xFF == ord('s'):
                collecting = True
            elif key & 0xFF == ord('p'):
                collecting = False
            elif key & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print(f"Collected {samples_collected} samples for {gesture_name}")
    
    def _save_sample(self, gesture_name, landmarks):
        """
        Save a sample to the CSV file
        """
        # Prepare data row
        data = [gesture_name] + landmarks
        
        # Read existing data
        df = pd.read_csv(self.csv_path)
        
        # Append new row
        df.loc[len(df)] = data
        
        # Save back to CSV
        df.to_csv(self.csv_path, index=False)
    
    def collect_all_gestures(self, samples_per_gesture=50):
        """
        Collect data for all gestures
        """
        for gesture in self.gestures:
            print(f"\n{'='*50}")
            print(f"Collecting data for '{gesture}'")
            print(f"{'='*50}\n")
            
            input("Press Enter when ready to start collection...")
            self.collect_gesture_data(gesture, samples_per_gesture)
            
            print(f"Completed collection for '{gesture}'")
            print("Take a short break if needed.")
            input("Press Enter to continue to the next gesture...")

if __name__ == "__main__":
    collector = GestureCollector()
    
    print("Welcome to the Gesture Data Collection System")
    print("\nAvailable options:")
    print("1. Collect data for a specific gesture")
    print("2. Collect data for all gestures")
    
    choice = input("\nEnter your choice (1/2): ")
    
    if choice == '1':
        print("\nAvailable gestures:")
        for i, gesture in enumerate(collector.gestures):
            print(f"{i+1}. {gesture}")
        
        gesture_index = int(input("\nEnter gesture number: ")) - 1
        samples = int(input("Enter number of samples to collect: "))
        
        if 0 <= gesture_index < len(collector.gestures):
            collector.collect_gesture_data(collector.gestures[gesture_index], samples)
        else:
            print("Invalid gesture number")
    
    elif choice == '2':
        samples = int(input("Enter number of samples per gesture: "))
        collector.collect_all_gestures(samples)
    
    else:
        print("Invalid choice")