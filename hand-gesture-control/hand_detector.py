# hand_detector.py

import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, mode=False, max_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=True):
        """
        Find hands in the image and optionally draw landmarks
        Returns the image with drawings and hand landmarks
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return img
    
    def find_position(self, img, hand_no=0):
        """
        Find the position of hand landmarks
        Returns a list of landmark positions [id, x, y]
        """
        landmark_list = []
        img_height, img_width, _ = img.shape
        
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, lm in enumerate(hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * img_width), int(lm.y * img_height)
                    landmark_list.append([id, cx, cy, lm.z])
                    
        return landmark_list
    
    def get_normalized_landmarks(self, hand_no=0):
        """
        Returns normalized landmark positions (range 0-1) for model training
        """
        normalized_landmarks = []
        
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                hand = self.results.multi_hand_landmarks[hand_no]
                
                # Extract x, y, z coordinates from each landmark
                for lm in hand.landmark:
                    normalized_landmarks.extend([lm.x, lm.y, lm.z])
                    
        return normalized_landmarks
    
    def preprocess_landmarks(self, landmarks):
        """
        Preprocesses landmarks for better model performance:
        1. Center around wrist
        2. Scale based on hand size
        """
        if not landmarks or len(landmarks) < 21 * 3:  # 21 landmarks, each with x,y,z
            return []
        
        # Reshape landmarks to (21, 3) for preprocessing
        landmarks_array = np.array(landmarks).reshape(-1, 3)
        
        # Use wrist (landmark 0) as reference point
        wrist = landmarks_array[0]
        
        # Center around wrist
        centered_landmarks = landmarks_array - wrist
        
        # Scale based on the distance between wrist and middle finger MCP (landmark 9)
        scale_reference = np.linalg.norm(centered_landmarks[9])
        if scale_reference > 0:
            normalized_landmarks = centered_landmarks / scale_reference
        else:
            normalized_landmarks = centered_landmarks
            
        # Flatten back to 1D array
        return normalized_landmarks.flatten().tolist()


# Test the hand detector
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector(min_detection_confidence=0.7)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break
            
        img = detector.find_hands(img)
        landmarks = detector.find_position(img)
        
        if landmarks:
            # Draw circles on key landmarks for visibility
            for id, x, y, _ in landmarks:
                cv2.circle(img, (x, y), 5, (255, 0, 0), cv2.FILLED)
                
            # Display the normalized landmark data for debugging
            norm_landmarks = detector.get_normalized_landmarks()
            if norm_landmarks:
                processed = detector.preprocess_landmarks(norm_landmarks)
                print(f"Preprocessed landmarks shape: {len(processed)}")
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()