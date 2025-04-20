# main.py

import os
import sys
import time
from gesture_collector import GestureCollector
from gesture_trainer import GestureModelTrainer
from gesture_controller import GestureController

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    """Display the main menu"""
    clear_screen()
    print("\n" + "="*60)
    print("  HAND GESTURE CONTROL SYSTEM")
    print("="*60)
    print("\nWhat would you like to do?")
    print("\n1. Collect gesture data")
    print("2. Train gesture recognition model")
    print("3. Run gesture control system")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ")
    return choice

def collect_data_menu():
    """Menu for data collection"""
    clear_screen()
    print("\n" + "="*60)
    print("  GESTURE DATA COLLECTION")
    print("="*60)
    
    collector = GestureCollector()
    
    print("\n1. Collect data for a specific gesture")
    print("2. Collect data for all gestures")
    print("3. Back to main menu")
    
    subchoice = input("\nEnter your choice (1-3): ")
    
    if subchoice == '1':
        print("\nAvailable gestures:")
        for i, gesture in enumerate(collector.gestures):
            print(f"{i+1}. {gesture}")
        
        gesture_index = int(input("\nEnter gesture number: ")) - 1
        samples = int(input("Enter number of samples to collect (recommended: 50-100): "))
        
        if 0 <= gesture_index < len(collector.gestures):
            collector.collect_gesture_data(collector.gestures[gesture_index], samples)
        else:
            print("Invalid gesture number")
            input("Press Enter to continue...")
    
    elif subchoice == '2':
        samples = int(input("Enter number of samples per gesture (recommended: 50-100): "))
        collector.collect_all_gestures(samples)
    
    elif subchoice == '3':
        return
    
    else:
        print("Invalid choice")
        input("Press Enter to continue...")

def train_model_menu():
    """Menu for model training"""
    clear_screen()
    print("\n" + "="*60)
    print("  GESTURE RECOGNITION MODEL TRAINING")
    print("="*60)
    
    # Check if data exists
    data_path = os.path.join("gesture_data", "gesture_data.csv")
    if not os.path.exists(data_path):
        print("\nNo gesture data found. Please collect data first.")
        input("Press Enter to continue...")
        return
    
    trainer = GestureModelTrainer()
    
    print("\nTraining Settings:")
    epochs = int(input("Enter number of training epochs (default: 50): ") or "50")
    batch_size = int(input("Enter batch size (default: 32): ") or "32")
    
    print("\nStarting model training...")
    trainer.train_model(epochs=epochs, batch_size=batch_size)
    
    print("\nTraining completed!")
    input("Press Enter to return to main menu...")

def run_controller_menu():
    """Menu for running the gesture controller"""
    clear_screen()
    print("\n" + "="*60)
    print("  RUNNING GESTURE CONTROL SYSTEM")
    print("="*60)
    
    # Check if model exists
    model_path = os.path.join("models", "gesture_model.h5")
    encoder_path = os.path.join("models", "label_encoder.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print("\nNo trained model found. Please train a model first.")
        input("Press Enter to continue...")
        return
    
    try:
        confidence = float(input("\nEnter confidence threshold (0.0-1.0, default: 0.7): ") or "0.7")
        
        print("\nStarting gesture controller...")
        print("Position your hand in front of the camera.")
        print("Press 'q' in the camera window to stop the controller.")
        
        # Wait for a moment to let the user read the instructions
        time.sleep(2)
        
        # Start the controller
        controller = GestureController(confidence_threshold=confidence)
        controller.run()
        
    except Exception as e:
        print(f"\nError: {e}")
    
    input("\nPress Enter to return to main menu...")

def main():
    """Main function"""
    while True:
        choice = main_menu()
        
        if choice == '1':
            collect_data_menu()
        elif choice == '2':
            train_model_menu()
        elif choice == '3':
            run_controller_menu()
        elif choice == '4':
            print("\nExiting program. Goodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    main()