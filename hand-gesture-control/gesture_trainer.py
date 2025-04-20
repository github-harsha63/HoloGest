# gesture_trainer.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib

class GestureModelTrainer:
    def __init__(self, data_dir="gesture_data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.csv_path = os.path.join(data_dir, "gesture_data.csv")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        self.model = None
        self.label_encoder = None
    
    def load_data(self):
        """
        Load the collected gesture data
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Data file not found at {self.csv_path}")
        
        # Load data
        df = pd.read_csv(self.csv_path)
        
        # Check if data is available
        if len(df) == 0:
            raise ValueError("No data available in the CSV file")
        
        # Print some information about the data
        print(f"Loaded {len(df)} samples")
        print(f"Gesture distribution:")
        print(df['label'].value_counts())
        
        # Extract features and labels
        X = df.drop('label', axis=1).values
        y = df['label'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Convert to categorical
        y_categorical = to_categorical(y_encoded)
        
        # Save label encoder for later use
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, "label_encoder.pkl"))
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test, self.label_encoder.classes_
    
    def build_model(self, input_shape, num_classes):
        """
        Build the neural network model for gesture recognition
        """
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            
            # Hidden layers
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the model on the collected data
        """
        # Load data
        X_train, X_test, y_train, y_test, classes = self.load_data()
        
        # Build model
        model = self.build_model(X_train.shape[1], len(classes))
        
        # Print model summary
        model.summary()
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Save model
        model.save(os.path.join(self.model_dir, "gesture_model.h5"))
        
        # Plot training history
        self._plot_training_history(history)
        
        return model, history, classes
    
    def _plot_training_history(self, history):
        """
        Plot training and validation accuracy and loss
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training and validation accuracy
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training and validation loss
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper left')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "training_history.png"))
        plt.close()

if __name__ == "__main__":
    trainer = GestureModelTrainer()
    
    print("Starting gesture recognition model training...")
    
    # Custom training parameters
    epochs = int(input("Enter number of training epochs (default: 50): ") or "50")
    batch_size = int(input("Enter batch size (default: 32): ") or "32")
    
    # Train the model
    model, history, classes = trainer.train_model(epochs=epochs, batch_size=batch_size)
    
    print("\nTraining completed!")
    print(f"Model saved to {os.path.join(trainer.model_dir, 'gesture_model.h5')}")
    print(f"Recognized gestures: {classes}")