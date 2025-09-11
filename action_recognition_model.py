import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime
import os

class ActionRecognitionModel:
    def __init__(self, actions, sequence_length=30, model_path='models/action_model_cnn_lstm_new.h5'):
        self.actions = actions
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = None
        self.label_map = {str(label): num for num, label in enumerate(actions)}

        self.logs_dir = 'logs'
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            print(f"Created logs directory: {self.logs_dir}")
        
    def build_model(self, input_shape, num_classes):
        """Build LSTM model architecture"""
        model = Sequential([
            # LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
            # LSTM(128, return_sequences=True, activation='relu'),
            # LSTM(64, return_sequences=False, activation='relu'),
            # Dense(64, activation='relu'),
            # Dropout(0.2),
            # Dense(32, activation='relu'),
            # Dense(num_classes, activation='softmax')
            #new config
            LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape),
            Dropout(0.3),
            LSTM(256, return_sequences=True, activation='relu'),
            Dropout(0.3),
            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(64, return_sequences=False, activation='relu'),  # Note: return_sequences=False here
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
            
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001),
                     loss='categorical_crossentropy',
                     metrics=['categorical_accuracy'])
        return model
    
    def build_cnn_lstm_model(self, input_shape, num_classes):
        """Build CNN+LSTM model architecture"""
        model = Sequential([
            # CNN layers for feature extraction
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        # LSTM layers for temporal dependencies
        LSTM(256, return_sequences=True, activation='relu'),
        Dropout(0.3),
        LSTM(128, return_sequences=False, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
            
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001),
                     loss='categorical_crossentropy',
                     metrics=['categorical_accuracy'])
        return model
    
    def augment_sequence(self, sequence):
        """Apply data augmentation to a sequence"""
        # Random rotation
        angle = np.random.uniform(-5, 5)
        cos_angle = np.cos(np.radians(angle))
        sin_angle = np.sin(np.radians(angle))
        
        # Random scaling
        scale = np.random.uniform(0.9, 1.1)
        
        # Random noise
        noise_factor = 0.02
        
        augmented_sequence = []
        for frame in sequence:
            # Apply rotation and scaling to coordinates
            pose_points = frame.reshape(-1, 3)
            
            # Center the points
            center = np.mean(pose_points, axis=0)
            centered_points = pose_points - center
            
            # Apply rotation
            x = centered_points[:, 0]
            y = centered_points[:, 1]
            rotated_x = x * cos_angle - y * sin_angle
            rotated_y = x * sin_angle + y * cos_angle
            
            # Apply scaling
            centered_points[:, 0] = rotated_x * scale
            centered_points[:, 1] = rotated_y * scale
            
            # Move back to original center
            augmented_points = centered_points + center
            
            # Add noise
            noise = np.random.normal(0, noise_factor, augmented_points.shape)
            augmented_points += noise
            
            augmented_sequence.append(augmented_points.reshape(frame.shape))
        
        return np.array(augmented_sequence)

    def load_data(self, data_path):
        """Load and preprocess the data with augmentation"""
        sequences, labels = [], []
        for action in self.actions:
            for sequence in range(30):  # Assuming 30 sequences per action
                window = []
                for frame_num in range(self.sequence_length):
                    res = np.load(os.path.join(data_path, action, str(sequence), f"{frame_num}.npy"))
                    window.append(res)
                
                # Add original sequence
                sequences.append(window)
                labels.append(self.label_map[str(action)])
                
                # Add augmented sequence
                aug_window = self.augment_sequence(window)
                sequences.append(aug_window)
                labels.append(self.label_map[str(action)])

        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=16):
        """
        Train the model with early stopping. If a saved model exists, load it first.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Maximum number of epochs
            batch_size: Batch size for training
        """
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(self.actions)
        
        # Check if saved model exists
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
        else:
            print("No existing model found. Creating new model...")
            self.model = self.build_cnn_lstm_model(input_shape, num_classes)
        
        early_stopping = EarlyStopping(
            monitor='val_categorical_accuracy',
            min_delta=0.005,  # Smaller delta for finer improvement detection
            patience=30,      # More patience
            mode='max',
            restore_best_weights=True,
            verbose=1)
            
        model_checkpoint = ModelCheckpoint(
            self.model_path,                      # Path where the model will be saved
            monitor='val_categorical_accuracy',    # Metric to monitor
            mode='max',                           # 'max' because we want highest accuracy
            save_best_only=True,                  # Only save when model improves
            verbose=1)                            # Print message when saving)
        
        # Add learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_categorical_accuracy',
            factor=0.2,
            patience=10,
            min_lr=0.00001,
            verbose=1)
        
        # Train with both callbacks
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, reduce_lr]
        )

             # Plot and save training history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        # Save training history plot
        history_path = os.path.join(self.logs_dir, f'training_history_{timestamp}.png')
        plt.savefig(history_path)
        plt.close()
        
        print(f"\nTraining history plot saved to: {history_path}")
        
        # Save training history data
        history_data_path = os.path.join(self.logs_dir, f'training_history_{timestamp}.npy')
        np.save(history_data_path, history.history)
        print(f"Training history data saved to: {history_data_path}")
        
        return history
        

    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        if self.model is None:
            raise ValueError("Model needs to be trained or loaded first")
        
        # Get current timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        
        # Get predictions for confusion matrix
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.actions,
                   yticklabels=self.actions)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        
        # Save confusion matrix plot
        cm_path = os.path.join(self.logs_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(cm_path)
        plt.close()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=self.actions))
        
        # Save classification report to file
        report_path = os.path.join(self.logs_dir, f'classification_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("Classification Report:\n")
            f.write(classification_report(y_true_classes, y_pred_classes, target_names=self.actions))
        
        print(f"\nConfusion matrix saved to: {cm_path}")
        print(f"Classification report saved to: {report_path}")
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classification_report": classification_report(y_true_classes, y_pred_classes, target_names=self.actions)
        }

    def predict(self, sequence):
        """Make predictions on new sequences"""
        if self.model is None:
            raise ValueError("Model needs to be trained or loaded first")
            
        # Ensure sequence has correct shape
        sequence = np.expand_dims(sequence, axis=0)
        prediction = self.model.predict(sequence)
        action_idx = np.argmax(prediction[0])
        confidence = prediction[0][action_idx]
        
        # Get action name from index
        action_name = list(self.label_map.keys())[list(self.label_map.values()).index(action_idx)]
        
        return {
            "action": action_name,
            "confidence": float(confidence)
        }

    def save_model(self, filepath=None):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save")
        if filepath is None:
            filepath = self.model_path
        self.model.save(filepath)

    def load_model(self, filepath=None):
        """Load a trained model"""
        if filepath is None:
            filepath = self.model_path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at {filepath}")
        self.model = tf.keras.models.load_model(filepath)

def main():
    # Example usage
    from test_action_recognition import SignLanguageDetector

    detector = SignLanguageDetector()
    actions = detector.actions
    data_path = detector.data_path
    
    # Initialize model
    model = ActionRecognitionModel(actions)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = model.load_data(data_path)
    
    # Train model
    print("\nStarting model training...")
    history = model.train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model with detailed metrics
    print("\nEvaluating model...")
    results = model.evaluate_model(X_test, y_test)
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print(f"Test Loss: {results['loss']:.4f}")
    
    # Save the model
    print("\nSaving model...")
    model.save_model()
    print(f"Model saved to: {model.model_path}")
    
    print("\nAll visualizations and reports have been saved to the 'logs' directory.")

if __name__ == "__main__":
    main()