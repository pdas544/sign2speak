import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from test_action_recognition import SignLanguageDetector
from tts_helper import TTSHelper

class RealTimePredictor:
    def __init__(self, model_path='models/action_model_cnn_lstm_new.h5', sequence_length=30):
        self.detector = SignLanguageDetector()
        self.actions = self.detector.actions
        self.model = tf.keras.models.load_model(model_path)
        self.sequence_length = sequence_length
        self.sequence = []
        self.predictions = []
        self.threshold = 0.7
        
        # Initialize TTS
        try:
            self.tts = TTSHelper()
            print("TTS system initialized successfully")
        except Exception as e:
            print(f"Warning: TTS initialization failed: {e}")
            self.tts = None
        
        # Initialize mediapipe
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # System states
        self.WAITING = "waiting"    # Waiting for activity
        self.COLLECTING = "collecting"  # Collecting frames for prediction
        self.PREDICTING = "predicting"  # Showing prediction
        self.current_state = self.WAITING
        
        # Timing parameters
        self.collection_window = 5.0  # Time window to collect frames (seconds)
        self.display_time = 1      # Time to display prediction (seconds)
        self.state_start_time = 0
        self.last_movement_time = 0
        self.last_prediction_time = 0
        self.prediction_delay = 0.5   # Delay between predictions
        
        # Prediction smoothing parameters
        self.prediction_history = []
        self.history_length = 5
        self.min_consistent_predictions = 3
        
        # Display parameters
        self.confidence_colors = {
            'high': (0, 255, 0),    # Green
            'medium': (0, 255, 255), # Yellow
            'low': (0, 0, 255)      # Red
        }
        
        # Movement detection
        self.prev_keypoints = None
        self.no_movement_frames = 0
        self.movement_check_frames = 10  # Frames to check for movement
        self.movement_threshold = 0.005  # Reduced threshold for more sensitivity
        self.prev_hand_landmarks = None  # Specifically track hand movements

        print("\nModel Summary:")
        self.model.summary()
        print(f"\nInput shape: {self.model.input_shape}")
        print(f"Output shape: {self.model.output_shape}\n")
        
    def mediapipe_detection(self, frame, holistic):
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Process the image and detect landmarks
        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            results = holistic.process(image)
        
        # Convert the image back to BGR and mark as writeable
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image, results

    def detect_movement(self, results):
        """Detect if there is significant movement in the hand landmarks."""
        if not results.left_hand_landmarks and not results.right_hand_landmarks:
            return False

        def get_hand_position(hand_landmarks):
            if hand_landmarks:
                return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            return None

        # Get current hand positions
        current_left = get_hand_position(results.left_hand_landmarks)
        current_right = get_hand_position(results.right_hand_landmarks)
        
        movement = False
        
        if self.prev_hand_landmarks is None:
            self.prev_hand_landmarks = (current_left, current_right)
            return True  # Return True on first detection to start collecting
        
        prev_left, prev_right = self.prev_hand_landmarks
        
        # Check left hand movement
        if current_left is not None and prev_left is not None:
            movement_left = np.mean(np.abs(current_left - prev_left))
            movement = movement or movement_left > self.movement_threshold
            
        # Check right hand movement
        if current_right is not None and prev_right is not None:
            movement_right = np.mean(np.abs(current_right - prev_right))
            movement = movement or movement_right > self.movement_threshold
            
        # Update previous landmarks
        self.prev_hand_landmarks = (current_left, current_right)
        
        return movement
    
    def update_state(self):
        """Update the system state based on timing and conditions."""
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        if self.current_state == self.WAITING:
            # If movement detected, switch to collecting state
            if current_time - self.last_movement_time < 0.5:  # Movement in last 0.5 seconds
                self.current_state = self.COLLECTING
                self.state_start_time = current_time
                self.sequence = []  # Reset sequence
        
        elif self.current_state == self.COLLECTING:
            # If collection window is over, switch to predicting state
            if current_time - self.state_start_time > self.collection_window:
                self.current_state = self.PREDICTING
                self.state_start_time = current_time
        
        elif self.current_state == self.PREDICTING:
            # If display time is over, switch back to waiting state
            if current_time - self.state_start_time > self.display_time:
                self.current_state = self.WAITING
                self.prediction_history = []  # Reset prediction history
    
    def get_confidence_color(self, confidence):
        if confidence >= 0.8:
            return self.confidence_colors['high']
        elif confidence >= 0.6:
            return self.confidence_colors['medium']
        else:
            return self.confidence_colors['low']
    
    def get_smooth_prediction(self, new_prediction, confidence):
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Add new prediction to history
        self.prediction_history.append((new_prediction, confidence))
        if len(self.prediction_history) > self.history_length:
            self.prediction_history.pop(0)
        
        # Count occurrences of each prediction in history
        prediction_counts = {}
        for pred, conf in self.prediction_history:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        # Get the most common prediction
        if prediction_counts:
            most_common_pred = max(prediction_counts.items(), key=lambda x: x[1])
            if (most_common_pred[1] >= self.min_consistent_predictions and 
                current_time - self.last_prediction_time >= self.prediction_delay):
                self.last_prediction_time = current_time
                return most_common_pred[0], confidence
        
        return None, None
    
    def draw_prediction_overlay(self, image, prediction="", confidence=0):
        # Get frame dimensions
        height, width = image.shape[:2]
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Create semi-transparent overlay for the top bar
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw different content based on state
        if self.current_state == self.WAITING:
            # Show "Waiting for action" message
            if self.no_movement_frames > self.movement_check_frames:
                cv2.putText(image, "No activity detected", 
                          (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, "Perform a sign gesture", 
                          (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        elif self.current_state == self.COLLECTING:
            # Show collection progress
            time_left = self.collection_window - (current_time - self.state_start_time)
            progress = 1.0 - (time_left / self.collection_window)
            bar_width = int((width - 20) * progress)
            
            cv2.putText(image, "Recording gesture...", 
                      (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Draw progress bar
            cv2.rectangle(image, (10, 55), (width - 10, 65), (100, 100, 100), -1)
            cv2.rectangle(image, (10, 55), (10 + bar_width, 65), (0, 255, 255), -1)
            
            # Show time remaining
            cv2.putText(image, f"Time: {time_left:.1f}s", 
                      (width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        
        elif self.current_state == self.PREDICTING:
            # Show prediction with confidence
            color = self.get_confidence_color(confidence)
            cv2.putText(image, f'Detected Sign: {prediction}', 
                      (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            if confidence > 0:
                confidence_text = f"{confidence*100:.1f}%"
                cv2.putText(image, confidence_text, 
                          (width - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        return image

    def predict_in_realtime(self):
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Set webcam properties
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Webcam initialized. Starting prediction...")
        
        # Set mediapipe model 
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while cap.isOpened():
                # Read feed
                ret, frame = cap.read()
                # if not ret:
                #     print("Failed to grab frame")
                #     break

                try:
                    # Make detections
                    image, results = self.mediapipe_detection(frame, holistic)
                    
                    # Draw landmarks
                    self.detector.draw_styled_landmarks(image, results)
                    
                    # Extract keypoints and check for movement
                    keypoints = self.detector.extract_keypoints(results)
                    has_movement = self.detect_movement(results)
                    
                    current_time = cv2.getTickCount() / cv2.getTickFrequency()
                    if has_movement:
                        self.last_movement_time = current_time
                        self.no_movement_frames = 0
                        if self.current_state == self.WAITING:
                            print("Movement detected - Starting collection")  # Debug print
                    else:
                        if current_time - self.last_movement_time > 1.0:  # Only increment after 1 second of no movement
                            self.no_movement_frames += 1
                    
                    # Update system state
                    self.update_state()
                    
                    # Handle different states
                    if self.current_state == self.COLLECTING:
                        # Collect frames during the collection window
                        self.sequence.append(keypoints)
                        self.sequence = self.sequence[-self.sequence_length:]
                        image = self.draw_prediction_overlay(image)
                        
                        # Check if we have enough frames and collection time is up
                        if len(self.sequence) >= self.sequence_length:
                            current_time = cv2.getTickCount() / cv2.getTickFrequency()
                            if current_time - self.state_start_time > self.collection_window:
                                self.current_state = self.PREDICTING
                                self.state_start_time = current_time
                                print("Collection complete - Making prediction")  # Debug print
                        
                    elif self.current_state == self.PREDICTING:
                        try:
                            # Make prediction
                            if len(self.sequence) >= self.sequence_length:
                                res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                                max_confidence = res[np.argmax(res)]
                                predicted_action = self.actions[np.argmax(res)]
                                
                                print(f"Prediction: {predicted_action} with confidence: {max_confidence}")  # Debug print
                                
                                if max_confidence > self.threshold:
                                    # Show prediction immediately
                                    image = self.draw_prediction_overlay(image, predicted_action, max_confidence)
                                    
                                    # Save audio file for the prediction if TTS is available
                                    if self.tts is not None:
                                        audio_file = self.tts.save_to_file(predicted_action)
                                        if audio_file:
                                            print(f"Generated audio for '{predicted_action}'")
                        except Exception as e:
                            print(f"Prediction error: {e}")
                            self.current_state = self.WAITING
                    else:
                        # Waiting state
                        image = self.draw_prediction_overlay(image)
                    
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                except Exception as e:
                    print(f"Error in processing frame: {e}")
                    continue

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        # Cleanup
        if self.tts is not None:
            self.tts.shutdown()
        cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        # Create predictor instance
        predictor = RealTimePredictor()
        
        # Start real-time prediction
        print("Starting real-time prediction... Press 'q' to quit.")
        predictor.predict_in_realtime()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()