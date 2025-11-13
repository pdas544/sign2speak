import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

class SignLanguageDetector:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
        
        # Initialize data paths and parameters
        self.data_path = os.path.join('test_data')
        self.actions = np.array(['hello', 'yes', 'no', 'teacher','beautiful','boy','nice',
                               'like','big','friend','happy','go','good','sister','brother',
                               'how_are_you', 'family'])
        self.no_sequences = 30
        self.sequence_length = 30
        self.start_folder = 0
        
        # Create necessary directories
        self._create_directories()
        
    def _create_directories(self):
        """Create complete directory structure for data collection"""
        try:
            # 1. Create root directory
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
                print(f"Created root directory: {self.data_path}")
            
            # 2. Create action subdirectories and sequence directories
            for action in self.actions:
                action_dir = os.path.join(self.data_path, action)
                if not os.path.exists(action_dir):
                    os.makedirs(action_dir)
                    print(f"Created action directory: {action_dir}")
                
                # 3. Create sequence subdirectories (0-29)
                for sequence in range(self.no_sequences):
                    sequence_dir = os.path.join(action_dir, str(sequence))
                    if not os.path.exists(sequence_dir):
                        os.makedirs(sequence_dir)
                        print(f"Created sequence directory: {sequence_dir}")
                        
            print("Directory structure created successfully!")
            
        except Exception as e:
            print(f"Error creating directory structure: {e}")
            raise

    def mediapipe_detection(self, image, model):
        """Process image and detect landmarks using MediaPipe"""
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = model.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image, results
        except Exception as e:
            print(f"Error in mediapipe_detection: {e}")
            return None, None

    def draw_landmarks(self, image, results):
        """Draw basic landmarks on the image"""
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.face_connections)
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

    def draw_styled_landmarks(self, image, results):
        """Draw styled landmarks with colors and different sizes"""
        # Draw face connections
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.face_connections,
            self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
        
        # Draw pose connections
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
        
        # Draw hand connections
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results"""
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def capture_video(self):
        """Capture and process video from webcam"""
        print("Starting video capture. Press 'q' to quit.")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None, None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        
        holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                image, results = self.mediapipe_detection(frame, holistic)
                if image is None:
                    continue
                
                self.draw_styled_landmarks(image, results)
                cv2.imshow('OpenCV Feed', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"Error in capture_video: {e}")
        finally:
            print("Cleaning up...")
            holistic.close()
            cap.release()
            cv2.destroyAllWindows()
        
        return image, results


    def collect_and_extract_keypoints(self):
        """Collect and save keypoints for training data with improved user experience"""
        print("\nStarting data collection...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        try:
            with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for action in self.actions:
                    action_dir = os.path.join(self.data_path, action)
                    
                    # Display preparation screen
                    self._show_action_preparation(cap, action, holistic)
                    
                    # Loop through sequences/videos
                    for sequence in range(self.no_sequences):
                        sequence_dir = os.path.join(action_dir, str(sequence))
                        
                        # Countdown before starting sequence
                        self._show_countdown(cap, holistic, action, sequence)
                        
                        # Collect frames for the sequence
                        frames_collected = self._collect_sequence(cap, holistic, action, sequence, sequence_dir)
                        
                        if not frames_collected:
                            print("\nCollection interrupted by user")
                            return
                        
                        # Show success message and wait before next sequence
                        self._show_sequence_complete(cap, holistic, action, sequence)
                    
                    # Show action complete message
                    self._show_action_complete(cap, holistic, action)
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nData collection complete!")

    def _show_action_preparation(self, cap, action, holistic, wait_time=3):
        """Show preparation screen for next action"""
        for _ in range(wait_time * 30):  # 30 fps * wait_time seconds
            ret, frame = cap.read()
            if not ret:
                continue
                
            image, results = self.mediapipe_detection(frame, holistic)
            if image is None:
                continue
                
            # Create preparation screen
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (640, 480), (0, 0, 0), -1)
            alpha = 0.6
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # Add text
            cv2.putText(image, f"Prepare for Action: {action}", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Get ready to perform the action", (50, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        return True

    def _show_countdown(self, cap, holistic, action, sequence, countdown_time=3):
        """Show countdown before starting sequence"""
        for i in range(countdown_time, 0, -1):
            for _ in range(30):  # Show each number for 30 frames (1 second)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                image, results = self.mediapipe_detection(frame, holistic)
                if image is None:
                    continue
                    
                self.draw_styled_landmarks(image, results)
                
                # Add countdown overlay
                cv2.rectangle(image, (0, 0), (640, 100), (245, 117, 16), -1)
                cv2.putText(image, f"Starting sequence {sequence} in: {i}", (15, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow('OpenCV Feed', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False
        return True

    def _collect_sequence(self, cap, holistic, action, sequence, sequence_dir):
        """Collect frames for a single sequence"""
        frames = []
        progress_bar_width = 400
        
        for frame_num in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                continue
                
            image, results = self.mediapipe_detection(frame, holistic)
            if image is None:
                continue
                
            self.draw_styled_landmarks(image, results)
            
            # Add progress bar
            progress = int((frame_num + 1) / self.sequence_length * progress_bar_width)
            cv2.rectangle(image, (120, 0), (120 + progress_bar_width, 40), (245, 117, 16), -1)
            cv2.rectangle(image, (120, 0), (120 + progress, 40), (0, 255, 0), -1)
            
            # Add text overlay
            cv2.putText(image, f"Recording {action}: {frame_num+1}/{self.sequence_length}", (15, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image)
            
            # Extract and save keypoints
            keypoints = self.extract_keypoints(results)
            npy_path = os.path.join(sequence_dir, f"{frame_num}.npy")
            np.save(npy_path, keypoints)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
                
        return True

    def _show_sequence_complete(self, cap, holistic, action, sequence, wait_time=1):
        """Show completion message for sequence"""
        start_time = time.time()
        while (time.time() - start_time) < wait_time:
            ret, frame = cap.read()
            if not ret:
                continue
                
            image, results = self.mediapipe_detection(frame, holistic)
            if image is None:
                continue
                
            self.draw_styled_landmarks(image, results)
            
            # Add completion message
            cv2.rectangle(image, (0, 0), (640, 100), (0, 255, 0), -1)
            cv2.putText(image, f"Sequence {sequence} Complete!", (15, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        return True

    def _show_action_complete(self, cap, holistic, action, wait_time=2):
        """Show completion message for entire action"""
        start_time = time.time()
        while (time.time() - start_time) < wait_time:
            ret, frame = cap.read()
            if not ret:
                continue
                
            image, results = self.mediapipe_detection(frame, holistic)
            if image is None:
                continue
                
            # Create completion screen
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (640, 480), (0, 255, 0), -1)
            alpha = 0.6
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # Add completion message
            cv2.putText(image, f"Action '{action}' Complete!", (50, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, "Preparing for next action...", (50, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        return True

    def verify_data_structure(self):
        """Verify the data structure and count files"""
        try:
            print("\nVerifying data structure...")
            for action in self.actions:
                action_dir = os.path.join(self.data_path, action)
                if not os.path.exists(action_dir):
                    print(f"Missing action directory: {action}")
                    continue
                
                for sequence in range(self.no_sequences):
                    sequence_dir = os.path.join(action_dir, str(sequence))
                    if not os.path.exists(sequence_dir):
                        print(f"Missing sequence directory: {sequence_dir}")
                        continue
                    
                    # Count .npy files
                    npy_files = [f for f in os.listdir(sequence_dir) if f.endswith('.npy')]
                    if len(npy_files) != self.sequence_length:
                        print(f"Incorrect number of frames in {sequence_dir}: {len(npy_files)}/{self.sequence_length}")
                    
            print("Verification complete!")
            
        except Exception as e:
            print(f"Error during verification: {e}")

def main():
    try:
        print("Starting application...")

        detector = SignLanguageDetector()
        # Verify initial directory structure
        detector.verify_data_structure()
        
        # Collect data
        detector.collect_and_extract_keypoints()
        
        # Verify after collection
        detector.verify_data_structure()
    except Exception as e:
        print(f"Main error: {e}")

if __name__ == "__main__":
    main()







