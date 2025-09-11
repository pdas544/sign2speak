import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from tqdm import tqdm

class KeypointExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results"""
        pose = np.zeros(33*4)
        face = np.zeros(468*3)
        lh = np.zeros(21*3)
        rh = np.zeros(21*3)

        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility]
                            for res in results.pose_landmarks.landmark]).flatten()
        if results.face_landmarks:
            face = np.array([[res.x, res.y, res.z]
                            for res in results.face_landmarks.landmark]).flatten()
        if results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z]
                        for res in results.left_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z]
                        for res in results.right_hand_landmarks.landmark]).flatten()
        return np.concatenate([pose, face, lh, rh])

    def process_video(self, video_path, sequence_length=30):
        cap = cv2.VideoCapture(video_path)
        sequence = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.holistic.process(image)
            keypoints = self.extract_keypoints(results)
            sequence.append(keypoints)
        cap.release()
        if len(sequence) < sequence_length:
            padding = [np.zeros_like(sequence[0])] * (sequence_length - len(sequence))
            sequence.extend(padding)
        else:
            sequence = sequence[:sequence_length]
        return np.array(sequence)

    def process_dataset_recursive(self, input_root_dir, output_dir, sequence_length=30):
        os.makedirs(output_dir, exist_ok=True)
        for dirpath, dirnames, filenames in os.walk(input_root_dir):
            for filename in tqdm(filenames):
                if filename.endswith('.mp4'):
                    video_path = os.path.join(dirpath, filename)
                    # Determine the gloss from the directory name just beneath "videos/"
                    relative_path = os.path.relpath(video_path, input_root_dir)
                    gloss = os.path.normpath(relative_path).split(os.sep)[0]
                    # Remove file extension for keypoints filename
                    file_stem = os.path.splitext(filename)[0]
                    # Define output .npy filename
                    output_filename = f"{gloss}_{file_stem}_keypoints.npy"
                    output_path = os.path.join(output_dir, output_filename)
                    if os.path.exists(output_path):
                        continue  # Skip if keypoints already exist
                    try:
                        keypoints = self.process_video(video_path, sequence_length)
                        np.save(output_path, keypoints)
                    except Exception as e:
                        print(f"Error processing {video_path}: {str(e)}")

if __name__ == "__main__":
    # Change to your actual input/output directories!
    input_root = "./videos"
    output_root = "./keypoints_full"
    extractor = KeypointExtractor()
    extractor.process_dataset_recursive(input_root, output_root, sequence_length=30)
