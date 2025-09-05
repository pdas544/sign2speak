import cv2
import mediapipe as mp
import torch
import pandas as pd
from tqdm import tqdm
import os
import json
from collections import defaultdict
import re

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(video_path):
    """
    Extract pose, hand, and facial landmarks from a video using MediaPipe Holistic.
    Returns a tensor of shape (num_frames, num_landmarks * 3)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detections
            results = holistic.process(image)
            image.flags.writeable = True
            
            # Extract keypoints with fallback for missing landmarks
            pose = [[0, 0, 0] for _ in range(33)] if not results.pose_landmarks else [
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ]
            
            lh = [[0, 0, 0] for _ in range(21)] if not results.left_hand_landmarks else [
                [lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark
            ]
            
            rh = [[0, 0, 0] for _ in range(21)] if not results.right_hand_landmarks else [
                [lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark
            ]
            
            face = [[0, 0, 0] for _ in range(468)] if not results.face_landmarks else [
                [lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark
            ]
            
            # Flatten all landmarks into a single vector per frame
            frame_landmarks = []
            for landmark_list in [pose, lh, rh, face]:
                for landmark in landmark_list:
                    frame_landmarks.extend(landmark)
            
            frames.append(frame_landmarks)
    
    cap.release()
    
    # Convert to tensor and return
    if frames:
        return torch.tensor(frames, dtype=torch.float32)
    else:
        return torch.tensor([], dtype=torch.float32)

def extract_video_id_from_url(url):
    """
    Extract video ID from a YouTube URL.
    """
    # Handle different YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'youtu.be\/([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def find_video_file(video_id, gloss, base_dir="selected_videos"):
    """
    Find a video file for a given video ID and gloss
    """
    gloss_dir = os.path.join(base_dir, gloss)
    
    if not os.path.exists(gloss_dir):
        return None
    
    # Look for files that start with the video ID
    for filename in os.listdir(gloss_dir):
        if filename.startswith(video_id) and filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return os.path.join(gloss_dir, filename)
    
    return None

def extract_keypoints_with_splits():
    """
    Main function to extract keypoints from videos while preserving split information
    """
    # Paths and parameters
    annotations_path = 'filtered_annotations_selected_glosses.json'
    videos_base_dir = 'selected_videos'
    base_output_dir = 'processed'
    
    # Create output directories for each split
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(base_output_dir, split), exist_ok=True)
    
    # Load the filtered annotations
    print("Loading filtered annotations...")
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    print(f"Found {len(data)} entries in filtered annotations")
    
    # Prepare metadata storage
    metadata = []
    stats = defaultdict(lambda: defaultdict(int))
    unmatched_videos = []
    
    # Process each JSON entry
    for item in tqdm(data, desc="Processing videos"):
        # Extract information from JSON
        gloss = item.get('text', 'unknown')
        split = item.get('split', 'train')
        url = item.get('url', '')
        
        # Extract video ID from URL
        video_id = extract_video_id_from_url(url)
        if not video_id:
            print(f"Could not extract video ID from URL: {url}")
            unmatched_videos.append(f"URL: {url}, Gloss: {gloss}")
            continue
        
        # Find the video file
        video_path = find_video_file(video_id, gloss, videos_base_dir)
        if not video_path:
            print(f"Video file not found for ID: {video_id}, Gloss: {gloss}")
            unmatched_videos.append(f"Video ID: {video_id}, Gloss: {gloss}")
            continue
        
        # Construct output path
        output_filename = f"{gloss.replace(' ', '_')}_{video_id}.pt"
        output_path = os.path.join(base_output_dir, split, output_filename)
        
        # Skip if already processed
        if os.path.exists(output_path):
            stats[gloss][split] += 1
            metadata.append({
                'video_id': video_id,
                'filename': os.path.basename(video_path),
                'file_path': output_path,
                'label': gloss,
                'split': split,
                'status': 'skipped (already processed)'
            })
            continue
        
        try:
            # Extract keypoints
            keypoints_tensor = extract_landmarks(video_path)
            
            # Check if extraction was successful
            if keypoints_tensor.nelement() == 0:
                print(f"Failed to extract keypoints from: {video_path}")
                metadata.append({
                    'video_id': video_id,
                    'filename': os.path.basename(video_path),
                    'file_path': 'N/A',
                    'label': gloss,
                    'split': split,
                    'status': 'failed (no keypoints extracted)'
                })
                continue
            
            # Save the tensor
            torch.save(keypoints_tensor, output_path)
            
            # Update stats and metadata
            stats[gloss][split] += 1
            metadata.append({
                'video_id': video_id,
                'filename': os.path.basename(video_path),
                'file_path': output_path,
                'label': gloss,
                'split': split,
                'status': 'success',
                'num_frames': keypoints_tensor.shape[0],
                'num_features': keypoints_tensor.shape[1]
            })
            
        except Exception as e:
            print(f"Failed to process {video_path}: {e}")
            metadata.append({
                'video_id': video_id,
                'filename': os.path.basename(video_path),
                'file_path': 'N/A',
                'label': gloss,
                'split': split,
                'status': f'failed ({str(e)})'
            })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(base_output_dir, 'metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    # Save list of unmatched videos
    if unmatched_videos:
        unmatched_path = os.path.join(base_output_dir, 'unmatched_videos.txt')
        with open(unmatched_path, 'w') as f:
            for video in unmatched_videos:
                f.write(f"{video}\n")
        print(f"List of unmatched videos saved to: {unmatched_path}")
    
    # Print summary
    print("\n=== Extraction Summary ===")
    print(f"Processed {len(metadata)} videos")
    print(f"Metadata saved to: {metadata_path}")
    
    successful = len([m for m in metadata if m['status'] == 'success'])
    skipped = len([m for m in metadata if 'skipped' in m['status']])
    failed = len([m for m in metadata if 'failed' in m['status']])
    
    print(f"Successful: {successful}, Skipped: {skipped}, Failed: {failed}")
    print(f"Unmatched videos: {len(unmatched_videos)}")
    
    # Print stats per gloss and split
    print("\n=== Videos per Gloss and Split ===")
    for gloss in sorted(stats.keys()):
        print(f"{gloss}:")
        for split in splits:
            count = stats[gloss].get(split, 0)
            print(f"  {split}: {count}")
        print()
    
    return metadata_df

if __name__ == "__main__":
    # Run the extraction
    metadata = extract_keypoints_with_splits()
    
    # Additional analysis
    print("\n=== Additional Analysis ===")
    if not metadata.empty:
        # Count success/failure by split
        split_status = metadata.groupby(['split', 'status']).size().unstack(fill_value=0)
        print("Status counts by split:")
        print(split_status)
        
        # Count videos by gloss and split
        gloss_split_count = metadata[metadata['status'] == 'success'].groupby(['label', 'split']).size().unstack(fill_value=0)
        print("\nSuccessful videos by gloss and split:")
        print(gloss_split_count)