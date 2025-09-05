import os
import shutil
from tqdm import tqdm

def reorganize_videos(source_dir, target_dir):
    """
    Move all videos from gloss subdirectories to a single directory
    with filenames that include the gloss name to prevent conflicts
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # Count total videos
    total_videos = 0
    for gloss_dir in os.listdir(source_dir):
        gloss_path = os.path.join(source_dir, gloss_dir)
        if os.path.isdir(gloss_path):
            total_videos += len([f for f in os.listdir(gloss_path) if f.endswith('.mp4')])
    
    print(f"Found {total_videos} videos to reorganize")
    
    # Move and rename videos
    moved_count = 0
    for gloss_dir in tqdm(os.listdir(source_dir), desc="Processing gloss directories"):
        gloss_path = os.path.join(source_dir, gloss_dir)
        
        if os.path.isdir(gloss_path):
            for filename in os.listdir(gloss_path):
                if filename.endswith('.mp4'):
                    # Extract video ID from filename (remove extension)
                    video_id = os.path.splitext(filename)[0]
                    
                    # Create new filename with gloss prefix
                    new_filename = f"{gloss_dir}_{video_id}.mp4"
                    new_path = os.path.join(target_dir, new_filename)
                    
                    # Move and rename the file
                    old_path = os.path.join(gloss_path, filename)
                    shutil.move(old_path, new_path)
                    moved_count += 1
    
    print(f"Successfully moved {moved_count} videos to {target_dir}")
    return moved_count

if __name__ == "__main__":
    source_directory = "videos_top_10_gloss"  # Directory with gloss subdirectories
    target_directory = "videos_all"        # Directory where all videos will be stored
    
    reorganize_videos(source_directory, target_directory)