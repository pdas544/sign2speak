import json
import os
import subprocess
import pandas as pd
from collections import Counter
from tqdm import tqdm
import argparse
from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    """
    Extract video ID from a YouTube URL.
    Handles various formats: 
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    """
    if 'youtu.be' in url:
        return url.split('/')[-1]
    elif 'youtube.com' in url:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
    return None

def load_msasl_data(json_files):
    """
    Load MS-ASL data from JSON files
    """
    data = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data.extend(json.load(f))
    return data

def get_top_glosses(data, top_n=100):
    """
    Identify the top N most frequent glosses in the dataset using the 'text' field
    """
    gloss_counter = Counter()
    for item in data:
        gloss = item.get('text')
        if gloss:
            gloss_counter[gloss] += 1
    
    top_glosses = [gloss for gloss, count in gloss_counter.most_common(top_n)]
    return top_glosses

def filter_data_by_gloss(data, glosses):
    """
    Filter dataset to include only items with the specified glosses (using 'text' field)
    """
    return [item for item in data if item.get('text') in glosses]

def download_videos(data, output_dir, max_videos_per_gloss=50, skip_existing=True):
    """
    Download videos using yt-dlp
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group videos by gloss
    gloss_videos = {}
    for item in data:
        gloss = item.get('text')
        if not gloss:
            continue
        if gloss not in gloss_videos:
            gloss_videos[gloss] = []
        gloss_videos[gloss].append(item)
    
    # Download videos for each gloss
    for gloss, videos in tqdm(gloss_videos.items(), desc="Downloading gloss videos"):
        gloss_dir = os.path.join(output_dir, gloss.replace(" ", "_"))
        os.makedirs(gloss_dir, exist_ok=True)
        
        # Limit the number of videos per gloss
        videos = videos[:max_videos_per_gloss]
        
        for video in tqdm(videos, desc=f"Downloading {gloss} videos", leave=False):
            url = video['url']
            video_id = extract_video_id(url)
            if not video_id:
                print(f"Could not extract video ID from URL: {url}")
                continue
                
            start_time = video['start_time']
            end_time = video['end_time']
            
            # Output filename pattern
            output_template = os.path.join(gloss_dir, f"{video_id}_%(title)s.%(ext)s")
            
            # Check if file already exists
            if skip_existing:
                existing_files = [f for f in os.listdir(gloss_dir) if f.startswith(video_id)]
                if existing_files:
                    continue
            
            # Build yt-dlp command
            cmd = [
                'yt-dlp',
                '-f', 'best[height<=720]',  # Download up to 720p
                '--download-sections', f'*{start_time}-{end_time}',
                '--output', output_template,
                url
            ]
            
            try:
                # Run yt-dlp
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {url}: {e}")
            except subprocess.TimeoutExpired:
                print(f"Timeout downloading {url}")

def main():
    parser = argparse.ArgumentParser(description='Download MS-ASL videos for top glosses')
    parser.add_argument('--json-files', nargs='+', required=True,
                        help='Paths to MS-ASL JSON files (e.g., MSASL_train.json MSASL_val.json MSASL_test.json)')
    parser.add_argument('--output-dir', default='msasl_videos',
                        help='Output directory for downloaded videos')
    parser.add_argument('--top-n', type=int, default=100,
                        help='Number of top glosses to download')
    parser.add_argument('--max-videos-per-gloss', type=int, default=50,
                        help='Maximum number of videos to download per gloss')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip downloading if file already exists')
    
    args = parser.parse_args()
    
    # Load MS-ASL data
    print("Loading MS-ASL data...")
    data = load_msasl_data(args.json_files)
    
    # Get top glosses
    print("Identifying top glosses...")
    top_glosses = get_top_glosses(data, args.top_n)
    print(f"Top {args.top_n} glosses: {top_glosses}")
    
    # Filter data
    filtered_data = filter_data_by_gloss(data, top_glosses)
    print(f"Found {len(filtered_data)} videos for top {args.top_n} glosses")
    
    # Download videos
    print("Starting video download...")
    download_videos(filtered_data, args.output_dir, args.max_videos_per_gloss, args.skip_existing)
    print("Download completed!")

if __name__ == "__main__":
    main()