import json
import os
import subprocess
import pandas as pd
from collections import Counter
from tqdm import tqdm
import argparse
from urllib.parse import urlparse, parse_qs
import time

def extract_video_id(url):
    """
    Extract video ID from a YouTube URL.
    Handles various formats: 
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    """
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
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
        try:
            with open(json_file, 'r') as f:
                data.extend(json.load(f))
            print(f"Loaded {len(data)} entries from {json_file}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return data

def get_top_glosses(data, top_n=10):
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
    Download videos using yt-dlp with enhanced error handling and balanced download per gloss
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a log file for failed downloads
    log_file = os.path.join(output_dir, "failed_downloads.log")
    failed_downloads = []
    
    # Group videos by gloss
    gloss_videos = {}
    for item in data:
        gloss = item.get('text')
        if not gloss:
            continue
        if gloss not in gloss_videos:
            gloss_videos[gloss] = []
        gloss_videos[gloss].append(item)
    
    print(f"Found {len(gloss_videos)} unique glosses")
    
    # Track successful downloads per gloss
    successful_downloads = {gloss: 0 for gloss in gloss_videos.keys()}
    
    # Download videos for each gloss
    for gloss, videos in tqdm(gloss_videos.items(), desc="Processing glosses"):
        gloss_dir = os.path.join(output_dir, gloss.replace(" ", "_").replace("/", "_"))
        os.makedirs(gloss_dir, exist_ok=True)
        
        # Try to download until we reach the desired number of videos per gloss
        for video in tqdm(videos, desc=f"Downloading {gloss} videos", leave=False):
            if successful_downloads[gloss] >= max_videos_per_gloss:
                break
                
            url = video['url']
            video_id = extract_video_id(url)
            if not video_id:
                print(f"Could not extract video ID from URL: {url}")
                failed_downloads.append({"url": url, "reason": "Invalid URL format", "gloss": gloss})
                continue
                
            start_time = video['start_time']
            end_time = video['end_time']
            
            # Output filename pattern
            output_template = os.path.join(gloss_dir, f"{video_id}.%(ext)s")
            
            # Check if file already exists
            if skip_existing:
                existing_files = [f for f in os.listdir(gloss_dir) if f.startswith(video_id)]
                if existing_files:
                    successful_downloads[gloss] += 1
                    continue
            
            # Build yt-dlp command with additional options to handle common issues
            cmd = [
                'yt-dlp',
                '-f', 'best[height<=720]',  # Download up to 720p
                '--download-sections', f'*{start_time}-{end_time}',
                '--output', output_template,
                '--no-check-certificates',
                '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
                '--retries', '3',  # Retry up to 3 times
                '--fragment-retries', '3',
                '--socket-timeout', '30',
                '--source-address', '0.0.0.0',
                url
            ]
            
            try:
                # Run yt-dlp
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    successful_downloads[gloss] += 1
                    print(f"Successfully downloaded: {url} ({successful_downloads[gloss]}/{max_videos_per_gloss} for {gloss})")
                else:
                    error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                    print(f"Failed to download {url}: {error_msg}")
                    failed_downloads.append({
                        "url": url, 
                        "reason": error_msg,
                        "gloss": gloss
                    })
                    
            except subprocess.TimeoutExpired:
                error_msg = "Timeout after 300 seconds"
                print(f"Timeout downloading {url}: {error_msg}")
                failed_downloads.append({
                    "url": url, 
                    "reason": error_msg,
                    "gloss": gloss
                })
            except Exception as e:
                error_msg = str(e)
                print(f"Error downloading {url}: {error_msg}")
                failed_downloads.append({
                    "url": url, 
                    "reason": error_msg,
                    "gloss": gloss
                })
            
            # Add a small delay to avoid overwhelming YouTube servers
            time.sleep(1)
        
        print(f"Downloaded {successful_downloads[gloss]} videos for gloss '{gloss}'")
    
    # Print summary
    print("\nDownload Summary:")
    for gloss, count in successful_downloads.items():
        print(f"{gloss}: {count}/{max_videos_per_gloss}")
    
    # Save failed downloads to a log file
    if failed_downloads:
        with open(log_file, 'w') as f:
            for entry in failed_downloads:
                f.write(f"{entry['url']} | {entry['gloss']} | {entry['reason']}\n")
        print(f"Logged {len(failed_downloads)} failed downloads to {log_file}")
    
    return successful_downloads, failed_downloads

def main():
    parser = argparse.ArgumentParser(description='Download MS-ASL videos for top glosses')
    parser.add_argument('--json-files', nargs='+', required=True,
                        help='Paths to MS-ASL JSON files (e.g., MSASL_train.json MSASL_val.json MSASL_test.json)')
    parser.add_argument('--output-dir', default='msasl_videos',
                        help='Output directory for downloaded videos')
    parser.add_argument('--top-n', type=int, default=10,
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
    successful_downloads, failed_downloads = download_videos(
        filtered_data, args.output_dir, args.max_videos_per_gloss, args.skip_existing
    )
    
    total_successful = sum(successful_downloads.values())
    print(f"Download completed! Successful: {total_successful}, Failed: {len(failed_downloads)}")

if __name__ == "__main__":
    main()