import json
import os
import subprocess
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse
from urllib.parse import urlparse, parse_qs
import time

def extract_video_id(url):
    """
    Extract video ID from a YouTube URL.
    """
    if 'youtu.be' in url:
        return url.split('/')[-1].split('?')[0]
    elif 'youtube.com' in url:
        parsed_url = urlparse(url)
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
    return None

def load_filtered_json(json_file):
    """
    Load filtered JSON data
    """
    with open(json_file, 'r') as f:
        return json.load(f)

def download_videos_with_split_distribution(data, output_dir, max_videos_per_gloss=50, skip_existing=True):
    """
    Download videos while maintaining the split distribution
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a log file for failed downloads
    log_file = os.path.join(output_dir, "failed_downloads.log")
    failed_downloads = []
    
    # Group videos by gloss and split
    gloss_split_videos = defaultdict(lambda: defaultdict(list))
    for item in data:
        gloss = item.get('text')
        split = item.get('split', 'train')
        gloss_split_videos[gloss][split].append(item)
    
    print(f"Found {len(gloss_split_videos)} unique glosses")
    
    # Track successful downloads per gloss and split
    successful_downloads = defaultdict(lambda: defaultdict(int))
    
    # Download videos for each gloss and split
    for gloss, split_data in tqdm(gloss_split_videos.items(), desc="Processing glosses"):
        gloss_dir = os.path.join(output_dir, gloss.replace(" ", "_").replace("/", "_"))
        os.makedirs(gloss_dir, exist_ok=True)
        
        for split, videos in split_data.items():
            # Try to download until we reach the desired number of videos per gloss per split
            for video in tqdm(videos, desc=f"Downloading {gloss} ({split}) videos", leave=False):
                if successful_downloads[gloss][split] >= max_videos_per_gloss:
                    break
                    
                url = video['url']
                video_id = extract_video_id(url)
                if not video_id:
                    print(f"Could not extract video ID from URL: {url}")
                    failed_downloads.append({"url": url, "reason": "Invalid URL format", "gloss": gloss, "split": split})
                    continue
                    
                start_time = video['start_time']
                end_time = video['end_time']
                
                # Output filename pattern
                output_template = os.path.join(gloss_dir, f"{video_id}.%(ext)s")
                
                # Check if file already exists
                if skip_existing:
                    existing_files = [f for f in os.listdir(gloss_dir) if f.startswith(video_id)]
                    if existing_files:
                        successful_downloads[gloss][split] += 1
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
                        successful_downloads[gloss][split] += 1
                        print(f"Successfully downloaded: {url} ({successful_downloads[gloss][split]}/{max_videos_per_gloss} for {gloss} in {split})")
                    else:
                        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                        print(f"Failed to download {url}: {error_msg}")
                        failed_downloads.append({
                            "url": url, 
                            "reason": error_msg,
                            "gloss": gloss,
                            "split": split
                        })
                        
                except subprocess.TimeoutExpired:
                    error_msg = "Timeout after 300 seconds"
                    print(f"Timeout downloading {url}: {error_msg}")
                    failed_downloads.append({
                        "url": url, 
                        "reason": error_msg,
                        "gloss": gloss,
                        "split": split
                    })
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error downloading {url}: {error_msg}")
                    failed_downloads.append({
                        "url": url, 
                        "reason": error_msg,
                        "gloss": gloss,
                        "split": split
                    })
                
                # Add a small delay to avoid overwhelming YouTube servers
                time.sleep(1)
            
            print(f"Downloaded {successful_downloads[gloss][split]} videos for gloss '{gloss}' in {split}")
    
    # Print summary
    print("\nDownload Summary:")
    for gloss, split_data in successful_downloads.items():
        for split, count in split_data.items():
            print(f"{gloss} ({split}): {count}/{max_videos_per_gloss}")
    
    # Save failed downloads to a log file
    if failed_downloads:
        with open(log_file, 'w') as f:
            for entry in failed_downloads:
                f.write(f"{entry['url']} | {entry['gloss']} | {entry['split']} | {entry['reason']}\n")
        print(f"Logged {len(failed_downloads)} failed downloads to {log_file}")
    
    return successful_downloads, failed_downloads

def main():
    parser = argparse.ArgumentParser(description='Download MS-ASL videos for selected glosses with split distribution')
    parser.add_argument('--filtered-json', default='filtered_annotations_selected_glosses.json',
                        help='Path to filtered JSON file')
    parser.add_argument('--output-dir', default='selected_videos',
                        help='Output directory for downloaded videos')
    parser.add_argument('--max-videos-per-split', type=int, default=50,
                        help='Maximum number of videos to download per gloss per split')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip downloading if file already exists')
    
    args = parser.parse_args()
    
    # Check if filtered JSON exists
    if not os.path.exists(args.filtered_json):
        print(f"Filtered JSON file '{args.filtered_json}' not found. Please create it first.")
        return
    
    # Load filtered data
    print("Loading filtered JSON data...")
    filtered_data = load_filtered_json(args.filtered_json)
    print(f"Found {len(filtered_data)} entries in filtered JSON")
    
    # Download videos
    print("Starting video download with split distribution...")
    successful_downloads, failed_downloads = download_videos_with_split_distribution(
        filtered_data, args.output_dir, args.max_videos_per_split, args.skip_existing
    )
    
    total_successful = sum(sum(split_data.values()) for split_data in successful_downloads.values())
    print(f"Download completed! Successful: {total_successful}, Failed: {len(failed_downloads)}")

if __name__ == "__main__":
    main()