import json
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_wsasl_dataset(json_file_path):
    """
    Analyze the WSASL dataset to count videos per gloss and generate statistics.
    
    Args:
        json_file_path (str): Path to the WSASL JSON file
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize counters
    gloss_video_count = defaultdict(int)
    split_counts = defaultdict(lambda: defaultdict(int))
    signer_counts = defaultdict(set)
    source_counts = defaultdict(int)
    
    # Process each gloss
    for gloss_entry in data:
        gloss = gloss_entry['gloss']
        instances = gloss_entry['instances']
        
        # Count total videos for this gloss
        gloss_video_count[gloss] += len(instances)
        
        # Count by split and other attributes
        for instance in instances:
            split = instance.get('split', 'unknown')
            signer_id = instance.get('signer_id', 'unknown')
            source = instance.get('source', 'unknown')
            
            split_counts[gloss][split] += 1
            signer_counts[gloss].add(signer_id)
            source_counts[source] += 1
    
    # Print results
    print("WSASL Dataset Analysis")
    print("=" * 50)
    
    # Total statistics
    total_videos = sum(gloss_video_count.values())
    total_glosses = len(gloss_video_count)
    
    print(f"Total glosses: {total_glosses}")
    print(f"Total videos: {total_videos}")
    print(f"Average videos per gloss: {total_videos/total_glosses:.2f}")
    print()
    
    # Print video counts per gloss
    print("Videos per gloss:")
    print("-" * 30)
    for gloss, count in sorted(gloss_video_count.items(), key=lambda x: x[1], reverse=True)[:20]:  # Show top 20 glosses
        print(f"{gloss}: {count} videos")
        
        # Print split distribution for this gloss
        for split, split_count in split_counts[gloss].items():
            print(f"  {split}: {split_count} videos")
        
        # Print number of unique signers for this gloss
        print(f"  Unique signers: {len(signer_counts[gloss])}")
        print()
    
    # Print source distribution
    print("Video sources:")
    print("-" * 30)
    for source, count in source_counts.items():
        print(f"{source}: {count} videos")
    
    # Create a bar chart of the top N glosses by video count
    top_n = 20  # Show top 20 glosses
    sorted_glosses = sorted(gloss_video_count.items(), key=lambda x: x[1], reverse=True)[:top_n]
    gloss_names = [item[0] for item in sorted_glosses]
    video_counts = [item[1] for item in sorted_glosses]
    
    plt.figure(figsize=(12, 8))
    plt.barh(gloss_names, video_counts)
    plt.xlabel('Number of Videos')
    plt.title(f'Top {top_n} Glosses by Video Count in WSASL Dataset')
    plt.gca().invert_yaxis()  # Display highest count at top
    plt.tight_layout()
    plt.savefig('wsasl_gloss_video_counts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return gloss_video_count, split_counts, signer_counts, source_counts

# Example usage
if __name__ == "__main__":
    # Replace with the actual path to your WSASL JSON file
    json_file_path = "wlasl_dataset.json"
    
    try:
        gloss_counts, split_counts, signer_counts, source_counts = analyze_wsasl_dataset(json_file_path)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{json_file_path}' is not a valid JSON file.")