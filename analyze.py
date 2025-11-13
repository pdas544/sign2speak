import json
import os
from collections import Counter

def analyze_msasl_file(file_path):
    """
    Analyze an MS-ASL JSON file and return statistics
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Count videos per gloss
        gloss_counter = Counter()
        for item in data:
            gloss = item.get('text')
            if gloss:
                gloss_counter[gloss] += 1
        
        return {
            'total_entries': len(data),
            'unique_glosses': len(gloss_counter),
            'gloss_counts': gloss_counter
        }
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def main():
    # List of MS-ASL JSON files
    # json_files = ['annotations/MSASL_train.json', 'annotations/MSASL_val.json', 'annotations/MSASL_test.json']
    json_files = ['annotations/filtered_annotations_selected_glosses.json']
   
    
    # Check which files exist
    existing_files = [f for f in json_files if os.path.exists(f)]
    
    if not existing_files:
        print("No MS-ASL JSON files found in the current directory.")
        return
    
    print("MS-ASL Dataset Analysis")
    print("=" * 50)
    
    # Analyze each file
    for file_path in existing_files:
        print(f"\nAnalyzing {file_path}:")
        print("-" * 30)
        
        stats = analyze_msasl_file(file_path)
        if stats:
            print(f"Total entries: {stats['total_entries']}")
            print(f"Unique glosses: {stats['unique_glosses']}")
            
            # Display top 10 glosses by count
            print("\nTop 10 glosses by video count:")
            for gloss, count in stats['gloss_counts'].most_common(10):
                print(f"  {gloss}: {count} videos")
            
            # Display gloss counts in descending order (optional)
            # Uncomment the following lines if you want to see all gloss counts
            # print("\nAll gloss counts:")
            # for gloss, count in stats['gloss_counts'].most_common():
            #     print(f"  {gloss}: {count}")

if __name__ == "__main__":
    main()