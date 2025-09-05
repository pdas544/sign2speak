import json
import os
from collections import defaultdict

def create_filtered_json(selected_glosses, output_filename='filtered_annotations_selected_glosses.json'):
    """
    Create a filtered JSON file containing only entries for the selected glosses
    """
    # Define the input files
    json_files = {
        'train': 'MSASL_train.json',
        'val': 'MSASL_val.json',
        'test': 'MSASL_test.json'
    }
    
    # Initialize counters
    gloss_counts = defaultdict(lambda: defaultdict(int))
    filtered_data = []
    
    # Process each file
    for split, filename in json_files.items():
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping...")
            continue
            
        with open(filename, 'r') as f:
            data = json.load(f)
            
        for item in data:
            gloss = item.get('text')
            if gloss in selected_glosses:
                # Add split information to the item
                item['split'] = split
                filtered_data.append(item)
                gloss_counts[gloss][split] += 1
    
    # Save the filtered data
    with open(output_filename, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    # Print summary
    print("Filtered JSON created successfully!")
    print(f"Total entries: {len(filtered_data)}")
    print("\nGloss counts per split:")
    for gloss in selected_glosses:
        print(f"{gloss}: Train={gloss_counts[gloss]['train']}, Val={gloss_counts[gloss]['val']}, Test={gloss_counts[gloss]['test']}")
    
    return output_filename

# Your selected glosses
selected_glosses = [
    'teacher', 'happy', 'nice', 'good', 
    'no', 'go', 'what', 'like', 'hello',
    'white', 'friend', 'big', 'beautiful', 'boy', 'sister'
]

if __name__ == "__main__":
    create_filtered_json(selected_glosses)