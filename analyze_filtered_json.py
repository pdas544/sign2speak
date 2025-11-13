import json

# Load the JSON data from the file
with open('annotations/filtered_annotations_selected_glosses.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract all values for the key "clean_text"
clean_text_values = [item['clean_text'] for item in data if 'clean_text' in item]

# Get the set of unique values
unique_clean_texts = set(clean_text_values)

# Print the count of unique values
print("Count of unique 'clean_text' values:", len(unique_clean_texts))