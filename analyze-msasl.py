import json
from collections import Counter

def analyze_glosses_across_splits(gloss_list):
    """
    Analyze specific glosses across all dataset splits
    """
    files = {
        'train': 'MSASL_train.json',
        'val': 'MSASL_val.json', 
        'test': 'MSASL_test.json'
    }
    
    results = {}
    
    for gloss in gloss_list:
        results[gloss] = {}
        for split, filename in files.items():
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    count = sum(1 for item in data if item.get('text') == gloss)
                    results[gloss][split] = count
            except:
                results[gloss][split] = 0
    
    return results

# Glosses to analyze
glosses_to_check = [
    'teacher', 'happy', 'nice', 'good', 'sorry', 
    'no', 'go', 'what', 'like', 'hello',
    'white', 'friend', 'big', 'beautiful', 'boy', 'sister'
]

results = analyze_glosses_across_splits(glosses_to_check)

print("Gloss Counts Across All Splits:")
print("=" * 50)
for gloss, counts in results.items():
    print(f"{gloss}: Train={counts.get('train', 0)}, Val={counts.get('val', 0)}, Test={counts.get('test', 0)}")