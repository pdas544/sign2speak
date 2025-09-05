import pandas as pd

def analyze_and_balance_dataset(metadata_path, max_samples_per_class=50):
    metadata = pd.read_csv(metadata_path)
    
    # Analyze class distribution
    class_distribution = metadata['label'].value_counts()
    print("Class distribution:")
    print(class_distribution)
    
    # Balance the dataset by undersampling majority classes
    balanced_metadata = pd.DataFrame()
    for class_name in metadata['label'].unique():
        class_data = metadata[metadata['label'] == class_name]
        if len(class_data) > max_samples_per_class:
            class_data = class_data.sample(max_samples_per_class, random_state=42)
        balanced_metadata = pd.concat([balanced_metadata, class_data])
    
    # Shuffle the balanced dataset
    balanced_metadata = balanced_metadata.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save balanced metadata
    balanced_metadata.to_csv(metadata_path.replace('.csv', '_balanced.csv'), index=False)
    
    return balanced_metadata

analyze_and_balance_dataset('processed/metadata.csv')