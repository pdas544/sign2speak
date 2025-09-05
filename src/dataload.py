class SignLanguageDataset(Dataset):
    def __init__(self, metadata_path, keypoints_dir, gloss_to_idx, split=None):
        self.metadata = pd.read_csv(metadata_path)
        self.keypoints_dir = keypoints_dir
        
        # Filter by split if specified
        if split:
            self.metadata = self.metadata[self.metadata['split'] == split]
            
        self.gloss_to_idx = gloss_to_idx
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        keypoints_path = row['file_path']
        label = row['label']
        
        # Load keypoints
        keypoints = torch.load(keypoints_path)
        
        # Convert label to index
        label_idx = self.gloss_to_idx[label]
        
        return keypoints, label_idx, keypoints.shape[0]  # Return sequence length
    
def collate_fn(batch):
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: x[2], reverse=True)
    sequences, labels, lengths = zip(*batch)
    
    # Pad sequences
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    
    return padded_sequences, torch.tensor(labels), torch.tensor(lengths)