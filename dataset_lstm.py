import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class ASLKeypointDataset(Dataset):
    def __init__(self, keypoints_dir, sequence_length=30):
        self.keypoints_files = glob.glob(os.path.join(keypoints_dir, "*_keypoints.npy"))
        self.sequence_length = sequence_length
        
        # Extract labels from filenames
        self.labels = []
        self.data_files = []
        
        for file_path in self.keypoints_files:
            filename = os.path.basename(file_path)
            # Extract label from filename (assuming format: label_index_keypoints.npy)
            base = filename.split('.')[0]
            label = base.split('_')[0]
            self.labels.append(label)
            self.data_files.append(file_path)
            # print(self.labels[:10])
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load keypoints
        keypoints = np.load(self.data_files[idx])
        
        # Ensure consistent sequence length
        if keypoints.shape[0] != self.sequence_length:
            if keypoints.shape[0] > self.sequence_length:
                keypoints = keypoints[:self.sequence_length]
            else:
                # Pad with zeros
                padding = np.zeros((self.sequence_length - keypoints.shape, keypoints.shape[21]))
                keypoints = np.vstack([keypoints, padding])
        
        # Convert to tensor
        keypoints = torch.FloatTensor(keypoints)
        label = torch.LongTensor([self.encoded_labels[idx]])
        
        return keypoints, label
    
    def get_label_names(self):
        return self.label_encoder.classes_

def create_data_loaders(keypoints_dir, batch_size=32, train_split=0.8):
    """Create train and validation data loaders"""
    dataset = ASLKeypointDataset(keypoints_dir)
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.num_classes, dataset.get_label_names()
