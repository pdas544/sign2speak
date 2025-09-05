import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class TemporalBlock(nn.Module):
    """Temporal block for the TCN architecture"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                    dilation=dilation_size, padding=(kernel_size-1)*dilation_size, 
                                    dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class SignLanguageModel(nn.Module):
    """Complete model for sign language recognition"""
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super(SignLanguageModel, self).__init__()
        
        # TCN parameters
        tcn_channels = [64, 128]
        
        # TCN for temporal feature extraction
        self.tcn = TCN(input_size, tcn_channels, kernel_size=3, dropout=dropout)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=tcn_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        
        # Reshape for TCN: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Apply TCN
        tcn_out = self.tcn(x)  # (batch_size, tcn_channels[-1], seq_len)
        
        # Reshape for LSTM: (batch_size, seq_len, tcn_channels[-1])
        tcn_out = tcn_out.transpose(1, 2)
        
        # Pack padded sequences
        packed_input = nn.utils.rnn.pack_padded_sequence(
            tcn_out, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Apply LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Unpack sequences
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        
        # Classify
        output = self.classifier(context_vector)
        
        return output

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