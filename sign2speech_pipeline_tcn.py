import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import argparse
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Chomp1d(nn.Module):
    """Chomp1d module for causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

# Model Architecture
class TemporalBlock(nn.Module):
    """Temporal block for the TCN architecture with proper padding handling"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """Temporal Convolutional Network with proper initialization"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                    dilation=dilation_size, 
                                    padding=(kernel_size-1)*dilation_size, 
                                    dropout=dropout)]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class SignLanguageModel(nn.Module):
    """Complete model for sign language recognition with fixed TCN"""
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

# Dataset and Data Loading
class SignLanguageDataset(Dataset):
    def __init__(self, metadata_path, keypoints_dir, gloss_to_idx, split=None):
        self.metadata = pd.read_csv(metadata_path)
        self.keypoints_dir = keypoints_dir
        
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
        
        return keypoints, label_idx, keypoints.shape[0]

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    sequences, labels, lengths = zip(*batch)
    
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    
    return padded_sequences, torch.tensor(labels), torch.tensor(lengths)

# Training Function
def train_model(metadata_path, keypoints_dir, output_dir, num_epochs=50):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Create gloss to index mapping
    glosses = sorted(metadata['label'].unique())
    gloss_to_idx = {gloss: idx for idx, gloss in enumerate(glosses)}
    num_classes = len(glosses)
    
    print(f"Found {num_classes} classes: {glosses}")
    
    # Save gloss mapping
    with open(os.path.join(output_dir, 'gloss_mapping.json'), 'w') as f:
        json.dump(gloss_to_idx, f)
    
    # Create datasets
    train_dataset = SignLanguageDataset(metadata_path, keypoints_dir, gloss_to_idx, split='train')
    val_dataset = SignLanguageDataset(metadata_path, keypoints_dir, gloss_to_idx, split='val')
    test_dataset = SignLanguageDataset(metadata_path, keypoints_dir, gloss_to_idx, split='test')
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # Initialize model
    input_size = 543 * 3  # 543 landmarks (pose+hands+face) * 3 coordinates (x,y,z)
    model = SignLanguageModel(input_size, num_classes).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training parameters
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for sequences, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * sequences.size(0)
        
        # Calculate average training loss
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences, lengths)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        val_accuracies.append(accuracy)
        
        # Update learning rate
        scheduler.step(accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, 'models', 'best_model.pth'))
            print(f"New best model saved with accuracy: {accuracy:.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Test phase
    print("Testing the best model...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'models', 'best_model.pth')))
    model.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences, lengths)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    # Save test results
    test_results = {
        'accuracy': test_accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'class_names': glosses
    }
    
    torch.save(test_results, os.path.join(output_dir, 'test_results.pth'))
    
    # Generate evaluation plots
    generate_evaluation_plots(test_results, output_dir)
    
    return model, test_accuracy

# Evaluation and Visualization
def generate_evaluation_plots(test_results, output_dir):
    accuracy = test_results['accuracy']
    predictions = test_results['predictions']
    labels = test_results['labels']
    class_names = test_results['class_names']
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Classification report
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names=class_names))
    
    # Per-class accuracy
    class_accuracy = 100 * cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(class_names)), class_accuracy)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'))

# TTS Integration (simplified)
def setup_tts_integration(model_path, gloss_mapping_path, output_dir):
    # Load gloss mapping
    with open(gloss_mapping_path, 'r') as f:
        gloss_to_idx = json.load(f)
    
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    glosses = [idx_to_gloss[i] for i in range(len(idx_to_gloss))]
    
    # Initialize model
    input_size = 543 * 3
    model = SignLanguageModel(input_size, len(glosses)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create TTS output directory
    tts_output_dir = os.path.join(output_dir, 'audio_samples')
    os.makedirs(tts_output_dir, exist_ok=True)
    
    print(f"TTS integration setup complete. Glosses: {glosses}")
    return model, glosses, tts_output_dir

# Main function
def main():
    parser = argparse.ArgumentParser(description='Sign2Speech Training Pipeline')
    parser.add_argument('--metadata_path', type=str, default='processed/metadata.csv',
                        help='Path to metadata CSV file')
    parser.add_argument('--keypoints_dir', type=str, default='processed/keypoints',
                        help='Directory containing keypoint files')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--setup_tts', action='store_true',
                        help='Setup TTS integration')
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting model training...")
        model, test_accuracy = train_model(
            args.metadata_path, 
            args.keypoints_dir, 
            args.output_dir, 
            args.epochs
        )
        print(f"Training completed. Test accuracy: {test_accuracy:.2f}%")
    
    if args.evaluate:
        print("Evaluating model...")
        # Load test results if they exist
        test_results_path = os.path.join(args.output_dir, 'test_results.pth')
        if os.path.exists(test_results_path):
            test_results = torch.load(test_results_path)
            generate_evaluation_plots(test_results, args.output_dir)
        else:
            print("No test results found. Please train the model first.")
    
    if args.setup_tts:
        print("Setting up TTS integration...")
        model_path = os.path.join(args.output_dir, 'models', 'best_model.pth')
        gloss_mapping_path = os.path.join(args.output_dir, 'gloss_mapping.json')
        
        if os.path.exists(model_path) and os.path.exists(gloss_mapping_path):
            model, glosses, tts_output_dir = setup_tts_integration(
                model_path, gloss_mapping_path, args.output_dir
            )
            print("TTS integration is ready. You can now use the TTS functions.")
        else:
            print("Model or gloss mapping not found. Please train the model first.")

if __name__ == "__main__":
    main()