import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import json
from collections import defaultdict

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
class Config:
    # Model architecture
    input_size = 1629  # From your keypoint data
    hidden_size = 256
    num_layers = 2
    num_classes = 15   # Your current glosses
    dropout = 0.3
    
    # Training parameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    weight_decay = 1e-4

    gloss_to_idx = {
            'teacher': 0, 'happy': 1, 'nice': 2, 'good': 3,
        'no': 4, 'go': 5, 'what': 6, 'like': 7, 'hello': 8,
        'white': 9, 'friend': 10, 'big': 11, 'beautiful': 12, 'boy': 13, 'sister': 14
        }
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}

config = Config()

# CNN-LSTM Model (Better for your dataset size)
class SignLanguageCNN_LSTM(nn.Module):
    def __init__(self, config):
        super(SignLanguageCNN_LSTM, self).__init__()
        
        # CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(config.input_size, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.MaxPool1d(2),
            
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_classes)
        )
        
    def forward(self, x, src_key_padding_mask=None):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # CNN expects (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        cnn_features = self.cnn(x)  # (batch_size, 512, reduced_seq_len)
        cnn_features = cnn_features.transpose(1, 2)  # (batch_size, reduced_seq_len, 512)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_features)  # (batch_size, reduced_seq_len, hidden_size*2)
        
        # Use mean pooling over time
        pooled = torch.mean(lstm_out, dim=1)
        
        # Classification
        return self.classifier(pooled)

# Dataset Class
class SignLanguageDataset(Dataset):
    def __init__(self, metadata_path, keypoints_dir,split='train'):
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['split'] == split]
        self.keypoints_dir = keypoints_dir
        self.split = split
        
        
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        keypoints_path = row['file_path']
        
        try:
            # Load keypoints
            keypoints = torch.load(keypoints_path, weights_only=True)
            
            # Handle different tensor shapes
            if keypoints.dim() == 1:
                keypoints = keypoints.unsqueeze(0)
            elif keypoints.dim() == 3:
                # If it's (seq_len, 21, 3), flatten to (seq_len, 63)
                keypoints = keypoints.view(keypoints.shape[0], -1)
            
            seq_len = keypoints.shape[0]
            label = config.gloss_to_idx[row['label']]
            
            return keypoints, label, seq_len
            
        except Exception as e:
            print(f"Error loading {keypoints_path}: {e}")
            # Return a dummy sample
            return torch.zeros(10, config.input_size), 0, 10

# Collate function
def collate_fn(batch):
    keypoints, labels, seq_lens = zip(*batch)
    
    # Pad sequences
    keypoints_padded = pad_sequence(keypoints, batch_first=True, padding_value=0)
    
    # Create padding mask
    max_len = keypoints_padded.shape[1]
    padding_mask = torch.zeros(len(keypoints_padded), max_len, dtype=torch.bool)
    for i, seq_len in enumerate(seq_lens):
        if seq_len < max_len:
            padding_mask[i, seq_len:] = True
    
    labels = torch.tensor(labels)
    
    return keypoints_padded, labels, padding_mask

def create_dataloaders(metadata_path, keypoints_dir, batch_size):
    train_dataset = SignLanguageDataset(metadata_path, keypoints_dir, 'train')
    val_dataset = SignLanguageDataset(metadata_path, keypoints_dir, 'val')
    test_dataset = SignLanguageDataset(metadata_path, keypoints_dir, 'test')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=2
    )
    
    return train_loader, val_loader, test_loader

# Training function
def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs):
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_acc = 0.0
    best_model_path = "./best_model_lstm.pth"
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for keypoints, labels, padding_mask in pbar:
            keypoints = keypoints.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(keypoints, src_key_padding_mask=padding_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100 * correct / total
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for keypoints, labels, padding_mask in pbar:
                keypoints = keypoints.to(device)
                labels = labels.to(device)
                padding_mask = padding_mask.to(device)
                
                outputs = model(keypoints, src_key_padding_mask=padding_mask)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / (pbar.n + 1),
                    'acc': 100 * correct / total
                })
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print()
    
    return train_losses, val_losses, train_accs, val_accs


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for keypoints, labels, padding_mask in tqdm(test_loader, desc='Testing'):
            keypoints = keypoints.to(device)
            labels = labels.to(device)
            padding_mask = padding_mask.to(device)
            
            outputs = model(keypoints, src_key_padding_mask=padding_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
    # Get the unique classes that appear in the predictions and labels
    unique_labels = np.unique(all_labels + all_preds)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                               labels=list(range(config.num_classes)),
                               target_names=list(config.gloss_to_idx.keys())))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(config.num_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(config.gloss_to_idx.keys()),
                yticklabels=list(config.gloss_to_idx.keys()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return accuracy
# Plot training history
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/training_history.png')
    plt.show()

# Inference function
def predict_sign(model, keypoints):
    model.eval()
    
    with torch.no_grad():
        # Preprocess keypoints
        if isinstance(keypoints, np.ndarray):
            keypoints = torch.from_numpy(keypoints).float()
        
        # Add batch dimension
        if keypoints.dim() == 2:
            keypoints = keypoints.unsqueeze(0)
        
        keypoints = keypoints.to(device)
        
        # Forward pass
        outputs = model(keypoints)
        _, predicted = torch.max(outputs.data, 1)
        
        # Convert to gloss
        gloss = config.idx_to_gloss[predicted.item()]
        
        return gloss, outputs.softmax(dim=1).cpu().numpy()

# Main execution
def main():
    # Paths
    metadata_path = "processed/metadata.csv"
    keypoints_dir = "processed/keypoints"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path, keypoints_dir, config.batch_size
    )
    
    # Initialize model
    model = SignLanguageCNN_LSTM(config).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, optimizer, scheduler, config.num_epochs
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Load best model
    model.load_state_dict(torch.load("best_model_lstm.pth", map_location=device))
    
    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_loader)
    

    # Example inference
    print("\nExample inference:")
    sample_keypoints, sample_label, sample_seq_len = next(iter(test_loader))
    predicted_gloss, confidence = predict_sign(model, sample_keypoints[0])
    true_gloss = config.idx_to_gloss[sample_label[0].item()]
    
    print(f"True gloss: {true_gloss}")
    print(f"Predicted gloss: {predicted_gloss}")
    print(f"Confidence: {confidence[0].max():.4f}")
    
    # Convert to speech

    # print("\nGenerating speech...")
    # tts_output = text_to_speech(predicted_gloss)
    # if tts_output:
    #     print(f"Speech saved to {tts_output}")
    # else:
    #     print("Speech generation failed")

if __name__ == "__main__":
    main()