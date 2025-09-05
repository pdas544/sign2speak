import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
class Config:
    # Model parameters
    d_model = 256
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    
    # Training parameters
    batch_size = 16  # Reduced to handle memory issues
    learning_rate = 0.0001
    num_epochs = 50
    weight_decay = 0.01
    
    # Data parameters
    max_seq_length = 150
    num_classes = 16
    input_features = 1629  # From your metadata
    gloss_to_idx = {
        'teacher': 0, 'happy': 1, 'nice': 2, 'good': 3, 'sorry': 4, 
        'no': 5, 'go': 6, 'what': 7, 'like': 8, 'hello': 9,
        'white': 10, 'friend': 11, 'big': 12, 'beautiful': 13, 'boy': 14, 'sister': 15
    }
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}

config = Config()

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Transformer Model
class SignLanguageTransformer(nn.Module):
    def __init__(self, config):
        super(SignLanguageTransformer, self).__init__()
        self.config = config
        
        # Input projection with batch normalization
        self.input_projection = nn.Linear(config.input_features, config.d_model)
        self.bn1 = nn.BatchNorm1d(config.d_model)
        
        # Additional feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Attention pooling instead of just using CLS token
        self.attention_pool = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Tanh()
        )
        
        # Output layers with more capacity
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_classes)
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_key_padding_mask=None):
        batch_size, seq_len, _ = src.shape
        
        # Project input to d_model dimension
        src = self.input_projection(src)
        
        # Apply batch norm across features
        src = src.transpose(1, 2)
        src = self.bn1(src)
        src = src.transpose(1, 2)
        
        # Additional feature extraction
        src = self.feature_extractor(src)
        
        # Add positional encoding
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # Attention pooling
        attention_weights = self.attention_pool(encoded).squeeze(-1)
        if src_key_padding_mask is not None:
            attention_weights = attention_weights.masked_fill(src_key_padding_mask, -1e9)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Weighted sum of all tokens
        pooled = torch.sum(encoded * attention_weights.unsqueeze(-1), dim=1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits

# Dataset Class
class SignLanguageDataset(Dataset):
    def __init__(self, metadata_path, keypoints_dir, split='train'):
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['split'] == split]
        self.keypoints_dir = keypoints_dir
        self.split = split
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        keypoints_path = row['file_path']
        
        # Load keypoints
        keypoints = torch.load(keypoints_path, weights_only=True)
        
        # Get sequence length and ensure it's 2D
        if keypoints.dim() == 1:
            # Handle 1D tensors by reshaping
            seq_len = 1
            keypoints = keypoints.unsqueeze(0)
        else:
            seq_len = keypoints.shape[0]
        
        # Get label
        label = config.gloss_to_idx[row['label']]
        
        return keypoints, label, seq_len

# Collate function for padding
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

# Create datasets and dataloaders
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
    best_model_path = "best_model.pth"
    
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

# TTS Integration (Coqui TTS)
def text_to_speech(text, output_path="output.wav"):
    try:
        # Import Coqui TTS
        from TTS.api import TTS
        
        # Initialize TTS
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", 
                  vocoder_name="vocoder_models/en/ljspeech/hifigan_v2",
                  gpu=True if torch.cuda.is_available() else False)
        
        # Generate speech
        tts.tts_to_file(text=text, file_path=output_path)
        
        print(f"Speech generated and saved to {output_path}")
        return output_path
        
    except ImportError:
        print("Coqui TTS not installed. Please install it with: pip install TTS")
        return None
    except Exception as e:
        print(f"Error in TTS: {e}")
        return None

# Main function
def main():
    # Paths
    metadata_path = "processed/metadata.csv"
    keypoints_dir = "processed/keypoints"
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path, keypoints_dir, config.batch_size
    )
    
    # Initialize model
    model = SignLanguageTransformer(config).to(device)
    
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
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    
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
    if test_accuracy >= 0.8:  # Only use TTS if model is accurate enough
        print("\nGenerating speech...")
        tts_output = text_to_speech(predicted_gloss)
        if tts_output:
            print(f"Speech saved to {tts_output}")
    else:
        print("Model accuracy is below 80%, skipping TTS generation.")

if __name__ == "__main__":
    main()