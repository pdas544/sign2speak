import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import os
import random
import torch.nn.functional as F
import argparse
import json

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simplified Model Architecture
class SimplifiedSignLanguageModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super(SimplifiedSignLanguageModel, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        
        # Reshape for CNN: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Apply CNN
        cnn_out = self.cnn(x)  # (batch_size, 128, 1)
        
        # Flatten
        cnn_out = cnn_out.view(batch_size, -1)
        
        # Classify
        output = self.classifier(cnn_out)
        
        return output

# Data Augmentation
class KeypointAugmentation:
    def __init__(self, augment_prob=0.5):
        self.augment_prob = augment_prob
        
    def __call__(self, keypoints):
        if random.random() > self.augment_prob:
            return keypoints
            
        # Time warping (slight speed changes)
        if random.random() < 0.3 and keypoints.shape[0] > 1:
            orig_len = keypoints.shape[0]
            new_len = random.randint(max(1, int(orig_len * 0.9)), int(orig_len * 1.1))
            
            # Reshape for interpolation: [seq_len, features] -> [1, features, seq_len]
            keypoints_reshaped = keypoints.transpose(0, 1).unsqueeze(0)
            
            # Interpolate
            keypoints_reshaped = F.interpolate(
                keypoints_reshaped, 
                size=new_len, 
                mode='linear', 
                align_corners=False
            )
            
            # Reshape back: [1, features, new_len] -> [new_len, features]
            keypoints = keypoints_reshaped.squeeze(0).transpose(0, 1)
            
        # Add noise
        if random.random() < 0.4:
            noise = torch.randn_like(keypoints) * 0.01
            keypoints = keypoints + noise
            
        # Random horizontal flip (mirroring)
        if random.random() < 0.5 and keypoints.shape[1] % 3 == 0:
            # Reshape to [seq_len, num_landmarks, 3]
            keypoints_3d = keypoints.view(keypoints.shape[0], -1, 3)
            
            # Flip x coordinates (assuming x is the first coordinate)
            keypoints_3d[:, :, 0] = 1 - keypoints_3d[:, :, 0]  # Flip x
            
            # Reshape back
            keypoints = keypoints_3d.view(keypoints.shape[0], -1)
            
        return keypoints

# Dataset with Augmentation
class SignLanguageDataset(Dataset):
    def __init__(self, metadata_path, keypoints_dir, gloss_to_idx, split=None, augment=False):
        self.metadata = pd.read_csv(metadata_path)
        self.keypoints_dir = keypoints_dir
        
        if split:
            self.metadata = self.metadata[self.metadata['split'] == split]
            
        self.gloss_to_idx = gloss_to_idx
        self.augment = augment
        self.augmenter = KeypointAugmentation() if augment else None
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        keypoints_path = row['file_path']
        label = row['label']
        
        # Load keypoints
        keypoints = torch.load(keypoints_path)
        
        # Apply augmentation if training
        if self.augment and self.augmenter:
            keypoints = self.augmenter(keypoints)
        
        # Convert label to index
        label_idx = self.gloss_to_idx[label]
        
        return keypoints, label_idx, keypoints.shape[0]

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    sequences, labels, lengths = zip(*batch)
    
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    
    return padded_sequences, torch.tensor(labels), torch.tensor(lengths)

# Calculate class weights
def calculate_class_weights(metadata_path, split='train'):
    metadata = pd.read_csv(metadata_path)
    train_metadata = metadata[metadata['split'] == split]
    
    # Get all labels
    labels = train_metadata['label'].values
    unique_labels = np.unique(labels)
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    return torch.tensor(class_weights, dtype=torch.float).to(device)

# Training function
def train_model(metadata_path, keypoints_dir, output_dir, num_epochs=100):
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
    
    # Create datasets with augmentation for training
    train_dataset = SignLanguageDataset(metadata_path, keypoints_dir, gloss_to_idx, split='train', augment=True)
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
    
    # Calculate class weights
    class_weights = calculate_class_weights(metadata_path)
    
    # Initialize model
    input_size = 543 * 3  # 543 landmarks (pose+hands+face) * 3 coordinates (x,y,z)
    model = SimplifiedSignLanguageModel(input_size, num_classes).to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # Training parameters
    best_accuracy = 0.0
    patience_counter = 0
    patience = 20
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
        
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'models', 'best_model.pth'))
            print(f"New best model saved with accuracy: {accuracy:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
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
    print(classification_report(labels, predictions, target_names=class_names, zero_division=0))
    
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
    # parser.add_argument('--setup_tts', action='store_true',
    #                     help='Setup TTS integration')
    
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
    
    # if args.setup_tts:
    #     print("Setting up TTS integration...")
    #     model_path = os.path.join(args.output_dir, 'models', 'best_model.pth')
    #     gloss_mapping_path = os.path.join(args.output_dir, 'gloss_mapping.json')
        
    #     if os.path.exists(model_path) and os.path.exists(gloss_mapping_path):
    #         model, glosses, tts_output_dir = setup_tts_integration(
    #             model_path, gloss_mapping_path, args.output_dir
    #         )
    #         print("TTS integration is ready. You can now use the TTS functions.")
    #     else:
    #         print("Model or gloss mapping not found. Please train the model first.")

if __name__ == "__main__":
    main()