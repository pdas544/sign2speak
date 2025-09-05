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
import json
import math

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Data Analysis and Balancing
def analyze_and_balance_dataset(metadata_path, max_samples_per_class=30, min_samples_per_class=5):
    """Analyze dataset distribution and create a balanced version"""
    metadata = pd.read_csv(metadata_path)
    
    # Analyze class distribution
    class_distribution = metadata['label'].value_counts()
    print("Original class distribution:")
    print(class_distribution)
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    class_distribution.plot(kind='bar')
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()
    
    # Create balanced dataset
    balanced_metadata = pd.DataFrame()
    
    for class_name in metadata['label'].unique():
        class_data = metadata[metadata['label'] == class_name]
        n_samples = len(class_data)
        
        if n_samples > max_samples_per_class:
            # Undersample majority classes
            class_data = class_data.sample(max_samples_per_class, random_state=42)
        elif n_samples < min_samples_per_class:
            # Skip classes with too few samples (consider collecting more data)
            print(f"Warning: Class '{class_name}' has only {n_samples} samples (min is {min_samples_per_class})")
            continue
        
        balanced_metadata = pd.concat([balanced_metadata, class_data])
    
    # Shuffle the balanced dataset
    balanced_metadata = balanced_metadata.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save balanced metadata
    balanced_path = metadata_path.replace('.csv', '_balanced.csv')
    balanced_metadata.to_csv(balanced_path, index=False)
    
    print(f"Balanced dataset created with {len(balanced_metadata)} samples")
    print("Balanced class distribution:")
    print(balanced_metadata['label'].value_counts())
    
    return balanced_path


# Step 2: Advanced Feature Engineering
def add_derivative_features(keypoints):
    """Add velocity and acceleration features to keypoints"""
    # Calculate velocity (difference between consecutive frames)
    velocity = torch.zeros_like(keypoints)
    if keypoints.shape[0] > 1:
        velocity[1:] = keypoints[1:] - keypoints[:-1]
    
    # Calculate acceleration (difference of velocity)
    acceleration = torch.zeros_like(keypoints)
    if keypoints.shape[0] > 2:
        acceleration[2:] = velocity[2:] - velocity[1:-1]
    
    # Combine original features with derivatives
    enhanced_features = torch.cat([keypoints, velocity, acceleration], dim=1)
    
    return enhanced_features

# Replace the current balancing approach with stratified sampling
def create_stratified_dataset(metadata_path, min_samples_per_class=10):
    metadata = pd.read_csv(metadata_path)
    
    # Group by class and split
    grouped = metadata.groupby('label')
    stratified_data = []
    
    for label, group in grouped:
        n_samples = len(group)
        if n_samples < min_samples_per_class:
            # Oversample minority classes
            oversampled = group.sample(min_samples_per_class, replace=True, random_state=42)
            stratified_data.append(oversampled)
        else:
            # Use all samples for majority classes
            stratified_data.append(group)
    
        stratified_metadata = pd.concat(stratified_data)

    stratified_metadata = stratified_metadata.sample(frac=1, random_state=42).reset_index(drop=True)

    stratified_data_path = metadata_path.replace('.csv', '_stratified.csv')
    stratified_metadata.to_csv(stratified_data_path, index=False)
    print(f"Stratified dataset created with {len(stratified_metadata)} samples")
    
    return stratified_data_path

# Step 3: Advanced Data Augmentation
class SimpleKeypointAugmentation:
    def __init__(self, augment_prob=0.7):
        self.augment_prob = augment_prob
        
    def __call__(self, keypoints):
        if random.random() > self.augment_prob:
            return keypoints
            
        # 1. Spatial augmentation
        if random.random() < 0.6:
            # Random scaling
            scale = random.uniform(0.8, 1.2)
            keypoints = keypoints * scale
            
            # Random translation
            translation = torch.randn(keypoints.shape[1]) * 0.05
            keypoints = keypoints + translation
            
        # 2. Random temporal cropping
        if random.random() < 0.5 and keypoints.shape[0] > 10:
            crop_start = random.randint(0, keypoints.shape[0] // 4)
            crop_end = random.randint(3 * keypoints.shape[0] // 4, keypoints.shape[0])
            keypoints = keypoints[crop_start:crop_end]
            
        # 3. Gaussian noise
        if random.random() < 0.5:
            noise = torch.randn_like(keypoints) * 0.03
            keypoints = keypoints + noise
            
        return keypoints

class AdvancedKeypointAugmentation:
    def __init__(self, augment_prob=0.8):
        self.augment_prob = augment_prob
        
    def __call__(self, keypoints):
        if random.random() > self.augment_prob:
            return keypoints
            
        # 1. Time warping with dynamic time warping (DTW) based approach
        if random.random() < 0.6 and keypoints.shape[0] > 10:
            keypoints = self.time_warp(keypoints)
            
        # 2. Spatial transformation
        if random.random() < 0.7:
            keypoints = self.spatial_transform(keypoints)
            
        # 3. Random masking of joints
        if random.random() < 0.5:
            keypoints = self.joint_masking(keypoints)
            
        # 4. Gaussian noise with adaptive scaling
        if random.random() < 0.6:
            noise = torch.randn_like(keypoints) * 0.02
            keypoints = keypoints + noise
            
        return keypoints
    
    def time_warp(self, keypoints):
        # Implement dynamic time warping based augmentation
        orig_len = keypoints.shape[0]
        warp_factor = random.uniform(0.8, 1.2)
        new_len = max(5, int(orig_len * warp_factor))
        
        # Resample using linear interpolation
        x_old = torch.linspace(0, 1, orig_len)
        x_new = torch.linspace(0, 1, new_len)
        
        keypoints_warped = torch.zeros(new_len, keypoints.shape[1])
        for i in range(keypoints.shape[1]):
            keypoints_warped[:, i] = torch.interp(x_new, x_old, keypoints[:, i])
            
        return keypoints_warped
    
    def spatial_transform(self, keypoints):
        # Apply random rotation, scaling, and translation
        # Reshape to [seq_len, num_joints, 3]
        num_joints = 543  # MediaPipe Holistic has 543 landmarks
        keypoints_3d = keypoints.view(-1, num_joints, 3)
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-0.2, 0.2)
            rot_mat = torch.tensor([
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1]
            ], dtype=keypoints.dtype)
            keypoints_3d = torch.matmul(keypoints_3d, rot_mat)
        
        # Random scaling
        scale = random.uniform(0.8, 1.2)
        keypoints_3d = keypoints_3d * scale
        
        # Random translation
        translation = torch.randn(3) * 0.05
        keypoints_3d = keypoints_3d + translation
        
        # Flatten back to original shape
        return keypoints_3d.view(-1, num_joints * 3)
    
    def joint_masking(self, keypoints):
        # Randomly mask out some joints to make model robust to missing data
        num_joints = 543
        mask_prob = random.uniform(0.1, 0.3)
        
        # Reshape to [seq_len, num_joints, 3]
        keypoints_3d = keypoints.view(-1, num_joints, 3)
        
        # Create mask
        mask = torch.rand(num_joints) > mask_prob
        keypoints_3d = keypoints_3d * mask.unsqueeze(0).unsqueeze(2)
        
        # Flatten back to original shape
        return keypoints_3d.view(-1, num_joints * 3)

# Step 4: Transformer Model Architecture
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

class SignLanguageTransformer(nn.Module):
    def __init__(self, input_size, num_classes, d_model=256, nhead=8, num_layers=3, dropout=0.3):
        super(SignLanguageTransformer, self).__init__()
        
        # Feature projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Global average pooling and classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )
        
    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.size()
        
        # Project input to higher dimension
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Create padding mask
        max_len = x.size(1)
        mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        mask = mask.to(x.device)
        
        # Apply transformer
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global average pooling (ignore padded elements)
        # Create mask for pooling
        mask = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x_masked = x * (~mask).float()  # Zero out padded elements
        
        # Sum along sequence dimension and divide by actual lengths
        pooled = x_masked.sum(dim=1) / lengths.unsqueeze(1).float()
        
        # Classify
        output = self.classifier(pooled)
        
        return output

class EnhancedSignLanguageTransformer(nn.Module):
    def __init__(self, input_size, num_classes, d_model=256, nhead=16, num_layers=4, dropout=0.2):
        super(EnhancedSignLanguageTransformer, self).__init__()
        
        # Input projection with layer normalization
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder with increased capacity
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Multi-head attention pooling instead of simple average pooling
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Classifier with additional layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model//2),
            nn.Linear(d_model//2, num_classes)
        )
        
    def forward(self, x, lengths):
        batch_size, seq_len, input_size = x.size()
        
        # Project input to higher dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Create padding mask
        max_len = x.size(1)
        mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
        mask = mask.to(x.device)
        
        # Apply transformer
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use attention pooling instead of average pooling
        # Create a query for pooling
        query = torch.mean(x, dim=1, keepdim=True)
        pooled, _ = self.attention_pool(query, x, x, key_padding_mask=mask)
        pooled = pooled.squeeze(1)
        
        # Classify
        output = self.classifier(pooled)
        
        return output

# Dataset with enhanced features and augmentation
class EnhancedSignLanguageDataset(Dataset):
    def __init__(self, metadata_path, keypoints_dir, gloss_to_idx, split=None, augment=False, enhanced_features=True):
        self.metadata = pd.read_csv(metadata_path)
        self.keypoints_dir = keypoints_dir
        
        if split:
            self.metadata = self.metadata[self.metadata['split'] == split]
            
        self.gloss_to_idx = gloss_to_idx
        self.augment = augment
        self.augmenter = SimpleKeypointAugmentation() if augment else None
        self.enhanced_features = enhanced_features
        
        # Create a fixed feature dropout mask that will be applied to all samples
        # We'll initialize it when we see the first sample
        self.feature_dropout_mask = None
        self.original_feature_dim = None
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        keypoints_path = row['file_path']
        label = row['label']
        
        # Load keypoints
        keypoints = torch.load(keypoints_path, weights_only=True)
        original_shape = keypoints.shape
        # print(f"Original shape: {original_shape}")
        
        # Apply augmentation if training
        if self.augment and self.augmenter:
            keypoints = self.augmenter(keypoints)
            # print(f"After augmentation: {keypoints.shape}")
        
        # Add derivative features if enabled
        if self.enhanced_features:
            keypoints = add_derivative_features(keypoints)
            # print(f"After derivative features: {keypoints.shape}")
        
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

# Use focal loss with adaptive gamma
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma_range=(2, 5), reduction='mean'):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_range = gamma_range
        self.reduction = reduction
        self.gamma = gamma_range[0]  # Start with minimum gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Gradually increase gamma during training
        self.gamma = min(self.gamma_range[1], self.gamma + 0.01)
        focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Step 5: Advanced Training Techniques
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # If targets are one-hot encoded, convert to class indices
        if targets.dim() > 1:
            targets = targets.argmax(dim=1)
            
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# def smooth_labels(labels, n_classes, smoothing=0.1):
#     smoothed = torch.full_like(labels.float(), smoothing / (n_classes - 1))
#     smoothed.scatter_(1, labels.unsqueeze(1), 1 - smoothing)
#     return smoothed
def smooth_labels(labels, n_classes, smoothing=0.1):
    """
    Convert labels to one-hot format and apply label smoothing
    """
    # labels is 1D tensor of class indices
    batch_size = labels.size(0)
    
    # Create one-hot encoded labels
    one_hot = torch.zeros(batch_size, n_classes, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    
    # Apply label smoothing
    smoothed = one_hot * (1 - smoothing) + smoothing / n_classes
    
    return smoothed

# Main training function
def train_transformer_model(metadata_path, keypoints_dir, output_dir, num_epochs=150):
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
    
    # Create datasets with enhanced features and augmentation
    train_dataset = EnhancedSignLanguageDataset(metadata_path, keypoints_dir, gloss_to_idx, split='train', augment=True, enhanced_features=True)
    val_dataset = EnhancedSignLanguageDataset(metadata_path, keypoints_dir, gloss_to_idx, split='val', augment=False, enhanced_features=True)
    test_dataset = EnhancedSignLanguageDataset(metadata_path, keypoints_dir, gloss_to_idx, split='test', augment=False, enhanced_features=True)
    
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
    input_size = 543 * 3 * 3  # Original + velocity + acceleration
    model = EnhancedSignLanguageTransformer(input_size, num_classes).to(device)
    
    # Use focal loss with class weights
    criterion = AdaptiveFocalLoss(alpha=class_weights, gamma_range=(2,5))
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < 10:
            return (epoch + 1) / 10  # Warmup
        elif epoch < 100:
            return 1.0  # Constant LR
        else:
            # Cosine annealing for the rest
            return 0.5 * (1 + math.cos(math.pi * (epoch - 100) / (num_epochs - 100)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    max_grad_norm = 1.0  # Gradient clipping value
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            running_loss += loss.item() * sequences.size(0)
        
        # Update learning rate
        scheduler.step()
        
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

# Main execution

if __name__ == "__main__":
    # Step 1: Data Analysis and Balancing
    original_metadata_path = "processed/metadata.csv"
    balanced_metadata_path = create_stratified_dataset(original_metadata_path)
    
    # Step 2-5: Train the transformer model
    keypoints_dir = "processed/keypoints"
    output_dir = "transformer_outputs"
    
    model, test_accuracy = train_transformer_model(
        balanced_metadata_path, keypoints_dir, output_dir, num_epochs=150
    )
    
    print(f"Training completed. Test accuracy: {test_accuracy:.2f}%")