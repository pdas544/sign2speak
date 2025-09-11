import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import os

from dataset_lstm import create_data_loaders
from model_lstm import ASLKeypointLSTM

class ASLTrainer:
    def __init__(self, model, device, train_loader, val_loader, label_names):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_names = label_names
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, verbose=True
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        
        
        for batch_data, batch_labels in tqdm(self.train_loader, desc="Training"):
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device).squeeze(1)

            # print(f"batch_labels shape: {batch_labels.shape}")
            # print(f"batch_labels example: {batch_labels}")
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(self.val_loader, desc="Validation"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device).squeeze(1)
                
                # Forward pass
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy * 100, all_labels, all_predictions
    
    def train(self, num_epochs=50):
        best_val_accuracy = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)

            
            
            # Training
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Validation
            val_loss, val_acc, val_labels, val_predictions = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(self.model.state_dict(), 'best_asl_model.pth')
                print(f"New best model saved with accuracy: {val_acc:.2f}%")
        
        return val_labels, val_predictions
    
    def plot_training_history(self):
        """Plot training and validation metrics"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Training Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history_lstm.png')
        plt.show()
    
    def evaluate_model(self, true_labels, predictions):
        """Comprehensive model evaluation"""
        # Accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Precision, Recall, F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names[:len(np.unique(true_labels))],
                   yticklabels=self.label_names[:len(np.unique(true_labels))])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix_lstm.png')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

# Training script
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, num_classes, label_names = create_data_loaders(
        keypoints_dir="./keypoints_full",
        batch_size=16,
        train_split=0.8
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {label_names}")
    
    # Calculate input size (total keypoints features)
    sample_data, _ = next(iter(train_loader))
    input_size = sample_data.shape[-1]  # Feature dimension
    print(f"Input size: {input_size}")
    
    # Create model
    model = ASLKeypointLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.5
    )
    
    # Create trainer
    trainer = ASLTrainer(model, device, train_loader, val_loader, label_names)
    
    # Train model
    val_labels, val_predictions = trainer.train(num_epochs=30)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    metrics = trainer.evaluate_model(val_labels, val_predictions)
    
    print("\nTraining completed!")
