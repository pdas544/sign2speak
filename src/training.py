def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load metadata
    metadata_path = 'processed/metadata.csv'
    metadata = pd.read_csv(metadata_path)
    
    # Create gloss to index mapping
    glosses = sorted(metadata['label'].unique())
    gloss_to_idx = {gloss: idx for idx, gloss in enumerate(glosses)}
    num_classes = len(glosses)
    
    print(f"Found {num_classes} classes: {glosses}")
    
    # Create datasets
    train_dataset = SignLanguageDataset(metadata_path, 'processed', gloss_to_idx, split='train')
    val_dataset = SignLanguageDataset(metadata_path, 'processed', gloss_to_idx, split='val')
    test_dataset = SignLanguageDataset(metadata_path, 'processed', gloss_to_idx, split='test')
    
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
    num_epochs = 50
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
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
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
            torch.save(model.state_dict(), 'best_model.pth')
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
    plt.savefig('training_history.png')
    plt.show()
    
    # Test phase
    print("Testing the best model...")
    model.load_state_dict(torch.load('best_model.pth'))
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
    
    torch.save(test_results, 'test_results.pth')
    
    return model, test_accuracy

if __name__ == "__main__":
    train_model()