def evaluate_model():
    # Load test results
    test_results = torch.load('test_results.pth')
    
    accuracy = test_results['accuracy']
    predictions = test_results['predictions']
    labels = test_results['labels']
    class_names = test_results['class_names']
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
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
    plt.savefig('confusion_matrix.png')
    plt.show()
    
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
    plt.savefig('per_class_accuracy.png')
    plt.show()

# Run evaluation
evaluate_model()