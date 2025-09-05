def predict_sign(model_path, keypoints_path, glosses):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model (adjust input_size based on your keypoint dimension)
    input_size = 543 * 3  # 543 landmarks * 3 coordinates
    model = SignLanguageModel(input_size, len(glosses)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load keypoints
    keypoints = torch.load(keypoints_path)
    sequence_length = torch.tensor([keypoints.shape[0]])
    
    # Add batch dimension
    keypoints = keypoints.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(keypoints, sequence_length)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return glosses[predicted.item()], confidence.item()

# Example usage
glosses = ['teacher', 'happy', 'nice', 'good', 'no', 'go', 'what', 'like', 'hello',
           'white', 'friend', 'big', 'beautiful', 'boy', 'sister']

sign, confidence = predict_sign('best_model.pth', 'processed/test/teacher_abc123.pt', glosses)
print(f"Predicted sign: {sign} with confidence: {confidence:.2f}")