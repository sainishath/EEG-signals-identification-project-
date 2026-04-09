import torch
import os
from model import EEG_2D_Hybrid_Model

def main():
    # The .pt file is the saved brain/memory of your trained PyTorch model
    model_path = "backend_model_completed.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found! Make sure sample.py ran successfully.")
        return

    print("1. Loading the Hybrid Architecture...")
    # We must instantiate the exact same model class that was used for training
    model = EEG_2D_Hybrid_Model(num_channels=22, num_classes=3)
    
    print(f"2. Loading Trained Weights from '{model_path}' into the Model...")
    # Load the state dictionary (the weights) back into the architecture
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # VERY IMPORTANT: Set the model to Evaluation mode!
    # This disables training features like Dropout so it acts predictably.
    model.eval()
    
    print("3. Generating a Simulated EEG Snippet...")
    # To test it, we need data! In production, you would process 2 seconds of live EEG data here.
    # We will simulate a random [Batch=1, Channels=22, Frequencies=50, Time=512] tensor.
    sample_input = torch.rand(1, 22, 50, 512)
    
    print("4. Running Inference (Making a Prediction)...")
    # Use torch.no_grad() because we are predicting, not training (saves memory & time)
    with torch.no_grad(): 
        raw_outputs = model(sample_input)
        # Softmax turns raw numbers into percentages that add up to 1 (or 100%)
        probabilities = torch.softmax(raw_outputs, dim=1)[0]
    
    # Map the model's output classes to readable strings based on your project
    classes = ["Normal", "Seizure (Ictal)", "Other"]
    
    print("\n" + "="*30)
    print("         RESULTS")
    print("="*30)
    
    for i, prob in enumerate(probabilities):
        print(f"Confidence for {classes[i]}: {prob.item() * 100:.2f}%")
        
    predicted_class = torch.argmax(probabilities).item()
    print("-" * 30)
    print(f"FINAL VERDICT: {classes[predicted_class]}")
    print("="*30)

if __name__ == "__main__":
    main()
