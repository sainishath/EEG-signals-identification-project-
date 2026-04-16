import torch
import pandas as pd
import os
from tqdm import tqdm
from model import EEG_2D_Hybrid_Model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Targeting Device: {device}")

    # 1. Load Model
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, "backend_model_completed.pt")
    if not os.path.exists(model_path):
        print(f"[!] Error: {model_path} not found.")
        return

    model = EEG_2D_Hybrid_Model(num_channels=22, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Load Metadata
    METADATA_FILE = os.path.join(root_dir, "processed_metadata.csv")
    if not os.path.exists(METADATA_FILE):
        print(f"[!] Error: {METADATA_FILE} not found.")
        return

    print(f"[*] Reading metadata from: {os.path.abspath(METADATA_FILE)}")
    df = pd.read_csv(METADATA_FILE)
    print(df.tail())
    df['label'] = df['label'].astype(int)
    # Seizure class is 2
    seizure_samples = df[df['label'] == 2]

    if len(seizure_samples) == 0:
        print("[!] No seizure samples found in metadata.")
        return

    print(f"[*] Found {len(seizure_samples)} seizure samples. Searching for the most unidentifiable one...")

    hardest_sample = None
    min_confidence = 1.1 # Probabilities are 0-1
    full_results = []

    # 3. Search Loop
    with torch.no_grad():
        for _, row in tqdm(seizure_samples.iterrows(), total=len(seizure_samples)):
            file_path = row['file']
            
            # Load and prepare data (ensure it's local/found)
            if not os.path.exists(file_path):
                # Try prepending project path if relative
                file_path = os.path.join(".", file_path)
                if not os.path.exists(file_path):
                    continue

            # Data shape is [22, 50, 512], model needs [Batch=1, 22, 50, 512]
            data = torch.load(file_path).to(device).to(torch.float32).unsqueeze(0)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Confidence for "Seizure" (Index 2)
            seizure_prob = probabilities[2].item()
            
            if seizure_prob < min_confidence:
                min_confidence = seizure_prob
                hardest_sample = {
                    'file': row['file'],
                    'probs': probabilities.tolist(),
                    'pred': torch.argmax(probabilities).item()
                }

    if hardest_sample:
        classes = ["Normal", "Preictal", "Seizure", "Postictal"]
        print("\n" + "="*50)
        print("      MOST UNIDENTIFIABLE SEIZURE FOUND")
        print("="*50)
        print(f"File: {hardest_sample['file']}")
        print(f"True Label: Seizure")
        print(f"Predicted Class: {classes[hardest_sample['pred']]}")
        print("-" * 50)
        for i, prob in enumerate(hardest_sample['probs']):
            print(f"{classes[i]}: {prob * 100:.2f}%")
        print("="*50)
        print(f"Minimum Confidence for Seizure class: {min_confidence * 100:.2f}%")
        print("="*50)
    else:
        print("[!] Search failed to identify an edge case.")

if __name__ == "__main__":
    main()
