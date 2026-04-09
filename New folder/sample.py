import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pywt
import mne
import sys
import glob

# Import the model from model.py
try:
    from model import EEG_2D_Hybrid_Model, get_training_components
except ImportError:
    print("Error importing model. Ensure this script is run from the same directory as model.py")
    sys.exit(1)

# ---------------------------------------------------------
# 1. Loading and Preprocessing dataset
# ---------------------------------------------------------
class EEG_Dataset(Dataset):
    def __init__(self, edf_path, summary_path, window_size=2, max_windows=None):
        self.window_size = window_size
        self.sfreq = 256
        self.window_samples = self.window_size * self.sfreq
        
        self.seizure_start, self.seizure_end = self.get_seizure_times(summary_path, os.path.basename(edf_path))
        
        print(f"Loading {edf_path}...")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.notch_filter(freqs=60.0, verbose=False)
        raw.filter(l_freq=0.5, h_freq=50.0, verbose=False)
        
        self.data = raw.get_data() 
        self.data = self.data[:22, :] 
        
        total_samples = self.data.shape[1]
        
        self.total_steps = (total_samples) // self.window_samples
        self.stride = 1
        if max_windows is not None and self.total_steps > max_windows:
             self.stride = max(1, self.total_steps // max_windows)
             self.total_steps = max_windows
             
        self.frequencies = np.arange(1, 51)
        self.wavelet = 'cmor1.5-1.0'
        self.scales = pywt.scale2frequency(self.wavelet, self.frequencies) / (1/self.sfreq)
        
        print(f"Dataset initialized with {self.total_steps} lazy-loaded windows.")

    def get_seizure_times(self, summary_file, edf_filename):
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        
        start_time = 0
        end_time = 0
        
        for i, line in enumerate(lines):
            if edf_filename in line:
                for j in range(i, i+10):
                    if "Seizure Start Time" in lines[j]:
                        start_time = int(lines[j].split(": ")[1].split(" ")[0])
                    if "Seizure End Time" in lines[j]:
                        end_time = int(lines[j].split(": ")[1].split(" ")[0])
                break
        return start_time, end_time

    def __len__(self):
        return self.total_steps
        
    def __getitem__(self, idx):
        start = idx * self.stride * self.window_samples
        end = start + self.window_samples
        
        segment = self.data[:, start:end]
        
        # Calculate CWT on the fly (Lazy Loading)
        cwt_features = np.zeros((22, 50, self.window_samples), dtype=np.float32)
        for ch in range(22):
            coeffs, _ = pywt.cwt(segment[ch], self.scales, self.wavelet, 1/self.sfreq)
            cwt_features[ch] = np.abs(coeffs) ** 2
            
        time_start = start / self.sfreq
        time_end = end / self.sfreq
        
        # 120 seconds = 2 minutes
        margin = 120.0
        
        if time_end < (self.seizure_start - margin):
            # Normal
            label = 0
        elif time_start >= (self.seizure_start - margin) and time_end <= self.seizure_start:
            # Preictal
            label = 1
        elif time_start < self.seizure_end and time_end > self.seizure_start:
            # Ictal (Seizure)
            label = 2
        elif time_start >= self.seizure_end and time_start <= (self.seizure_end + margin):
            # Postictal
            label = 3
        else:
            # Anything after the postictal margin is back to Normal
            label = 0
            
        return torch.tensor(cwt_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Use glob to cleanly find the files inside the dataset folder
    edf_files = glob.glob(os.path.join(project_root, "dataset", "**", "chb01_03.edf"), recursive=True)
    summary_files = glob.glob(os.path.join(project_root, "dataset", "**", "chb01-summary.txt"), recursive=True)
    
    if not edf_files:
        print(f"EDF file not found in {os.path.join(project_root, 'dataset')}")
        return
    if not summary_files:
        print(f"Summary file not found in {os.path.join(project_root, 'dataset')}")
        return
        
    edf_path = edf_files[0]
    summary_path = summary_files[0]
    print(f"Found EDF: {edf_path}")
    print(f"Found Summary: {summary_path}")
    
    # 1. Create Dataset
    dataset = EEG_Dataset(edf_path, summary_path, max_windows=900)
    
    # 2. Split into Train and Test sets (80% / 20%)
    from torch.utils.data import random_split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Split data into {train_size} train windows and {test_size} test windows.")

    # Note: num_workers=0 on Windows is typically safer to avoid progress bar glitching
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    # 3. Initialize Model
    model = EEG_2D_Hybrid_Model(num_channels=22, num_classes=4)
    
    # 4. Get Training Components
    criterion, optimizer, scheduler, scaler = get_training_components(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    from tqdm import tqdm
    # 5. Training Loop
    epochs = 2
    print("\nStarting Training Pipeline...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Training Phase with tqdm Progress Bar (computes CWT on the fly)
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for data, targets in train_pbar:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.1f}%"})
            
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {100.*correct/total:.2f}%")
        
        # Validation / Testing Phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=False)
        with torch.no_grad():
            for data, targets in test_pbar:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                test_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*test_correct/test_total:.1f}%"})
                
        print(f"Epoch {epoch+1}/{epochs} | Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {100.*test_correct/test_total:.2f}%")
        print("-" * 60)
        
    # Save the final model weights
    torch.save(model.state_dict(), "backend_model_completed.pt")
    print("\nBackend complete and model saved to backend_model_completed.pt")

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt

# Inside your loop or after creating a segment:
def visualize_step(cwt_features, segment_tensor):
    print(f"--- Tensor Trace ---")
    print(f"Segment Tensor Shape (Single Window): {segment_tensor.shape}")
    print(f"CWT Feature Map Shape: {cwt_features.shape}") # Should be [Channels, Freqs, Time]
    
    # Plotting CWT for the first channel
    plt.figure(figsize=(10, 4))
    plt.imshow(cwt_features[0], aspect='auto', cmap='jet', origin='lower')
    plt.title("CWT Scalogram (Channel 1)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (Samples)")
    plt.colorbar(label="Power")
    plt.show()

# To use this in your Dataset __init__:
# visualize_step(cwt_features, torch.tensor(segment))