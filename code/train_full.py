import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pywt
import mne
import sys
from tqdm import tqdm

from model import EEG_2D_Hybrid_Model, get_training_components

# Settings
BATCH_SIZE = 16
WINDOW_SIZE = 2
SFREQ = 256
MAX_WINDOWS_PER_FILE = 800  # Cap the number of 2-second windows per EDF to avoid infinite loops

class Stream_EEG_Dataset(Dataset):
    def __init__(self, edf_path, summary_path, window_size=2, max_windows=None):
        self.window_size = window_size
        self.sfreq = 256
        self.window_samples = self.window_size * self.sfreq
        
        self.seizure_start, self.seizure_end = self.get_seizure_times(summary_path, os.path.basename(edf_path))
        
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.notch_filter(freqs=60.0, verbose=False)
        raw.filter(l_freq=0.5, h_freq=50.0, verbose=False)
        
        self.data = raw.get_data() 
        self.data = self.data[:22, :] # Use first 22 channels
        
        total_samples = self.data.shape[1]
        
        self.total_steps = total_samples // self.window_samples
        self.stride = 1
        if max_windows is not None and self.total_steps > max_windows:
             self.stride = max(1, self.total_steps // max_windows)
             self.total_steps = max_windows
             
        self.frequencies = np.arange(1, 51)
        self.wavelet = 'cmor1.5-1.0'
        self.scales = pywt.scale2frequency(self.wavelet, self.frequencies) / (1/self.sfreq)

    def get_seizure_times(self, summary_file, edf_filename):
        if not os.path.exists(summary_file):
            return -1, -1
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        
        start_time = -1
        end_time = -1
        
        for i, line in enumerate(lines):
            if edf_filename in line:
                for j in range(i, i+10):
                    if j >= len(lines): break
                    if "Seizure Start Time" in lines[j]:
                        try:
                            start_time = int(lines[j].split(": ")[1].split(" ")[0])
                        except: pass
                    if "Seizure End Time" in lines[j]:
                        try:
                            end_time = int(lines[j].split(": ")[1].split(" ")[0])
                        except: pass
                break
        return start_time, end_time

    def __len__(self):
        return self.total_steps
        
    def __getitem__(self, idx):
        start = idx * self.stride * self.window_samples
        end = start + self.window_samples
        
        segment = self.data[:, start:end]
        
        # Calculate CWT
        cwt_features = np.zeros((22, 50, self.window_samples), dtype=np.float32)
        for ch in range(min(22, segment.shape[0])):
            coeffs, _ = pywt.cwt(segment[ch], self.scales, self.wavelet, 1/self.sfreq)
            cwt_features[ch] = np.abs(coeffs) ** 2
            
        time_start = start / self.sfreq
        time_end = end / self.sfreq
        
        # 4 Classes mapping
        margin = 120.0
        label = 0 # Normal default
        
        if self.seizure_start != -1 and self.seizure_end != -1:
            if time_end < (self.seizure_start - margin):
                label = 0
            elif time_start >= (self.seizure_start - margin) and time_end <= self.seizure_start:
                label = 1 # Preictal
            elif time_start < self.seizure_end and time_end > self.seizure_start:
                label = 2 # Ictal
            elif time_start >= self.seizure_end and time_start <= (self.seizure_end + margin):
                label = 3 # Postictal
                
        return torch.tensor(cwt_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def load_or_create_model(device, model_path="backend_model_completed.pt"):
    model = EEG_2D_Hybrid_Model(num_channels=22, num_classes=4)
    if os.path.exists(model_path):
        print(f"[+] Found existing weights at '{model_path}'. Updating them...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"[-] No existing weights found at '{model_path}'. Starting fresh training.")
    model = model.to(device)
    return model

def main():
    print("="*60)
    print("   NEUROSCAN - MASSIVE GPU TRAINING PIPELINE")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Targeting Device: {device}")
    if device.type == 'cpu':
        print("[!] WARNING: CUDA is not active! Training on CPU will be extremely slow.")
        
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(project_root, "dataset")
    
    # 1. Glob all EDF files
    edf_files = glob.glob(os.path.join(dataset_dir, "**", "*.edf"), recursive=True)
    if not edf_files:
        print(f"[!] No EDF files found in {dataset_dir}")
        return
        
    print(f"[*] Total EDF files discovered: {len(edf_files)}")
    
    # Load Model
    model_path = os.path.join(project_root, "backend_model_completed.pt")
    model = load_or_create_model(device, model_path)
    
    criterion, optimizer, scheduler, scaler = get_training_components(model)

    processed_count = 0
    MAX_FILES_TO_PROCESS = 30
    
    # Stream file by file
    for i, edf_path in enumerate(edf_files):
        if processed_count >= MAX_FILES_TO_PROCESS:
            print(f"\n[!] Reached manual cap of {MAX_FILES_TO_PROCESS} recordings. Halting pipeline safely.")
            break
            
        print(f"\n--- Processing File {i+1}/{len(edf_files)}: {os.path.basename(edf_path)} ---")
        
        # Construct summary path (same dir, usually patientid-summary.txt)
        dir_name = os.path.dirname(edf_path)
        patient_id = os.path.basename(dir_name) # e.g. chb01
        summary_path = os.path.join(dir_name, f"{patient_id}-summary.txt")
        
        try:
            dataset = Stream_EEG_Dataset(edf_path, summary_path, max_windows=MAX_WINDOWS_PER_FILE)
            if len(dataset) == 0:
                print("Skipping empty dataset.")
                continue
                
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device.type=='cuda'))
            
            model.train()
            file_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(dataloader, desc=f"Training", leave=False)
            for data, targets in pbar:
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
                    
                file_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            acc = 100. * correct / total if total > 0 else 0
            print(f"[+] Completed {os.path.basename(edf_path)} | Loss: {file_loss/len(dataloader):.4f} | Acc: {acc:.2f}%")
            
            # Checkpoint the model! Continual saving.
            torch.save(model.state_dict(), model_path)
            processed_count += 1
            
        except Exception as e:
            print(f"[!] Error processing {edf_path}: {str(e)}")
            continue

    print(f"\n=======================================================")
    print(f"TRAINING COMPLETE! Successfully processed {processed_count} EDF files.")
    print(f"Final model deployed to: {model_path}")

if __name__ == "__main__":
    main()
