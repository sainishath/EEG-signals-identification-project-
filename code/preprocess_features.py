import os
import glob
import torch
import numpy as np
import pywt
import mne
from tqdm import tqdm
import pandas as pd

# --- Configuration ---
SFREQ = 256
WINDOW_SIZE = 2
WINDOW_SAMPLES = WINDOW_SIZE * SFREQ
CHANNELS = 22
FREQS = np.arange(1, 51)
WAVELET = 'cmor1.5-1.0'
SCALES = pywt.scale2frequency(WAVELET, FREQS) / (1/SFREQ)

# Paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(root_dir, "dataset")
OUTPUT_DIR = os.path.join(root_dir, "processed_data")
METADATA_FILE = os.path.join(root_dir, "processed_metadata.csv")

# Sampling settings
MAX_FILES = 50 # Start with 50 diverse files to keep disk space manageable (~50-100GB)
WINDOWS_PER_RECORDING = 400 # Max windows per file

def get_seizure_times(summary_file, edf_filename):
    if not os.path.exists(summary_file):
        return -1, -1
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    
    start_time = -1
    end_time = -1
    
    base_filename = edf_filename.replace(".edf", "").split(" ")[0].split("(")[0].strip("_")
    
    for i, line in enumerate(lines):
        if base_filename in line:
            # Look ahead for seizure info
            for j in range(i, i+15):
                if j >= len(lines): break
                if "Seizure Start Time" in lines[j]:
                    try:
                        pts = lines[j].split(":")
                        val = pts[1].strip().split(" ")[0]
                        start_time = int(val)
                    except: pass
                if "Seizure End Time" in lines[j]:
                    try:
                        pts = lines[j].split(":")
                        val = pts[1].strip().split(" ")[0]
                        end_time = int(val)
                    except: pass
            break
    return start_time, end_time

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[*] Created {OUTPUT_DIR}")
    
    edf_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.edf"), recursive=True)
    if not edf_files:
        print(f"[!] No EDF files found in {DATASET_DIR}")
        return
        
    print(f"[*] Total EDF files discovered: {len(edf_files)}")
    print(f"[*] Starting preprocessing for up to {MAX_FILES} files...")
    
    metadata = []
    # Load existing metadata if resuming
    if os.path.exists(METADATA_FILE):
        try:
            metadata = pd.read_csv(METADATA_FILE).to_dict('records')
            print(f"[*] Resuming from {len(metadata)} existing samples.")
        except:
            pass

    processed_patient_files = 0
    
    for i, edf_path in enumerate(edf_files):
        if processed_patient_files >= MAX_FILES:
            break
            
        filename = os.path.basename(edf_path)
        dir_name = os.path.dirname(edf_path)
        patient_id = os.path.basename(dir_name)
        summary_path = os.path.join(dir_name, f"{patient_id}-summary.txt")
        
        # Check if we already have windows from this file in metadata
        if any(filename in m['file'] for m in metadata if 'file' in m):
            print(f"[-] Skipping {filename} (already processed)")
            continue

        s_start, s_end = get_seizure_times(summary_path, filename)
        
        try:
            print(f"\n--- Loading {filename} (Seizure: {s_start}s to {s_end}s) ---")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.notch_filter(60.0, verbose=False)
            raw.filter(0.5, 50.0, verbose=False)
            
            data = raw.get_data()
            if data.shape[0] < CHANNELS:
                print(f"[!] Skipping {filename}: only {data.shape[0]} channels.")
                continue
            data = data[:CHANNELS, :]
            
            total_samples = data.shape[1]
            total_steps = total_samples // WINDOW_SAMPLES
            
            # Smart Sampling Logic
            margin = 120.0
            indices_to_save = []
            
            # Step through the recording
            for step in range(total_steps):
                t_start = (step * WINDOW_SAMPLES) / SFREQ
                t_end = t_start + WINDOW_SIZE
                
                label = 0 # Normal
                if s_start != -1 and s_end != -1:
                    if t_start >= (s_start - margin) and t_end <= s_start:
                        label = 1 # Preictal
                    elif t_start < s_end and t_end > s_start:
                        label = 2 # Seizure
                    elif t_start >= s_end and t_start <= (s_end + margin):
                        label = 3 # Postictal
                
                # Priority: All Seizure/Preictal windows, sample Normal
                if label in [1, 2]:
                    indices_to_save.append((step, label))
                elif label == 3 and len([x for x in indices_to_save if x[1] == 3]) < 100:
                    indices_to_save.append((step, label))
                elif label == 0 and len([x for x in indices_to_save if x[1] == 0]) < 200:
                    indices_to_save.append((step, label))

            if not indices_to_save:
                continue

            print(f"[*] Extracting {len(indices_to_save)} windows...")
            for step, label in tqdm(indices_to_save, desc="Extracting CWT", leave=False):
                start = step * WINDOW_SAMPLES
                end = start + WINDOW_SAMPLES
                segment = data[:, start:end]
                
                # Compute CWT
                # Format: [22, 50, 512]
                cwt_features = np.zeros((CHANNELS, 50, WINDOW_SAMPLES), dtype=np.float16)
                for ch in range(CHANNELS):
                    coeffs, _ = pywt.cwt(segment[ch], SCALES, WAVELET, 1/SFREQ)
                    cwt_features[ch] = np.abs(coeffs).astype(np.float16)
                
                # Save
                save_name = f"{patient_id}_{filename}_s{step}_l{label}.pt"
                # Remove spaces/dots from filename to be safe
                save_name = save_name.replace(" ", "_").replace(".edf", "")
                save_path = os.path.join(OUTPUT_DIR, save_name)
                
                torch.save(torch.from_numpy(cwt_features), save_path)
                metadata.append({"file": save_path, "label": label})
            
            processed_patient_files += 1
            # Save metadata every file
            pd.DataFrame(metadata).to_csv(METADATA_FILE, index=False)
            
        except Exception as e:
            print(f"[!] Critical Error in {filename}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n=======================================================")
    print(f"PREPROCESSING COMPLETE!")
    print(f"Captured {len(metadata)} total windows into {OUTPUT_DIR}")
    print(f"Metadata saved to: {METADATA_FILE}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
