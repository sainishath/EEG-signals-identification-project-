import os
import torch
import numpy as np
import pywt
import mne
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
SFREQ = 256
WINDOW_SIZE = 2
WINDOW_SAMPLES = WINDOW_SIZE * SFREQ
CHANNELS = 22
FREQS = np.arange(1, 51)
WAVELET = 'cmor1.5-1.0'
SCALES = pywt.scale2frequency(WAVELET, FREQS) / (1/SFREQ)

def main():
    # Target file known to have a seizure
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    edf_path = os.path.join(root_dir, r"dataset\CHBMIT_BACKUP-20260408T091306Z-3-005\CHBMIT_BACKUP\chb02\chb02_16.edf")
    summary_path = os.path.join(root_dir, r"dataset\CHBMIT_BACKUP-20260408T091306Z-3-001\CHBMIT_BACKUP\chb02\chb02-summary.txt")
    output_dir = os.path.join(root_dir, "processed_data")
    metadata_file = os.path.join(root_dir, "processed_metadata.csv")
    
    s_start, s_end = 130, 212 # from summary for chb02_16
    
    print(f"[*] Processing Seizure Data from {edf_path}...")
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.notch_filter(60.0, verbose=False)
    raw.filter(0.5, 50.0, verbose=False)
    
    data = raw.get_data()[:CHANNELS, :]
    total_steps = data.shape[1] // WINDOW_SAMPLES
    
    metadata = []
    if os.path.exists(metadata_file):
        metadata = pd.read_csv(metadata_file).to_dict('records')

    # Extract ALL seizure windows
    for step in range(total_steps):
        t_start = (step * WINDOW_SAMPLES) / SFREQ
        t_end = t_start + WINDOW_SIZE
        
        if t_start >= s_start and t_end <= s_end:
            label = 2
            start = step * WINDOW_SAMPLES
            end = start + WINDOW_SAMPLES
            segment = data[:, start:end]
            
            # Compute CWT
            cwt_features = np.zeros((CHANNELS, 50, WINDOW_SAMPLES), dtype=np.float16)
            for ch in range(CHANNELS):
                coeffs, _ = pywt.cwt(segment[ch], SCALES, WAVELET, 1/SFREQ)
                cwt_features[ch] = np.abs(coeffs).astype(np.float16)
            
            save_name = f"chb02_16_s{step}_l2_target.pt"
            save_path = os.path.join(output_dir, save_name)
            torch.save(torch.from_numpy(cwt_features), save_path)
            metadata.append({"file": save_path, "label": label})
            print(f"Saved seizure window {step} to {save_path}")

    df_final = pd.DataFrame(metadata)
    df_final.to_csv(metadata_file, index=False)
    print(df_final.tail())
    print(f"[*] Done. Added {len([m for m in metadata if m['label']==2])} seizure samples to metadata.")

if __name__ == "__main__":
    main()
