import os
import torch
import numpy as np
import pywt
import mne
from model import EEG_2D_Hybrid_Model

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Targeting Device: {device}")

    # 2. Load Model
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, "backend_model_completed.pt")
    model = EEG_2D_Hybrid_Model(num_channels=22, num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 3. Load known seizure EDF
    # Found in CHBMIT_BACKUP-20260408T091306Z-3-005
    edf_path = r"d:\desktop\project file\dataset\CHBMIT_BACKUP-20260408T091306Z-3-005\CHBMIT_BACKUP\chb02\chb02_16.edf"
    s_start = 130  # from summary
    s_end = 212
    
    # Selection: middle of the seizure
    test_time = (s_start + s_end) // 2
    
    print(f"[*] Processing known seizure from {edf_path} at {test_time}s...")
    
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.notch_filter(60.0, verbose=False)
    raw.filter(0.5, 50.0, verbose=False)
    
    # 4. Extract 2s window
    SFREQ = 256
    WINDOW_SIZE = 2
    CHANNELS = 22
    FREQS = np.arange(1, 51)
    WAVELET = 'cmor1.5-1.0'
    SCALES = pywt.scale2frequency(WAVELET, FREQS) / (1/SFREQ)
    
    start_sample = int(test_time * SFREQ)
    end_sample = start_sample + int(WINDOW_SIZE * SFREQ)
    
    data = raw.get_data()[:CHANNELS, start_sample:end_sample]
    
    # Compute CWT
    cwt_features = np.zeros((CHANNELS, 50, int(WINDOW_SIZE * SFREQ)), dtype=np.float32)
    for ch in range(CHANNELS):
        coeffs, _ = pywt.cwt(data[ch], SCALES, WAVELET, 1/SFREQ)
        cwt_features[ch] = np.abs(coeffs).astype(np.float32)
        
    input_tensor = torch.from_numpy(cwt_features).to(device).unsqueeze(0)
    
    # 5. Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
    classes = ["Normal", "Preictal", "Seizure", "Postictal"]
    print("\n" + "="*50)
    print("      MANUAL SEIZURE TEST RESULTS")
    print("="*50)
    print(f"Sample: {os.path.basename(edf_path)} at {test_time}s")
    print(f"Ground Truth: Seizure")
    print("-" * 50)
    for i, prob in enumerate(probabilities):
        print(f"{classes[i]}: {prob * 100:.2f}%")
        
    pred_idx = torch.argmax(probabilities).item()
    print("-" * 50)
    print(f"PREDICTED: {classes[pred_idx]}")
    print("="*50)

if __name__ == "__main__":
    main()
