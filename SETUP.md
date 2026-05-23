# NeuroScan — Local Setup & Run Guide

## Project Structure
```text
neuroscan/
├── code/
│   ├── app.py                      ← Flask backend (all 4 pipeline stages)
│   ├── model.py                    ← CNN-Transformer model definition
│   ├── preprocess_features.py      ← CWT feature extractor (Stage 2)
│   ├── preprocess_seizure_only.py  ← Targeted seizure data preprocessing
│   ├── find_hardest_seizure.py     ← Seizure edge case locator
│   ├── test_manual_seizure.py      ← Model accuracy verification script
│   ├── resource_monitor.py         ← Hardware safety guard
│   └── training_config.yaml        ← Safe resource threshold limits
├── neuroscan_dashboard.html        ← Frontend (Stage 4.1 clinical output)
├── backend_model_completed.pt      ← Trained PyTorch model weights (automatically loaded)
├── processed_metadata.csv          ← Dataset preprocessing indices mapping
├── requirements.txt                ← Python package dependencies list
└── SETUP.md                        ← This run guide
```

## 1. Install Dependencies
Ensure you have Python installed, then install all requirements from the root directory:
```bash
pip install -r requirements.txt
```

## 2. Run the API
To launch the Flask backend server:
```bash
python code/app.py
```

Upon starting, the server will output connection details:
```text
[OK] Real model loaded: d:\desktop\project file\backend_model_completed.pt

========================================================
  NeuroScan EEG Analysis API - v1.0
  Model mode : real           ← 'real' since weights are present
  Classes    : ['Normal', 'Preictal', 'Seizure (Ictal)', 'Postictal']
  -------------------------------------------------
  Demo credentials:
    DEMO_CLINIC          -> key: NS2026
    chb_research         -> key: CHB_MIT_001
    neuroscan_dev        -> key: DEV_9999
  -------------------------------------------------
  Open: http://localhost:5000
========================================================
```

## 3. Open the Dashboard
Open your browser and navigate to:
```text
http://localhost:5000
```
Login with the credentials listed above, for example:
* **Practice ID:** `DEMO_CLINIC`
* **Access Key:** `NS2026`

## 4. Test the System
1. **Interactive Demo:** Click any of the four demo signal buttons (**Normal** / **Preictal** / **Seizure** / **Postictal**) on the sidebar control panel. The system will mock the raw EEG feed, pipeline it to `/api/predict`, and update all dashboard widgets.
2. **Standard File Analysis:** Upload or drag-and-drop a `.edf`, `.csv`, or `.txt` file into the upload zone to run full inference against the loaded neural network.

## 5. System Features
* **Authentication (Stage 1.2):** Validates clinician credentials and issues an 8-hour session token.
* **Signal Preprocessing (Stage 1.3 & 1.4):** Implements dual IIR notch filters (60Hz) and Butterworth bandpass filters (0.5–50Hz), splits data into 512-sample epochs, rejects artifacts (>300µV), and applies Z-score normalization.
* **Continuous Wavelet Transform (Stage 2):** Generates dual-channel CWT frequency-time scalograms (channels 0 and 6) using the Complex Morlet wavelet and returns base64 PNGs.
* **Model Inference (Stage 3):** Automatically uses the real hybrid CNN-Transformer classifier (`backend_model_completed.pt`) if present, otherwise falling back to signal heuristics.
* **Clinical Panel (Stage 4.1):** Presents patient metrics, visual CWT images, classification outcomes, animated confidence bars, and editable AI-generated medical reports.
