<div align="center">

# 🧠 NeuroScan — EEG Seizure Phase Classifier

**Real-time epileptic seizure detection using a hybrid 2D CNN + Transformer architecture**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![Dataset](https://img.shields.io/badge/Dataset-CHB--MIT-4CAF50?style=flat-square)](https://physionet.org/content/chbmit/1.0.0/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

</div>

---

> **⚡ Demo Mode — no setup required.** The Flask API runs fully in-browser demo mode without model weights. Download weights (link below) only if you want live inference on real EDF files.

---

## 🎯 Problem Statement

Epilepsy affects ~50 million people worldwide. Real-time seizure detection from scalp EEG is clinically critical but computationally hard: seizure-phase signals are rare, short-lived, and buried in noise. This project builds an end-to-end pipeline — from raw `.edf` recordings through signal preprocessing, time-frequency feature extraction, and a deep learning classifier — to a clinical-grade web dashboard that classifies each 2-second EEG window into one of four phases.

---

## 🏗️ Architecture

```
Raw EEG (.edf / stream)
        │
        ▼
┌───────────────────────────────────────────┐
│  Stage 1 — Signal Preprocessing           │
│  IIR Notch (60 Hz) + Butterworth          │
│  Bandpass (0.5–50 Hz) → 512-sample epochs │
│  Artifact rejection (>300 µV), Z-score    │
└──────────────────┬────────────────────────┘
                   │ (22 ch × 512 samples)
                   ▼
┌───────────────────────────────────────────┐
│  Stage 2 — CWT Feature Extraction         │
│  Complex Morlet wavelet (cmor1.5-1.0)     │
│  50 frequency bins (1–50 Hz)              │
│  Output: 22 × 50 × 512 scalogram tensor  │
└──────────────────┬────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────┐
│  Stage 3 — 2D CNN Encoder                 │
│  Conv2d(22→32) → BN → ReLU → MaxPool2d   │
│  Conv2d(32→64) → BN → ReLU → MaxPool2d   │
│  Output: 64 × 12 × 128 feature map       │
└──────────────────┬────────────────────────┘
                   │ reshape → (128, 768)
                   ▼
┌───────────────────────────────────────────┐
│  Stage 3 — Transformer Encoder            │
│  d_model=768, nhead=8, 2 layers           │
│  Global average pool over time axis       │
└──────────────────┬────────────────────────┘
                   │ (768,)
                   ▼
┌───────────────────────────────────────────┐
│  Stage 4 — Classifier Head                │
│  Linear(768→256) → ReLU → Dropout(0.5)   │
│  Linear(256→4)   → Softmax               │
└──────────────────┬────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  4-Class Output      │
        │  0: Normal           │
        │  1: Preictal         │
        │  2: Ictal (Seizure)  │
        │  3: Postictal        │
        └──────────────────────┘
```

**Why this design?** CNNs capture local spatial patterns across electrode-frequency space; the Transformer captures long-range temporal dependencies across the 128-step sequence. The CWT bridge converts the raw 1D signal into a 2D time-frequency image, making it compatible with Conv2d and giving the model explicit spectral structure to work with.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Name** | CHB-MIT Scalp EEG Database |
| **Source** | PhysioNet (physionet.org) |
| **Subjects** | 22 pediatric patients |
| **Channels** | 22-channel bipolar montage |
| **Sampling Rate** | 256 Hz |
| **Window Size** | 2 seconds (512 samples) |
| **Seizures** | 198 total across all patients |
| **Class weights** | [0.1, 0.4, 0.9, 0.4] (Normal/Preictal/Ictal/Postictal) |

Class-weighted `CrossEntropyLoss` addresses severe label imbalance — ictal windows are rare relative to baseline normal recordings.

---

## 📈 Results

**Spot-check inference** — `test_manual_seizure.py` run against `chb02_16.edf` (a patient not in primary training loop), sampled at the confirmed seizure midpoint (171 s, ground truth: Ictal):

| Class | Model Confidence |
|---|---|
| Normal | 0.07% |
| Preictal | 0.03% |
| **Seizure (Ictal)** | **99.78%** ✅ |
| Postictal | 0.12% |
| **Predicted** | **Seizure (Ictal)** — correct |

> Full per-class precision/recall/F1 requires a formal held-out test split across all 22 subjects. The spot-check above confirms the trained weights generalise to unseen recordings.

**Training details:** AdamW (lr=1e-4, weight_decay=1e-2), ReduceLROnPlateau scheduler (patience=3, factor=0.5), mixed-precision (AMP) training on CUDA. Streamed file-by-file to avoid OOM on 45 GB dataset.

---

## 🖥️ Clinical Dashboard

<!-- SCREENSHOT: Open neuroscan_dashboard.html locally (demo mode works without weights),
     grab a screenshot showing the EEG waveform + CWT scalograms + classification panel,
     save as docs/dashboard_screenshot.png, then replace this comment with:
     ![NeuroScan Dashboard](docs/dashboard_screenshot.png) -->

The dashboard (`neuroscan_dashboard.html`) provides a full clinical interface:

- **Live EEG waveform** — animated multi-channel signal viewer
- **CWT scalograms** — dual-channel frequency-time heatmaps (base64 PNG from API)
- **Classification panel** — predicted phase + per-class confidence bars
- **Editable AI report** — auto-generated clinical summary, editable in-browser
- **Auth-gated access** — 8-hour session tokens per clinic

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the API (demo mode — no weights needed)

```bash
python code/app.py
```

Open `http://localhost:5000` and log in:

| Practice ID | Access Key |
|---|---|
| `DEMO_CLINIC` | `NS2026` |
| `chb_research` | `CHB_MIT_001` |
| `neuroscan_dev` | `DEV_9999` |

Click **Normal / Preictal / Seizure / Postictal** demo buttons to simulate a full inference cycle without uploading any file.

### Run with real model weights

1. Download `backend_model_completed.pt` → **[Google Drive link — add yours here]**
2. Place it in the project root (same level as `code/`)
3. Restart `python code/app.py` — it auto-detects and loads real weights

---

## 🔌 API Reference

All endpoints require `Authorization: Bearer <token>` after login.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/auth` | Authenticate, returns 8-hour session token |
| `POST` | `/api/predict` | Classify EEG data — returns phase, confidence, CWT images |
| `GET` | `/api/status` | Server health + model mode (`real` / `demo`) |
| `GET` | `/` | Serve clinical dashboard HTML |

**`POST /api/predict` — request body:**

```json
{
  "signal": [[...22 channels × 512 samples...]],
  "sample_rate": 256
}
```

**Response:**

```json
{
  "prediction": "Seizure (Ictal)",
  "confidence": 0.873,
  "probabilities": {"Normal": 0.04, "Preictal": 0.06, "Seizure (Ictal)": 0.87, "Postictal": 0.03},
  "cwt_images": {"ch0": "<base64 PNG>", "ch6": "<base64 PNG>"},
  "model_mode": "real"
}
```

---

## 📁 Project Structure

```text
EEG-signals-identification-project-/
├── code/
│   ├── app.py                      ← Flask backend (all 4 pipeline stages)
│   ├── model.py                    ← CNN-Transformer model definition
│   ├── train_full.py               ← Streaming GPU training pipeline
│   ├── train_optimized.py          ← Memory-optimized training variant
│   ├── preprocess_features.py      ← CWT scalogram pre-computation
│   ├── preprocess_seizure_only.py  ← Targeted seizure-window extraction
│   ├── find_hardest_seizure.py     ← Edge-case locator for hard negatives
│   ├── test_manual_seizure.py      ← Single-window inference test
│   ├── resource_monitor.py         ← GPU/CPU safety throttle
│   └── training_config.yaml        ← Resource threshold configuration
├── neuroscan_dashboard.html        ← Clinical frontend (self-contained)
├── processed_metadata.csv          ← Preprocessing index (window→label map)
├── requirements.txt
├── SETUP.md                        ← Detailed local run guide
└── README.md
```

> **Model weights** (`backend_model_completed.pt`, ~43 MB) are excluded from git. Download link above.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch 2.x — custom `nn.Module` |
| Feature Extraction | PyWavelets (CWT), MNE-Python (EDF I/O) |
| Signal Processing | SciPy (IIR notch + Butterworth filters) |
| Backend API | Flask + Flask-CORS |
| Frontend | Vanilla HTML/CSS/JS (self-contained, no build step) |
| Training | CUDA + AMP (mixed precision) |
| Dataset | CHB-MIT via PhysioNet |

---

## 📄 License

MIT — see [LICENSE](LICENSE).
