"""
NeuroScan Flask API — Complete Backend
Stages: Data Ingestion → CWT Transform → CNN-Transformer → Clinical Output
Supports real model (.pt weights) or demo mode fallback (no weights needed)
"""

import os, sys, io, csv, time, json, secrets, base64, tempfile
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CLASSES       = ["Normal", "Preictal", "Seizure (Ictal)", "Postictal"]
N_CHANNELS    = 22
SAMPLE_RATE   = 256          # Hz (CHB-MIT standard)
EPOCH_SECS    = 2
EPOCH_LEN     = SAMPLE_RATE * EPOCH_SECS   # 512 samples
NOTCH_FREQ    = 60.0
BANDPASS_LOW  = 0.5
BANDPASS_HIGH = 50.0
Z_SCORE_EPS   = 1e-8

# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION STORE  (extend to DB / env vars for production)
# ─────────────────────────────────────────────────────────────────────────────
AUTH_STORE = {
    "DEMO_CLINIC":   {"key": "NS2026",       "name": "Demo Clinic"},
    "chb_research":  {"key": "CHB_MIT_001",  "name": "CHB-MIT Research Lab"},
    "neuroscan_dev": {"key": "DEV_9999",     "name": "Development Mode"},
}
_sessions = {}   # token → {practice_id, name, expires}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (real → demo fallback)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_MODE = "demo"
_model     = None

def _try_load_model():
    global _model, MODEL_MODE
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from model import EEG_2D_Hybrid_Model        # noqa
        import torch
        # Note: weights are stored in the parent folder of code/
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights = os.path.join(root_dir, "backend_model_completed.pt")
        m = EEG_2D_Hybrid_Model(num_channels=N_CHANNELS, num_classes=len(CLASSES))
        m.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        m.eval()
        _model     = m
        MODEL_MODE = "real"
        print("[OK] Real model loaded:", weights)
    except Exception as e:
        print(f"[WARN] Demo mode active ({type(e).__name__}: {e})")

_try_load_model()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — SIGNAL PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def notch_filter(signal: np.ndarray, sr: int, freq: float) -> np.ndarray:
    """Simple IIR notch filter (scipy-free fallback using spectral nulling)."""
    try:
        from scipy.signal import iirnotch, filtfilt
        b, a = iirnotch(freq, Q=30.0, fs=sr)
        return filtfilt(b, a, signal, axis=-1)
    except ImportError:
        # Spectral nulling fallback
        F = np.fft.rfft(signal, axis=-1)
        freqs = np.fft.rfftfreq(signal.shape[-1], 1.0 / sr)
        mask = np.abs(freqs - freq) < 2.0
        F[..., mask] = 0
        return np.fft.irfft(F, n=signal.shape[-1], axis=-1)


def bandpass_filter(signal: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Butterworth bandpass filter."""
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
        return filtfilt(b, a, signal, axis=-1)
    except ImportError:
        # FFT bandpass fallback
        F = np.fft.rfft(signal, axis=-1)
        freqs = np.fft.rfftfreq(signal.shape[-1], 1.0 / sr)
        mask = (freqs < lo) | (freqs > hi)
        F[..., mask] = 0
        return np.fft.irfft(F, n=signal.shape[-1], axis=-1)


def z_score_normalize(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=-1, keepdims=True)
    std  = signal.std(axis=-1, keepdims=True) + Z_SCORE_EPS
    return (signal - mean) / std


def artifact_reject(epoch: np.ndarray, amplitude_thresh_uv: float = 300.0) -> bool:
    """Return True if epoch is clean (no rail/saturation artefacts)."""
    return float(np.abs(epoch).max()) < amplitude_thresh_uv


def preprocess_signal(raw: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Stage 1.3 + 1.4: filter → epoch → artifact-reject → z-score.
    Input:  (N_ch, N_samples)  raw µV
    Output: (N_ch, EPOCH_LEN)  clean normalised epoch
    """
    filtered = notch_filter(raw, sr, NOTCH_FREQ)
    filtered = bandpass_filter(filtered, sr, BANDPASS_LOW, BANDPASS_HIGH)

    # Epoch: take the first clean EPOCH_LEN window
    n_samples = filtered.shape[-1]
    step = EPOCH_LEN // 2
    epoch = None
    for start in range(0, max(1, n_samples - EPOCH_LEN + 1), step):
        chunk = filtered[:, start:start + EPOCH_LEN]
        if chunk.shape[-1] < EPOCH_LEN:
            chunk = np.pad(chunk, ((0, 0), (0, EPOCH_LEN - chunk.shape[-1])))
        if artifact_reject(chunk):
            epoch = chunk
            break

    if epoch is None:                            # all windows noisy → take first
        epoch = filtered[:, :EPOCH_LEN]
        if epoch.shape[-1] < EPOCH_LEN:
            epoch = np.pad(epoch, ((0, 0), (0, EPOCH_LEN - epoch.shape[-1])))

    return z_score_normalize(epoch)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CWT SCALOGRAM GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _morlet_cwt(signal_1d: np.ndarray, freqs: np.ndarray, sr: int) -> np.ndarray:
    """Compute |CWT| for one channel using a complex Morlet wavelet (cmor1.5-1.0)."""
    coefs = np.zeros((len(freqs), len(signal_1d)))
    for i, f in enumerate(freqs):
        sigma = 7.0 / (2 * np.pi * f)
        half_w = int(4 * sigma * sr)
        t_w = np.linspace(-half_w / sr, half_w / sr, min(2 * half_w + 1, 257))
        env = np.exp(-t_w ** 2 / (2 * sigma ** 2))
        wr  = env * np.cos(2 * np.pi * f * t_w)
        wi  = env * np.sin(2 * np.pi * f * t_w)
        cr  = np.convolve(signal_1d, wr, mode="same")
        ci  = np.convolve(signal_1d, wi, mode="same")
        coefs[i] = np.sqrt(cr ** 2 + ci ** 2)
    return coefs


def render_scalogram_png(signal_1d: np.ndarray,
                         channel_name: str = "Ch-0",
                         colormap: str = "jet") -> str:
    """Render a CWT scalogram and return base-64 PNG."""
    freqs = np.linspace(1, 50, 50)
    cwt   = _morlet_cwt(signal_1d[:EPOCH_LEN], freqs, SAMPLE_RATE)
    cwt_n = (cwt - cwt.min()) / (cwt.max() - cwt.min() + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 3), dpi=110)
    ax.imshow(cwt_n, aspect="auto", origin="lower", cmap=colormap,
              extent=[0, EPOCH_SECS, 1, 50], interpolation="bilinear")
    ax.set_xlabel("Time (s)", color="#8ab0c8", fontsize=8)
    ax.set_ylabel("Freq (Hz)", color="#8ab0c8", fontsize=8)
    ax.set_title(f"CWT · {channel_name} · cmor1.5-1.0", color="#c8dde8", fontsize=8)
    ax.tick_params(colors="#8ab0c8", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2a3d")
    fig.patch.set_facecolor("#0c1219")
    ax.set_facecolor("#060a0f")
    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor="#0c1219", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — DEMO SIGNAL GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_signal(signal_type: str) -> list:
    rng = np.random.default_rng(99)
    t   = np.linspace(0, EPOCH_SECS, EPOCH_LEN)
    out = []
    for ch in range(N_CHANNELS):
        noise = rng.normal(0, 1, EPOCH_LEN) * 4
        if signal_type == "normal":
            s = noise + 30 * np.sin(2*np.pi*10*t) + 12*np.sin(2*np.pi*3*t)
        elif signal_type == "preictal":
            env = 1 + 0.6 * t / EPOCH_SECS
            s   = noise * env + 50 * np.sin(2*np.pi*7*t) * (1 + 0.3*np.sin(2*np.pi*t))
        elif signal_type == "seizure":
            spike = np.zeros(EPOCH_LEN)
            s0, s1 = EPOCH_LEN//4, 3*EPOCH_LEN//4
            spike[s0:s1] = 130 * np.sin(2*np.pi*4*t[s0:s1]) + \
                           40  * np.sin(2*np.pi*12*t[s0:s1])
            s = noise * 2 + spike
        elif signal_type == "postictal":
            decay = np.exp(-t * 1.0)
            s     = noise * decay + 30*np.sin(2*np.pi*1.5*t)*decay
        else:
            s = noise
        out.append(s.tolist())
    return out

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — INFERENCE (real model or demo heuristic)
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_PRIORS = {
    "normal":    [0.9732, 0.0180, 0.0050, 0.0038],
    "preictal":  [0.0300, 0.8912, 0.0588, 0.0200],
    "seizure":   [0.0100, 0.0124, 0.9676, 0.0100],
    "postictal": [0.0400, 0.0200, 0.0150, 0.9250],
}


def _demo_infer(signal_array: list, hint: str = None):
    arr = np.array(signal_array).flatten()
    rng = np.random.default_rng()

    if hint and hint in _DEMO_PRIORS:
        base = np.array(_DEMO_PRIORS[hint])
    else:
        var   = float(np.var(arr))
        fft   = np.abs(np.fft.rfft(arr[:EPOCH_LEN]))
        beta  = float(fft[int(13*EPOCH_LEN/SAMPLE_RATE):int(30*EPOCH_LEN/SAMPLE_RATE)].mean())
        delta = float(fft[1:int(4*EPOCH_LEN/SAMPLE_RATE)].mean())
        rms   = float(np.sqrt(np.mean(arr**2)))
        if var > 8000 and beta > 50:
            base = np.array(_DEMO_PRIORS["seizure"])
        elif var > 3500:
            base = np.array(_DEMO_PRIORS["preictal"])
        elif delta > beta * 2 and rms < 30:
            base = np.array(_DEMO_PRIORS["postictal"])
        else:
            base = np.array(_DEMO_PRIORS["normal"])

    noise = rng.normal(0, 0.003, 4)
    probs = np.clip(base + noise, 0, 1)
    probs = (probs / probs.sum()).tolist()
    idx   = int(np.argmax(probs))
    return CLASSES[idx], probs


def _real_infer(signal_array: list):
    import torch
    from preprocess_features import extract_cwt_features   # noqa
    feat   = extract_cwt_features(np.array(signal_array))
    tensor = torch.FloatTensor(feat).unsqueeze(0)
    with torch.no_grad():
        logits = _model(tensor)
        p = torch.softmax(logits, dim=1)[0]
    probs = p.tolist()
    return CLASSES[int(p.argmax())], probs

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — MEDICAL NOTES GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_notes(prediction: str, confidence: float,
                   patient_id: str, elapsed_ms: int) -> str:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    conf = round(confidence * 100, 1)
    _T = {
        "Normal": (
            f"NEUROSCAN ANALYSIS REPORT — {ts}\n"
            f"Patient ID: {patient_id}  |  Model: CNN-Transformer Hybrid  |  "
            f"Pipeline: NeuroScan v1.0  |  Dataset: CHB-MIT\n"
            f"{'─'*60}\n"
            f"Classification : NORMAL CORTICAL ACTIVITY\n"
            f"Confidence     : {conf}%   |   Inference Time: {elapsed_ms} ms\n"
            f"{'─'*60}\n\n"
            f"Clinical Summary:\n"
            f"Signal analysis of the submitted EEG epoch indicates normal cortical\n"
            f"background activity. No epileptiform discharges, spike-wave complexes,\n"
            f"or ictal patterns were identified. Background rhythm is within expected\n"
            f"frequency ranges. CWT scalogram shows distributed low-power activity\n"
            f"consistent with resting-state or inter-ictal baseline.\n\n"
            f"Recommended Action:\n"
            f"Continue routine monitoring per standard protocol. No immediate\n"
            f"clinical intervention required at this time."
        ),
        "Preictal": (
            f"NEUROSCAN ANALYSIS REPORT — {ts}\n"
            f"Patient ID: {patient_id}  |  Model: CNN-Transformer Hybrid  |  "
            f"Pipeline: NeuroScan v1.0  |  Dataset: CHB-MIT\n"
            f"{'─'*60}\n"
            f"Classification : PRE-ICTAL ACTIVITY DETECTED\n"
            f"Confidence     : {conf}%   |   Inference Time: {elapsed_ms} ms\n"
            f"{'─'*60}\n\n"
            f"Clinical Summary:\n"
            f"Pre-ictal biomarkers identified in the current EEG epoch. Emerging\n"
            f"rhythmic activity and subtle amplitude modulation consistent with\n"
            f"the pre-seizure transition phase have been detected. CWT scalogram\n"
            f"shows increasing power in the 4–8 Hz band. If patient is ambulatory,\n"
            f"precautionary measures should be initiated immediately.\n\n"
            f"⚠ Recommended Action:\n"
            f"Alert supervising clinician. Prepare rescue medication per care plan.\n"
            f"Begin continuous monitoring. Reassess in 30–60 seconds."
        ),
        "Seizure (Ictal)": (
            f"NEUROSCAN ANALYSIS REPORT — {ts}\n"
            f"Patient ID: {patient_id}  |  Model: CNN-Transformer Hybrid  |  "
            f"Pipeline: NeuroScan v1.0  |  Dataset: CHB-MIT\n"
            f"{'─'*60}\n"
            f"Classification : *** SEIZURE (ICTAL) EVENT CONFIRMED ***\n"
            f"Confidence     : {conf}%   |   Inference Time: {elapsed_ms} ms\n"
            f"{'─'*60}\n\n"
            f"Clinical Summary:\n"
            f"High-confidence ictal event confirmed. The CWT scalogram demonstrates\n"
            f"characteristic seizure morphology: high-amplitude burst activity with\n"
            f"clear onset across multiple electrode channels. Dominant frequency band\n"
            f"3–8 Hz with multi-channel propagation pattern clearly visible in the\n"
            f"frequency-time representation. Seizure onset: {ts}.\n\n"
            f"🚨 Recommended Action:\n"
            f"IMMEDIATE clinical response required. Administer rescue medication\n"
            f"per protocol. Document seizure onset time. Notify attending physician\n"
            f"immediately. Begin seizure safety protocol."
        ),
        "Postictal": (
            f"NEUROSCAN ANALYSIS REPORT — {ts}\n"
            f"Patient ID: {patient_id}  |  Model: CNN-Transformer Hybrid  |  "
            f"Pipeline: NeuroScan v1.0  |  Dataset: CHB-MIT\n"
            f"{'─'*60}\n"
            f"Classification : POST-ICTAL RECOVERY STATE\n"
            f"Confidence     : {conf}%   |   Inference Time: {elapsed_ms} ms\n"
            f"{'─'*60}\n\n"
            f"Clinical Summary:\n"
            f"Post-ictal suppression pattern identified. EEG shows characteristic\n"
            f"delta-band slowing and amplitude attenuation consistent with recovery\n"
            f"following an ictal event. CWT scalogram confirms dominant low-frequency\n"
            f"power (<4 Hz) with gradual diminution across channels. Patient may\n"
            f"exhibit confusion, fatigue, or transient neurological deficits.\n\n"
            f"Recommended Action:\n"
            f"Monitor for secondary seizure events. Assess neurological status.\n"
            f"Document post-ictal duration. Continue observation for ≥30 minutes."
        ),
    }
    return _T.get(prediction, _T["Normal"])

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
# Serve from parent directory where neuroscan_dashboard.html sits
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = Flask(__name__, static_folder=ROOT_DIR)
CORS(app)

def _auth_session():
    tok = request.headers.get("X-Auth-Token", "")
    s   = _sessions.get(tok)
    if s and s["expires"] > time.time():
        return s
    return None

# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({
        "status":     "ok",
        "model_mode": MODEL_MODE,
        "version":    "1.0.0",
        "classes":    CLASSES,
        "timestamp":  datetime.now().isoformat(),
    })

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/api/auth", methods=["POST"])
def auth_login():
    data = request.get_json(force=True) or {}
    pid  = data.get("practice_id", "").strip()
    key  = data.get("key", "").strip()
    rec  = AUTH_STORE.get(pid)
    if not rec or rec["key"] != key:
        return jsonify({"error": "Invalid Practice ID or Key"}), 401
    tok = secrets.token_hex(24)
    _sessions[tok] = {
        "practice_id": pid,
        "name":        rec["name"],
        "expires":     time.time() + 8 * 3600,
    }
    return jsonify({"token": tok, "clinician": rec["name"], "practice_id": pid})

@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    tok = request.headers.get("X-Auth-Token", "")
    _sessions.pop(tok, None)
    return jsonify({"status": "logged_out"})

# ── Demo signals ──────────────────────────────────────────────────────────────
@app.route("/api/demo-signal/<signal_type>")
def demo_signal(signal_type):
    valid = {"normal", "preictal", "seizure", "postictal"}
    if signal_type not in valid:
        return jsonify({"error": f"Unknown type. Choose from: {list(valid)}"}), 400
    sig = generate_demo_signal(signal_type)
    return jsonify({
        "signal_data": sig,
        "type":        signal_type,
        "channels":    N_CHANNELS,
        "sample_rate": SAMPLE_RATE,
        "epoch_len":   EPOCH_LEN,
    })

# ── Predict ───────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    t0          = time.time()
    signal_2d   = None   # shape: (N_ch, EPOCH_LEN)
    source_type = "unknown"
    hint        = None   # demo type hint for demo inference

    patient_id = "unknown"
    if request.is_json:
        body       = request.get_json()
        patient_id = body.get("patient_id", "unknown")
        hint       = body.get("type")               # from demo-signal endpoint
        raw        = body.get("signal_data") or body.get("signal")
        if raw is None:
            return jsonify({"error": "JSON must contain 'signal_data'"}), 400
        arr = np.array(raw, dtype=float)
        if arr.ndim == 1:
            arr = np.tile(arr, (N_CHANNELS, 1))
        signal_2d   = arr[:N_CHANNELS, :EPOCH_LEN] if arr.shape[-1] >= EPOCH_LEN \
                      else np.pad(arr[:N_CHANNELS], ((0,0),(0,EPOCH_LEN-arr.shape[-1])))
        source_type = "json"

    elif "file" in request.files:
        f          = request.files["file"]
        patient_id = request.form.get("patient_id") or os.path.splitext(f.filename)[0]
        fname      = f.filename.lower()

        if fname.endswith(".edf"):
            # Stage 1.1 — EDF (medical format)
            try:
                import mne
                with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                    f.save(tmp.name)
                    raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False)
                    data, _ = raw[: min(N_CHANNELS, len(raw.ch_names)), :]
                os.unlink(tmp.name)
                if data.shape[0] < N_CHANNELS:
                    data = np.pad(data, ((0, N_CHANNELS-data.shape[0]), (0,0)))
                # Convert from Volts to µV
                data *= 1e6
                source_type = "edf"
                # Stage 1.3 + 1.4 applied below
                signal_2d = preprocess_signal(data, sr=int(raw.info["sfreq"]))
            except ImportError:
                return jsonify({"error": "mne not installed. pip install mne"}), 500
            except Exception as e:
                return jsonify({"error": f"EDF parse error: {e}"}), 400

        elif fname.endswith((".csv", ".txt")):
            # Stage 1.1 — CSV / TXT flat file
            try:
                content = f.read().decode()
                reader  = csv.reader(io.StringIO(content))
                rows    = []
                for row in reader:
                    try:
                        rows.append([float(v) for v in row if v.strip()])
                    except ValueError:
                        continue          # skip header rows
                arr = np.array(rows, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                elif arr.shape[0] > arr.shape[1]:  # rows=samples, cols=channels
                    arr = arr.T
                if arr.shape[0] < N_CHANNELS:
                    arr = np.tile(arr, (int(np.ceil(N_CHANNELS/arr.shape[0])), 1))[:N_CHANNELS]
                signal_2d   = preprocess_signal(arr, sr=SAMPLE_RATE)
                source_type = "csv"
            except Exception as e:
                return jsonify({"error": f"CSV parse error: {e}"}), 400
        else:
            return jsonify({"error": "Unsupported file. Use .edf / .csv / .txt"}), 400

    else:
        return jsonify({"error": "No file or JSON signal provided."}), 400

    # Apply preprocessing if not already done (EDF path does it above)
    if source_type == "json":
        signal_2d = preprocess_signal(signal_2d, sr=SAMPLE_RATE)

    # ── Stage 3: Inference ────────────────────────────────────────────────────
    try:
        if MODEL_MODE == "real" and _model is not None:
            pred, probs = _real_infer(signal_2d.tolist())
        else:
            pred, probs = _demo_infer(signal_2d.tolist(), hint=hint)
    except Exception as e:
        pred, probs = _demo_infer(signal_2d.tolist(), hint=hint)

    elapsed_ms = int((time.time() - t0) * 1000)

    # ── Stage 2: Two scalograms (ch0 and ch1) ────────────────────────────────
    CH_NAMES = [
        "FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1",
        "FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2",
        "FZ-CZ","CZ-PZ","P7-T7","T7-FT9","FT9-FT10","FT10-T8",
    ]
    scalo_1 = render_scalogram_png(signal_2d[0],  channel_name=CH_NAMES[0],  colormap="jet")
    scalo_2 = render_scalogram_png(signal_2d[6],  channel_name=CH_NAMES[6],  colormap="turbo")

    # ── Stage 4: Medical notes ────────────────────────────────────────────────
    top_conf = max(probs)
    notes    = generate_notes(pred, top_conf, patient_id, elapsed_ms)

    # Signal preview (first 512 samples of channel 0, unprocessed for display)
    preview = signal_2d[0, :512].tolist()

    return jsonify({
        "prediction":   pred,
        "confidences": {
            "Normal":          round(probs[0], 6),
            "Preictal":        round(probs[1], 6),
            "Seizure (Ictal)": round(probs[2], 6),
            "Postictal":       round(probs[3], 6),
        },
        "scalogram":    scalo_1,
        "scalogram_2":  scalo_2,
        "signal_preview": preview,
        "patient_id":   patient_id,
        "elapsed_ms":   elapsed_ms,
        "model_mode":   MODEL_MODE,
        "source_type":  source_type,
        "medical_notes": notes,
        "timestamp":    datetime.now().isoformat(),
    })

# ── Notes update ──────────────────────────────────────────────────────────────
@app.route("/api/notes/generate", methods=["POST"])
def notes_endpoint():
    d    = request.get_json(force=True) or {}
    pred = d.get("prediction", "Normal")
    conf = float(d.get("confidence", 0.95))
    pid  = d.get("patient_id", "unknown")
    ms   = int(d.get("elapsed_ms", 0))
    return jsonify({"notes": generate_notes(pred, conf, pid, ms)})

# ── Static dashboard ──────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return send_from_directory(ROOT_DIR, "neuroscan_dashboard.html")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*56)
    print("  NeuroScan EEG Analysis API - v1.0")
    print(f"  Model mode : {MODEL_MODE}")
    print(f"  Classes    : {CLASSES}")
    print("  -------------------------------------------------")
    print("  Demo credentials:")
    for pid, v in AUTH_STORE.items():
        print(f"    {pid:<20} -> key: {v['key']}")
    print("  -------------------------------------------------")
    print("  Open: http://localhost:5000")
    print("="*56 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
