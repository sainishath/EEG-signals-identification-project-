"""
NeuroScan — Flask API Server
Integrates EEG_2D_Hybrid_Model with the frontend.
Run: python app.py
"""

import os
import sys
import io
import tempfile
import base64
import traceback

import numpy as np
import torch
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import pywt
import mne
import matplotlib
matplotlib.use('Agg')        # non-interactive backend (no GUI)
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from model import EEG_2D_Hybrid_Model

# ── Constants ─────────────────────────────────────────────────────────────────
SFREQ       = 256          # EEG sampling rate (Hz)
TARGET_CH   = 22           # model's expected channel count
TARGET_T    = 512          # time samples per window
FREQ_MIN    = 1
FREQ_MAX    = 50
N_FREQS     = FREQ_MAX - FREQ_MIN + 1   # 50
WAVELET     = 'cmor1.5-1.0'
FREQUENCIES = np.arange(FREQ_MIN, FREQ_MAX + 1, dtype=float)
SCALES      = pywt.scale2frequency(WAVELET, FREQUENCIES) / (1.0 / SFREQ)
CLASSES     = ['Normal', 'Preictal', 'Seizure', 'Postictal']
MODEL_PATH  = os.path.join(BASE_DIR, 'backend_model_completed.pt')

# Dark-theme palette for matplotlib
THEME = dict(
    bg     = '#0b1419',
    bg2    = '#0f1c24',
    fg     = '#e2edf2',
    muted  = '#6b8a96',
    accent = '#00c8b4',
    border = '#1d3040',
)

# ── App + CORS ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Lazy model loading ────────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model weights not found at '{MODEL_PATH}'. "
                "Run sample.py first to train and save the model."
            )
        print(f"[NeuroScan] Loading model from {MODEL_PATH}…")
        _model = EEG_2D_Hybrid_Model(num_channels=TARGET_CH, num_classes=len(CLASSES))
        state = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        _model.load_state_dict(state)
        _model.eval()
        print("[NeuroScan] Model ready.")
    return _model

# ── Signal helpers ────────────────────────────────────────────────────────────

def pad_or_trim(arr, length):
    """Ensure 1-D array has exactly `length` samples."""
    if len(arr) >= length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)))


def compute_cwt_power(signal_1d):
    """Return CWT power matrix  (N_FREQS × TARGET_T)  for one channel."""
    sig = pad_or_trim(signal_1d, TARGET_T).astype(np.float64)
    coeffs, _ = pywt.cwt(sig, SCALES, WAVELET, 1.0 / SFREQ)
    return np.abs(coeffs) ** 2          # (50, 512)


def build_model_input(data_nd, n_ch):
    """
    data_nd : (n_ch, samples) float32 array
    Returns  : torch tensor  (1, 22, 50, 512)
    """
    buf = np.zeros((TARGET_CH, N_FREQS, TARGET_T), dtype=np.float32)
    for ch in range(n_ch):
        buf[ch] = compute_cwt_power(data_nd[ch])
    # channels n_ch..21 stay zero-padded → model handles them gracefully
    return torch.tensor(buf, dtype=torch.float32).unsqueeze(0)   # (1,22,50,512)

# ── Image rendering ───────────────────────────────────────────────────────────

def generate_cwt_image(signal_1d, power):
    """
    Render a themed EEG + CWT figure and return it as a base64 PNG string.
    signal_1d : (TARGET_T,) 1-D array (channel 1)
    power     : (N_FREQS, TARGET_T) CWT power matrix
    """
    t = np.linspace(0, TARGET_T / SFREQ, TARGET_T)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 5.5),
                                    facecolor=THEME['bg2'],
                                    gridspec_kw={'hspace': 0.42})

    # ── Raw EEG signal ──
    ax0.set_facecolor(THEME['bg'])
    ax0.plot(t, signal_1d, color=THEME['accent'], linewidth=0.9, alpha=0.92)
    ax0.set_title('Input EEG Signal  (Channel 1)', color=THEME['fg'],
                  fontsize=10, pad=6, fontweight='bold')
    ax0.set_ylabel('Amplitude (µV)', color=THEME['muted'], fontsize=8)
    ax0.set_xlabel('Time (s)', color=THEME['muted'], fontsize=8)
    ax0.tick_params(colors=THEME['muted'], labelsize=7)
    ax0.grid(True, color=THEME['border'], linewidth=0.4, alpha=0.5)
    for sp in ax0.spines.values():
        sp.set_edgecolor(THEME['border'])

    # ── CWT Scalogram ──
    ax1.set_facecolor(THEME['bg'])
    extent = [t[0], t[-1], FREQ_MIN, FREQ_MAX]
    im = ax1.imshow(power, aspect='auto', cmap='inferno',
                    origin='lower', extent=extent)
    ax1.set_title('CWT Scalogram  (cmor1.5-1.0 wavelet, 1–50 Hz)',
                  color=THEME['fg'], fontsize=10, pad=6, fontweight='bold')
    ax1.set_ylabel('Frequency (Hz)', color=THEME['muted'], fontsize=8)
    ax1.set_xlabel('Time (s)', color=THEME['muted'], fontsize=8)
    ax1.tick_params(colors=THEME['muted'], labelsize=7)
    for sp in ax1.spines.values():
        sp.set_edgecolor(THEME['border'])
    cbar = fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(colors=THEME['muted'], labelsize=7)
    cbar.set_label('Power', color=THEME['muted'], fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor=THEME['bg2'], edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ── Inference + response builder ─────────────────────────────────────────────

def run_inference(tensor_in):
    """Return (probs np.array, predicted_idx int)."""
    mdl = get_model()
    with torch.no_grad():
        logits = mdl(tensor_in)
        probs  = torch.softmax(logits, dim=1)[0].numpy()
    return probs, int(np.argmax(probs))


def build_response(signal_1d, probs, pred_idx, n_channels_used):
    label      = CLASSES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    # Sensitivity ≈ how strongly model signals "Seizure"
    sensitivity = float(probs[2]) * 100
    # Loss proxy = NLL of the predicted class (lower = more confident)
    loss_proxy  = float(-np.log(max(float(probs[pred_idx]), 1e-8)))

    sig  = signal_1d[:TARGET_T].astype(np.float32)
    pwr  = compute_cwt_power(sig)
    cwt_b64 = generate_cwt_image(sig, pwr)

    return {
        'predicted_label': label,
        'confidence':      round(confidence, 2),
        'probabilities': {
            'Normal':    round(float(probs[0]) * 100, 2),
            'Preictal':  round(float(probs[1]) * 100, 2),
            'Seizure':   round(float(probs[2]) * 100, 2),
            'Postictal': round(float(probs[3]) * 100, 2),
        },
        'cwt_image_b64': cwt_b64,
        'display_signal': sig.tolist(),
        'num_channels': n_channels_used,
        'session_metrics': {
            'sensitivity': round(sensitivity, 2),
            'loss':        round(loss_proxy, 4),
        },
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Two accepted payloads:
      • multipart/form-data  with field 'edf_file'      → real EDF analysis
      • application/json     with {signal: float[]}     → demo signal analysis
    """
    try:
        # ── EDF upload ────────────────────────────────────────────────────────
        if 'edf_file' in request.files:
            file = request.files['edf_file']
            if not file.filename.lower().endswith('.edf'):
                return jsonify({'error': 'Only .edf files are accepted.'}), 400

            suffix  = '.edf'
            tmp     = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_path = tmp.name
            tmp.close()
            file.save(tmp_path)

            try:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                raw.notch_filter(freqs=60.0, verbose=False)
                raw.filter(l_freq=0.5, h_freq=50.0, verbose=False)

                total_ch = len(raw.ch_names)
                n_ch     = min(total_ch, TARGET_CH)   # at most 22

                if n_ch < 16:
                    return jsonify({
                        'error': (f'EDF has only {total_ch} channel(s). '
                                  'At least 16 channels are required.')
                    }), 400

                data      = raw.get_data()[:n_ch]          # (n_ch, samples)
                signal_1d = data[0, :TARGET_T].astype(np.float32)
                tensor_in = build_model_input(data, n_ch)
                probs, idx = run_inference(tensor_in)

                return jsonify(build_response(signal_1d, probs, idx, n_ch))

            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # ── Demo JSON signal ──────────────────────────────────────────────────
        elif request.is_json:
            body   = request.get_json(force=True)
            signal = np.array(body.get('signal', []), dtype=np.float32)

            if len(signal) < 64:
                return jsonify({'error': 'Signal too short (need ≥ 64 samples).'}), 400

            sig = pad_or_trim(signal, TARGET_T)

            # Replicate single channel to 22 channels for the model
            data_22ch = np.tile(sig, (TARGET_CH, 1))    # (22, 512)
            tensor_in = build_model_input(data_22ch, TARGET_CH)
            probs, idx = run_inference(tensor_in)

            return jsonify(build_response(sig, probs, idx, 1))

        else:
            return jsonify({'error': 'Send an .edf file or a JSON signal array.'}), 400

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 503
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Internal server error — check console.'}), 500


@app.route('/api/demo-signal/<signal_type>', methods=['GET'])
def demo_signal(signal_type):
    """Return a synthetic EEG signal array (512 samples)."""
    N   = TARGET_T
    t   = np.linspace(0, N / SFREQ, N)
    rng = np.random.default_rng(42)   # fixed seed for reproducibility

    if signal_type == 'seizure':
        sig = (  np.sin(2*np.pi*3*t) * 800
               + np.sin(2*np.pi*7*t) * 300
               + rng.standard_normal(N) * 100)
    elif signal_type == 'preictal':
        r   = np.linspace(1.0, 2.0, N)
        sig = (  np.sin(2*np.pi*4*t) * 60 * r
               + np.sin(2*np.pi*8*t) * 30 * r
               + rng.standard_normal(N) * 20)
    elif signal_type == 'postictal':
        # Characterized by slow waves and suppression directly after seizure
        sig = (  np.sin(2*np.pi*1.5*t) * 150
               + np.sin(2*np.pi*3*t) * 40
               + rng.standard_normal(N) * 30)
    else:   # normal
        sig = (  np.sin(2*np.pi*10*t) * 20
               + np.sin(2*np.pi*20*t) * 10
               + rng.standard_normal(N) * 15)

    return jsonify({'signal': sig.tolist()})


@app.route('/api/health', methods=['GET'])
def health():
    model_ok = os.path.exists(MODEL_PATH)
    return jsonify({
        'status': 'ok',
        'model_loaded': _model is not None,
        'model_file_exists': model_ok,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  NeuroScan API — http://localhost:5000")
    print(f"  Model path : {MODEL_PATH}")
    print("=" * 55)
    app.run(debug=True, port=5000, host='0.0.0.0')
