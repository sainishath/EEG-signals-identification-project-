import mne
import numpy as np
import matplotlib.pyplot as plt
import pywt  # This is PyWavelets

# 1. Load the data
file_path = "chb01_03.edf"
raw = mne.io.read_raw_edf(file_path, preload=True)

# 2. Clean the Signal (Preprocessing)
print("Filtering data...")
raw_cleaned = raw.copy()
# Remove the 60Hz power line hum
raw_cleaned.notch_filter(freqs=60.0, verbose=False) 
# Keep only relevant brainwave frequencies (0.5 to 50 Hz)
raw_cleaned.filter(l_freq=0.5, h_freq=50.0, verbose=False) 

# 3. Extract a 2-second snippet from ONE channel to test
# Let's grab the first channel ('FP1-F7'), from second 2990 to 2992
# (This is 6 seconds before the seizure starts!)
sfreq = int(raw.info['sfreq']) # Sampling frequency is 256 Hz
start_sample = 2990 * sfreq
end_sample = 2992 * sfreq

# Extract the data
data, times = raw_cleaned[0, start_sample:end_sample]
signal_1d = data[0] # Here is your clean 1D array!

# 4. Apply CWT (Converting 1D to 2D)
print("Applying CWT...")
# We use a Complex Morlet wavelet, which is the gold standard for EEG
frequencies = np.arange(1, 51) # We want to look at 1 Hz to 50 Hz
wavelet = 'cmor1.5-1.0' 
scales = pywt.scale2frequency(wavelet, frequencies) / (1/sfreq)

coefficients, freqs = pywt.cwt(signal_1d, scales, wavelet, 1/sfreq)

# Calculate power (magnitude squared) to create the image
power = np.abs(coefficients) ** 2

# 5. Plot the Results
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot the Cleaned 1D Signal
axes[0].plot(times, signal_1d, color='blue')
axes[0].set_title("Step 1: Cleaned EEG Signal (1D) - Channel FP1-F7")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlabel("Time (seconds)")

# Plot the 2D Scalogram
# This 'power' matrix is exactly what you will feed into your PyTorch 2D CNN!
im = axes[1].imshow(power, extent=[times[0], times[-1], freqs[-1], freqs[0]], 
                    aspect='auto', cmap='jet')
axes[1].invert_yaxis() # Put low frequencies at the bottom
axes[1].set_title("Step 2: CWT Scalogram (2D Image for CNN)")
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_xlabel("Time (seconds)")
fig.colorbar(im, ax=axes[1], label="Power (Intensity)")

plt.tight_layout()
plt.show()

print(f"Shape of the final 2D matrix: {power.shape}")
import torch
import torch.nn as nn

# 1. Convert the numpy array into a PyTorch Tensor
# We add two extra dimensions for [Batch, Channel, Height, Width]
tensor_image = torch.tensor(power, dtype=torch.float32)
tensor_image = tensor_image.unsqueeze(0).unsqueeze(0) 

print(f"\n--- PyTorch Transition ---")
print(f"Input Tensor Shape: {tensor_image.shape}  <-- [Batch, Channel, Freq(H), Time(W)]")

# 2. Define a very simple 2D CNN Extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Conv2d: Looks for patterns. 
        # in_channels=1 (our CWT image), out_channels=16 (it will find 16 different patterns)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # ReLU: The activation function (keeps only positive signals)
        self.relu = nn.ReLU()
        # MaxPool2d: Shrinks the image to make it easier to process
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# 3. Create the model and pass our data through it
cnn_model = FeatureExtractor()
output_features = cnn_model(tensor_image)

print(f"Output Tensor Shape: {output_features.shape} <-- [Batch, Feature_Maps, New_H, New_W]")
# 1. The Bridge: Prepare CNN output for the Transformer
# Current Shape: [Batch=1, Channels=16, Height=25, Width=256]
# We want: [Batch, Sequence_Length, Embedding_Dimension]

# We "Permute" to move the Width (Time) to the middle
# New Shape: [1, 256, 16, 25]
x = output_features.permute(0, 3, 1, 2) 

# We "Flatten" the channels and height into one long vector for each time step
# New Shape: [1, 256, 400]  <-- (16 * 25 = 400)
# This means we have a sequence of 256 "steps", each with 400 pieces of info.
batch_size, seq_len, _, _ = x.shape
x_flat = x.reshape(batch_size, seq_len, -1)

print(f"\n--- Transformer Transition ---")
print(f"Input to Transformer: {x_flat.shape} <-- [Batch, Sequence, Info_Per_Step]")

# 2. Define the Transformer Encoder
# d_model is the size of our info vector (400)
# nhead is how many "Attention Heads" we use (must divide 400 evenly)
d_model = x_flat.shape[2]
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

# 3. Pass the data through the Transformer
transformer_out = transformer_encoder(x_flat)

print(f"Transformer Output: {transformer_out.shape}")

# 4. The Fusion & Softmax Prep
# We take the average of the whole sequence to get one "summary" of the seizure activity
fusion_layer = torch.mean(transformer_out, dim=1) 
print(f"Final Features for Fusion: {fusion_layer.shape}")

# Final Classification Head (Seizure vs. Normal vs. Preictal)
classifier = nn.Linear(d_model, 3) 
logits = classifier(fusion_layer)
probabilities = torch.softmax(logits, dim=1)

print(f"\n--- Final Result ---")
print(f"Probabilities (Before, During, After): {probabilities.detach().numpy()}")
def get_seizure_times(summary_file, edf_filename):
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    
    start_time = None
    end_time = None
    
    for i, line in enumerate(lines):
        if edf_filename in line:
            # Look at the next few lines for seizure info
            for j in range(i, i+10):
                if "Seizure Start Time" in lines[j]:
                    start_time = int(lines[j].split(": ")[1].split(" ")[0])
                if "Seizure End Time" in lines[j]:
                    end_time = int(lines[j].split(": ")[1].split(" ")[0])
            break
            
    return start_time, end_time

# Test it
start, end = get_seizure_times("chb01-summary.txt", "chb01_03.edf")
print(f"File: chb01_03.edf")
print(f"Seizure starts at: {start} seconds")
print(f"Seizure ends at: {end} seconds")