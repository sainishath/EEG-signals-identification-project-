import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import mne
import torch
import glob
import numpy as np

# 1. Find all 5 EDF files in your directory
# Replace './data/' with the actual path where you downloaded the files
file_paths = sorted(glob.glob('./data/chb*.edf')) 

all_eeg_tensors = []

for file_path in file_paths:
    print(f"Loading {file_path}...")
    
    # 2. Read the EDF file
    # preload=True loads the data into memory, which is faster for extraction
    raw_data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
    # Optional: Filter the data (e.g., standard 1Hz to 50Hz bandpass for EEG)
    # raw_data.filter(l_freq=1.0, h_freq=50.0)
    
    # 3. Extract the numerical data as a NumPy array
    # The shape will be (Channels, Time_Steps)
    signal_array = raw_data.get_data() 
    
    # 4. Convert the NumPy array to a PyTorch Tensor
    # We use float32 as it is standard for PyTorch neural networks
    tensor_data = torch.tensor(signal_array, dtype=torch.float32)
    
    # Add a "Batch" dimension at the front so it becomes (1, Channels, Time_Steps)
    # This is the exact shape a Conv1D or Transformer layer expects as input
    tensor_data = tensor_data.unsqueeze(0) 
    
    all_eeg_tensors.append(tensor_data)

# 5. (Optional) If all 5 files have the exact same duration/time steps, 
# you can stack them into one massive batch. 
# If they have different lengths, keep them in the list or slice them into equal chunks.
try:
    final_batch = torch.cat(all_eeg_tensors, dim=0)
    print(f"\nSuccess! Final Tensor Shape: {final_batch.shape}")
    print("Format: (Batch_Size, Channels, Sequence_Length)")
except (RuntimeError, ValueError) as e:
    print("\nFiles have different lengths or none found, keeping them in a list or empty.")
    print(f"Number of loaded recordings: {len(all_eeg_tensors)}")

# --- 1. The 2D Hybrid Model (Optimized) ---
class EEG_2D_Hybrid_Model(nn.Module):
    def __init__(self, num_channels=22, num_classes=3):
        super(EEG_2D_Hybrid_Model, self).__init__()
        
        # 1. 2D CNN Extractor (Early Fusion of all channels)
        # Expected Input: [Batch, 22 Channels, 50 Frequency_Bins, 512 TimeSteps]
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # <-- OPTIMIZATION: Stabilizes training
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # Output shape from here: [Batch, 32, 25, 256]
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # <-- OPTIMIZATION
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  
            # Output shape from here: [Batch, 64, 12, 128]
        )

        # 2. Transformer Setup
        # We now have 128 time steps. Each step has 64 feature maps * 12 frequency bins.
        self.d_model = 64 * 12  # = 768 features per time step
        
        # Dim_feedforward is often 4x d_model, but reducing it saves VRAM
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, dim_feedforward=2048, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Drops 50% of connections to prevent over-fitting
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x input shape: [Batch, Channels(22), Freqs(50), Time(512)]
        
        # Step 1: CNN Feature Extraction
        # Output is: [Batch, 64, 12, 128]
        x = self.cnn(x) 
        
        # Step 2: The Bridge (Crucial Reshaping)
        # We must treat the final dimension (128) as our TIME sequence.
        # Permute to: [Batch, Time(128), Channels(64), Freqs(12)]
        x = x.permute(0, 3, 1, 2) 
        
        # Now flatten the spatial dimensions (64*12) into a single feature vector per time step.
        b, seq_len, c, h = x.shape
        x = x.reshape(b, seq_len, c * h) 
        # Final shape for Transformer: [Batch, 128, 768]
        
        # Step 3: Transformer Memory/Attention
        # Output is still: [Batch, 128, 768]
        x = self.transformer(x) 
        
        # Step 4: Time-Sequence Fusion (Global Average Pool over time steps)
        # We crush the 128 time steps into a single mean representation.
        # Output shape: [Batch, 768]
        x = torch.mean(x, dim=1) 
        
        # Step 5: Softmax Classification
        out = self.classifier(x) 
        return out

# --- 2. The Dataset (Optimized for Pre-computed Data) ---
class EEG_Precomputed_Dataset(Dataset):
    def __init__(self, file_paths, labels):
        """
        Args:
            file_paths (list): List of paths to .pt (PyTorch tensor) files.
            labels (list): List of corresponding labels (0, 1, 2).
        """
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. Load the pre-computed CWT scalogram tensor from disk.
        # We expect a file containing a tensor of shape: (22, 50, 512)
        scalogram = torch.load(self.file_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return scalogram, label

# --- 3. Example of Training Setup (Re-included for accuracy) ---
def get_training_components(model):
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    
    # Class weights for imbalanced data (Normal, Preictal, Ictal)
    # This prevents the model from just guessing "Normal" 99% of the time.
    weights = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float32)
    
    if torch.cuda.is_available():
        weights = weights.cuda()
        
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # AdamW handles weight decay better than Adam for Transformers
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    # Reduces LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # AMP Scaler for mixed-precision training (speeds up GPU usage)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    return criterion, optimizer, scheduler, scaler

# --- 4. Test the model with Dummy Data ---
if __name__ == "__main__":
    print("Testing the Fully Optimized 2D Hybrid Architecture...")
    
    # Create dummy tensor matching the CWT output: [Batch, 22, 50, 512]
    dummy_input = torch.randn(4, 22, 50, 512) 
    
    # Initialize the model
    model = EEG_2D_Hybrid_Model(num_channels=22, num_classes=3)
    
    # Pass data through
    predictions = model(dummy_input)
    probabilities = torch.softmax(predictions, dim=1)
    
    print(f"Input Scalogram Shape: {dummy_input.shape}")
    print(f"Output Probabilities Shape: {probabilities.shape} <-- [Batch Size, 3 Classes]")
    print(f"Sample Probabilities:\n{probabilities.detach().numpy()}")
    print("\n2D Model structure is verified and syntactically correct!")