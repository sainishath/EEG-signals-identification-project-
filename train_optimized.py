import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from model import EEG_2D_Hybrid_Model, get_training_components, EEG_Precomputed_Dataset
from tqdm import tqdm
import yaml
import time
import datetime
import logging
from resource_monitor import should_pause, log_resource_usage

# Settings (will be overridden by config)
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4

def main():
    # Load training configuration
    with open('training_config.yaml', 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    max_gpu_mem = cfg.get('max_gpu_mem_percent', 70)
    max_cpu = cfg.get('max_cpu_percent', 50)
    checkpoint_interval = cfg.get('checkpoint_interval_seconds', 300)
    num_workers = cfg.get('num_workers', 2)
    BATCH_SIZE = cfg.get('batch_size', 32)
    EPOCHS = cfg.get('epochs', 30)
    # Setup logger
    logging.basicConfig(filename=cfg.get('log_file', 'training_monitor.log'), level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Targeting Device: {device}")
    logging.info(f"Training started on device {device}")
    
    METADATA_FILE = "processed_metadata.csv"
    if not os.path.exists(METADATA_FILE):
        print(f"[!] Error: {METADATA_FILE} not found. Please run preprocess_features.py first.")
        return

    # 1. Load Metadata
    print(f"[*] Loading metadata from {METADATA_FILE}...")
    df = pd.read_csv(METADATA_FILE)
    file_paths = df['file'].tolist()
    labels = df['label'].tolist()
    
    if len(file_paths) == 0:
        print("[!] Error: No samples found in metadata file.")
        return
        
    print(f"[*] Total Samples: {len(file_paths)}")

    # 2. Create Dataset and Split (80/20)
    full_dataset = EEG_Precomputed_Dataset(file_paths, labels)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)

    # 3. Initialize Model
    model = EEG_2D_Hybrid_Model(num_channels=22, num_classes=4).to(device)
    
    # Load existing if available to continue training
    model_path = "backend_model_completed.pt"
    if os.path.exists(model_path):
        print(f"[*] Loading existing weights from {model_path}...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except:
            print("[!] Could not load weights, starting from scratch.")

    criterion, optimizer, scheduler, scaler = get_training_components(model)
    
    best_acc = 0.0

    # 4. Training Loop
    print("\n" + "="*50)
    print("   NEUROSCAN - OPTIMIZED TRAINING PIPELINE")
    print("="*50)

    # Initialize checkpoint timer
    last_checkpoint_time = time.time()
    try:
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for data, targets in pbar:
                # 1. Resource Monitor Check
                if should_pause(max_gpu_mem, max_cpu):
                    print(f"\n[!] Resource limits (GPU {max_gpu_mem}%, CPU {max_cpu}%) exceeded. Pausing for 60s...")
                    logging.warning("Resource limits exceeded. Training paused.")
                    time.sleep(60)
                    continue

                # 2. Log usage and process data
                log_resource_usage(max_gpu_mem, max_cpu)
                # Data processing
                data = data.to(device).to(torch.float32)
                targets = targets.to(device)
                
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
                    
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'acc': f"{100.*correct/total:.2f}%"
                })

                # 3. Intermediate Time-based Checkpoint
                if (time.time() - last_checkpoint_time) > checkpoint_interval:
                    torch.save(model.state_dict(), model_path)
                    last_checkpoint_time = time.time()
                    logging.info(f"Timed checkpoint saved: {datetime.datetime.now()}")
                    print(f"\n[*] Intermediate checkpoint saved (Interval: {checkpoint_interval}s)")

        # Validation at end of epoch
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device).to(torch.float32)
                targets = targets.to(device)
                outputs = model(data)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        print(f"--> [Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Adjust Learning Rate
        scheduler.step(avg_val_loss)
        
        # Checkpoint Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"[*] NEW BEST: {best_acc:.2f}% saved to {model_path}")
            logging.info(f"New best model: {best_acc:.2f}%")
            
        if best_acc >= 70.0:
            print(f"[!] TARGET REACHED: {best_acc:.2f}% Accuracy!")
            logging.info(f"Target accuracy {best_acc:.2f}% reached.")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving emergency checkpoint...")
        torch.save(model.state_dict(), model_path)
        logging.info("Training manually interrupted. Checkpoint saved.")
        return

    print(f"\n" + "="*50)
    print(f"TRAINING COMPLETE!")
    print(f"Best Target Accuracy: {best_acc:.2f}%")
    print(f"Model File: {model_path}")
    print("="*50)

if __name__ == "__main__":
    main()
