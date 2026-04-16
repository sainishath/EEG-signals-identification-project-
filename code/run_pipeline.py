import os
import subprocess
import sys

def run_command(command):
    print(f"[*] Executing: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"[!] Error: {command} failed with code {process.returncode}")
        return False
    return True

def main():
    print("="*60)
    print("   NEUROSCAN - COMPLETE PREPROCESS + TRAIN PIPELINE")
    print("="*60)
    
    # 1. Start Preprocessing
    print("\n[STEP 1/2] Starting CWT Feature Extraction...")
    if not run_command("python preprocess_features.py"):
        print("[!] Preprocessing failed. Aborting pipeline.")
        sys.exit(1)
        
    # 2. Start Training
    print("\n[STEP 2/2] starting Optimized Training...")
    if not run_command("python train_optimized.py"):
        print("[!] Training failed.")
        sys.exit(1)
        
    print("\n" + "="*60)
    print("   PIPELINE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
