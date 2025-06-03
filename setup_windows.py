#!/usr/bin/env python
"""
Windows-compatible setup script for Choreo2Groove
Creates synthetic test data to verify the training pipeline works
"""
import os
import numpy as np
import pretty_midi as pm
from pathlib import Path
import shutil

def create_synthetic_data(data_root="dataset_root", num_samples=50):
    """Create synthetic pose and drum data for testing"""
    
    data_path = Path(data_root)
    if data_path.exists():
        shutil.rmtree(data_path)
    
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    for i in range(num_samples):
        sample_dir = data_path / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic pose data: (T, J, 3) where T=120 frames, J=17 joints
        # Simulate 2 seconds at 60fps with 17 body joints
        T, J = 120, 17  
        pose = np.random.randn(T, J, 3).astype(np.float32)
        
        # Add some temporal structure (simple oscillation)
        for t in range(T):
            pose[t] *= (1 + 0.3 * np.sin(2 * np.pi * t / 30))  # 2Hz oscillation
        
        np.save(sample_dir / "pose.npy", pose)
        
        # Create synthetic drum MIDI
        midi = pm.PrettyMIDI()
        drum_instrument = pm.Instrument(program=0, is_drum=True)
        
        # Add some random drum hits
        duration = 2.0  # 2 seconds
        drum_pitches = [36, 38, 42]  # kick, snare, hi-hat
        
        for _ in range(np.random.randint(8, 16)):  # 8-16 hits
            pitch = np.random.choice(drum_pitches)
            start_time = np.random.uniform(0, duration - 0.1)
            note = pm.Note(
                velocity=np.random.randint(80, 127),
                pitch=pitch,
                start=start_time,
                end=start_time + 0.05
            )
            drum_instrument.notes.append(note)
        
        # Sort notes by start time
        drum_instrument.notes.sort(key=lambda x: x.start)
        midi.instruments.append(drum_instrument)
        midi.write(str(sample_dir / "drums.mid"))
    
    print(f"✓ Created {num_samples} synthetic samples in {data_root}/")

def install_missing_deps():
    """Install any missing dependencies"""
    import subprocess
    import sys
    
    required_packages = [
        "pytorch-lightning",
        "pretty_midi", 
        "tqdm",
        "numpy"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    print("Setting up Choreo2Groove for Windows...")
    
    # Install dependencies
    install_missing_deps()
    
    # Create synthetic test data
    create_synthetic_data()
    
    print("\n" + "="*50)
    print("Setup complete! You can now run:")
    print("  python choreo2groove.py --data_root dataset_root --epochs 5 --batch_size 4")
    print("="*50) 