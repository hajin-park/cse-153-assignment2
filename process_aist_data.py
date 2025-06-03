#!/usr/bin/env python
"""
Process real AIST++ dataset for Choreo2Groove training
Converts pickle files to pose.npy and creates synthetic drum data
"""
import os
import pickle
import numpy as np
import pretty_midi as pm
from pathlib import Path
import shutil
import random

def load_aist_pose(pkl_path):
    """Load 3D pose data from AIST++ pickle file"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # AIST++ format: {'keypoints3d': array of shape (T, J*3) or similar}
    if 'keypoints3d' in data:
        keypoints = data['keypoints3d']
    elif 'poses' in data:
        keypoints = data['poses']
    else:
        # Try to find the main data array
        keypoints = list(data.values())[0]
    
    # Reshape to (T, J, 3) if needed
    if keypoints.ndim == 2:
        # Assume format is (T, J*3) where J=17 (COCO) or J=24 (SMPL)
        if keypoints.shape[1] == 51:  # 17*3
            keypoints = keypoints.reshape(-1, 17, 3)
        elif keypoints.shape[1] == 72:  # 24*3  
            keypoints = keypoints.reshape(-1, 24, 3)
        else:
            # Fallback: try to guess joint count
            n_coords = keypoints.shape[1]
            if n_coords % 3 == 0:
                n_joints = n_coords // 3
                keypoints = keypoints.reshape(-1, n_joints, 3)
    
    print(f"Loaded pose data: {keypoints.shape}")
    return keypoints

def create_drum_from_pose(pose_data, duration=None, bpm=120):
    """Create a drum pattern based on pose movement energy"""
    T, J, _ = pose_data.shape
    fps = 60  # AIST++ is 60 fps
    
    if duration is None:
        duration = T / fps
    
    # Calculate movement energy
    velocity = np.diff(pose_data, axis=0)  # (T-1, J, 3)
    energy = np.sqrt(np.sum(velocity**2, axis=(1, 2)))  # (T-1,)
    
    # Smooth energy signal
    from scipy.ndimage import gaussian_filter1d
    energy_smooth = gaussian_filter1d(energy, sigma=2.0)
    
    # Create MIDI
    midi = pm.PrettyMIDI()
    drums = pm.Instrument(program=0, is_drum=True, name="Drums")
    
    # Time grid aligned to beats
    beat_duration = 60.0 / bpm
    time_grid = np.arange(0, duration, beat_duration/4)  # 16th notes
    
    # Map energy to time grid
    energy_grid = np.interp(time_grid, 
                           np.linspace(0, duration, len(energy_smooth)), 
                           energy_smooth)
    
    # Thresholds for different drums
    kick_thresh = np.percentile(energy_grid, 75)
    snare_thresh = np.percentile(energy_grid, 65) 
    hihat_thresh = np.percentile(energy_grid, 40)
    
    for i, (time, eng) in enumerate(zip(time_grid, energy_grid)):
        velocity = min(127, max(30, int(eng * 100)))
        
        # Kick on strong beats when energy is high
        if i % 4 == 0 and eng > kick_thresh:
            drums.notes.append(pm.Note(36, velocity, time, time + 0.1))
        
        # Snare on off-beats when energy is medium-high  
        if i % 8 == 4 and eng > snare_thresh:
            drums.notes.append(pm.Note(38, velocity, time, time + 0.1))
            
        # Hi-hat for regular rhythm
        if eng > hihat_thresh:
            drums.notes.append(pm.Note(42, velocity//2, time, time + 0.05))
    
    midi.instruments.append(drums)
    return midi

def process_aist_dataset(aist_raw_dir="aist_raw", output_dir="dataset_root", max_samples=100):
    """Process AIST++ data into training format"""
    
    aist_raw_path = Path(aist_raw_dir)
    keypoints_dir = aist_raw_path / "aist_plusplus_final" / "keypoints3d"
    output_path = Path(output_dir)
    
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    # Get all pickle files
    pkl_files = list(keypoints_dir.glob("*.pkl"))
    print(f"Found {len(pkl_files)} pose files")
    
    # Sample random subset for manageable processing
    if len(pkl_files) > max_samples:
        pkl_files = random.sample(pkl_files, max_samples)
        print(f"Using {max_samples} samples for training")
    
    success_count = 0
    
    for i, pkl_file in enumerate(pkl_files):
        try:
            print(f"Processing {i+1}/{len(pkl_files)}: {pkl_file.name}")
            
            # Load pose data
            pose_data = load_aist_pose(pkl_file)
            
            # Skip if too short or too long
            duration = pose_data.shape[0] / 60.0  # Convert frames to seconds
            if duration < 5.0 or duration > 30.0:
                print(f"  Skipping: duration {duration:.1f}s not in range [5, 30]s")
                continue
            
            # Create sample directory
            sample_dir = output_path / f"sample_{success_count:03d}"
            sample_dir.mkdir(exist_ok=True)
            
            # Save pose data
            np.save(sample_dir / "pose.npy", pose_data.astype(np.float32))
            
            # Create and save drum pattern
            drum_midi = create_drum_from_pose(pose_data)
            drum_midi.write(str(sample_dir / "drums.mid"))
            
            # Save metadata
            metadata = {
                "original_file": pkl_file.name,
                "duration": duration,
                "frames": pose_data.shape[0],
                "joints": pose_data.shape[1]
            }
            
            with open(sample_dir / "metadata.txt", "w") as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            success_count += 1
            print(f"  ✓ Created sample_{success_count-1:03d}")
            
        except Exception as e:
            print(f"  ✗ Error processing {pkl_file.name}: {e}")
            continue
    
    print(f"\n✅ Successfully processed {success_count} samples!")
    print(f"Dataset ready at: {output_path}")
    return success_count

if __name__ == "__main__":
    import sys
    
    # Install scipy if needed
    try:
        import scipy.ndimage
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        import scipy.ndimage
    
    count = process_aist_dataset(max_samples=100)
    print(f"Ready to train with {count} real dance sequences!") 