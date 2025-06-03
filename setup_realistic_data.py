#!/usr/bin/env python
"""
Create realistic synthetic dance-drum data that mimics real patterns
This creates more sophisticated test data while we work on getting AIST++
"""
import os
import numpy as np
import pretty_midi as pm
from pathlib import Path
import shutil
import math

def create_dance_pose_sequence(duration=2.0, fps=60):
    """Create realistic dance-like pose sequence"""
    T = int(duration * fps)  # 120 frames for 2 seconds
    J = 17  # 17 COCO joints
    
    # Initialize pose array
    pose = np.zeros((T, J, 3), dtype=np.float32)
    
    # Define basic body structure (COCO joint connections)
    # 0: nose, 1-2: eyes, 3-4: ears, 5-10: arms, 11-16: legs
    
    for t in range(T):
        time_sec = t / fps
        
        # Base standing pose
        pose[t, 0] = [0, 0, 1.7]  # head
        pose[t, 5] = [-0.3, 0, 1.4]  # left shoulder  
        pose[t, 6] = [0.3, 0, 1.4]   # right shoulder
        pose[t, 11] = [-0.2, 0, 0.9] # left hip
        pose[t, 12] = [0.2, 0, 0.9]  # right hip
        pose[t, 13] = [-0.2, 0, 0.45] # left knee
        pose[t, 14] = [0.2, 0, 0.45]  # right knee
        pose[t, 15] = [-0.2, 0, 0]    # left ankle
        pose[t, 16] = [0.2, 0, 0]     # right ankle
        
        # Add realistic dance movements
        # 1. Rhythmic up-down body movement (2Hz - 120 BPM)
        bounce = 0.1 * np.sin(2 * np.pi * 2 * time_sec)
        pose[t, :, 2] += bounce
        
        # 2. Arm movements (different frequencies for each arm)
        left_arm_swing = 0.3 * np.sin(2 * np.pi * 1.5 * time_sec)
        right_arm_swing = 0.3 * np.cos(2 * np.pi * 1.3 * time_sec)
        
        pose[t, 7] = pose[t, 5] + [left_arm_swing, 0.2, -0.3]   # left elbow
        pose[t, 8] = pose[t, 6] + [right_arm_swing, 0.2, -0.3]  # right elbow
        pose[t, 9] = pose[t, 7] + [0.1, 0.3, -0.2]              # left wrist
        pose[t, 10] = pose[t, 8] + [0.1, 0.3, -0.2]             # right wrist
        
        # 3. Hip sway
        hip_sway = 0.15 * np.sin(2 * np.pi * 1 * time_sec)
        pose[t, 11, 0] += hip_sway
        pose[t, 12, 0] -= hip_sway
        
        # 4. Foot tapping (alternating)
        if int(time_sec * 4) % 2 == 0:  # every quarter beat
            pose[t, 15, 2] += 0.05  # lift left foot
        else:
            pose[t, 16, 2] += 0.05  # lift right foot
    
    # Add some noise for realism
    pose += np.random.normal(0, 0.01, pose.shape)
    
    return pose

def create_realistic_drum_pattern(duration=2.0, tempo_bpm=120):
    """Create realistic drum pattern that matches dance tempo"""
    midi = pm.PrettyMIDI()
    drum_instrument = pm.Instrument(program=0, is_drum=True)
    
    beats_per_second = tempo_bpm / 60.0
    beat_duration = 1.0 / beats_per_second
    
    # Create a realistic 4/4 drum pattern
    num_beats = int(duration * beats_per_second)
    
    for beat in range(num_beats):
        beat_time = beat * beat_duration
        
        # Kick drum on beats 1 and 3
        if beat % 4 in [0, 2]:
            kick = pm.Note(
                velocity=np.random.randint(100, 127),
                pitch=36,  # kick
                start=beat_time,
                end=beat_time + 0.1
            )
            drum_instrument.notes.append(kick)
        
        # Snare on beats 2 and 4
        if beat % 4 in [1, 3]:
            snare = pm.Note(
                velocity=np.random.randint(90, 120),
                pitch=38,  # snare
                start=beat_time,
                end=beat_time + 0.08
            )
            drum_instrument.notes.append(snare)
        
        # Hi-hat on every eighth note
        for eighth in [0, 0.5]:
            hihat_time = beat_time + eighth * beat_duration
            if hihat_time < duration:
                hihat = pm.Note(
                    velocity=np.random.randint(60, 90),
                    pitch=42,  # closed hi-hat
                    start=hihat_time,
                    end=hihat_time + 0.05
                )
                drum_instrument.notes.append(hihat)
    
    # Add some fills and variations
    if duration > 1.5:
        # Add a simple fill in the last half beat
        fill_time = duration - 0.5
        for i, pitch in enumerate([45, 47, 48]):  # tom fills
            note_time = fill_time + i * 0.1
            fill_note = pm.Note(
                velocity=np.random.randint(80, 110),
                pitch=pitch,
                start=note_time,
                end=note_time + 0.08
            )
            drum_instrument.notes.append(fill_note)
    
    midi.instruments.append(drum_instrument)
    return midi

def create_realistic_dataset(data_root="dataset_realistic", num_samples=100):
    """Create realistic dance-drum dataset"""
    
    data_path = Path(data_root)
    if data_path.exists():
        shutil.rmtree(data_path)
    
    print(f"Creating realistic dance-drum dataset with {num_samples} samples...")
    
    # Different dance styles and tempos
    styles = [
        {"name": "hip-hop", "tempo": 120, "bounce": 1.2},
        {"name": "house", "tempo": 128, "bounce": 0.8},
        {"name": "jazz", "tempo": 100, "bounce": 0.6},
        {"name": "contemporary", "tempo": 90, "bounce": 0.4},
        {"name": "breakdance", "tempo": 140, "bounce": 1.5}
    ]
    
    for i in range(num_samples):
        sample_dir = data_path / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Choose random style
        style = styles[i % len(styles)]
        duration = np.random.uniform(1.5, 3.0)  # variable duration
        
        # Create coordinated pose and drum data
        pose = create_dance_pose_sequence(duration=duration, fps=60)
        
        # Scale movements by style
        pose[:, :, :2] *= style["bounce"]  # x,y movements
        
        # Save pose
        np.save(sample_dir / "pose.npy", pose)
        
        # Create matching drum pattern
        drum_midi = create_realistic_drum_pattern(duration=duration, tempo_bpm=style["tempo"])
        drum_midi.write(str(sample_dir / "drums.mid"))
        
        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{num_samples} samples...")
    
    print(f"✓ Created {num_samples} realistic samples in {data_root}/")
    print(f"  - 5 dance styles: {[s['name'] for s in styles]}")
    print(f"  - Tempo range: {min(s['tempo'] for s in styles)} - {max(s['tempo'] for s in styles)} BPM")
    print(f"  - Variable durations: 1.5 - 3.0 seconds")

if __name__ == "__main__":
    create_realistic_dataset()
    
    print("\n" + "="*60)
    print("Enhanced realistic dataset created! You can now run:")
    print("  py choreo2groove.py --data_root dataset_realistic --epochs 10 --batch_size 4")
    print("="*60) 