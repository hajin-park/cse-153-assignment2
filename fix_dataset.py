#!/usr/bin/env python
"""
Fix incomplete samples in the dataset
Remove samples that don't have both pose.npy and drums.mid
"""
import os
from pathlib import Path
import shutil

def fix_dataset(dataset_dir="dataset_root"):
    """Remove incomplete samples and renumber the rest"""
    dataset_path = Path(dataset_dir)
    
    # Find all sample directories
    sample_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("sample_")])
    
    incomplete_samples = []
    complete_samples = []
    
    for sample_dir in sample_dirs:
        pose_file = sample_dir / "pose.npy"
        drums_file = sample_dir / "drums.mid"
        
        if pose_file.exists() and drums_file.exists():
            complete_samples.append(sample_dir)
        else:
            incomplete_samples.append(sample_dir)
            print(f"Removing incomplete sample: {sample_dir.name}")
            if pose_file.exists() and not drums_file.exists():
                print(f"  Missing: drums.mid")
            elif drums_file.exists() and not pose_file.exists():
                print(f"  Missing: pose.npy")
            else:
                print(f"  Missing: both files")
    
    # Remove incomplete samples
    for sample_dir in incomplete_samples:
        shutil.rmtree(sample_dir)
    
    print(f"\\nFound {len(complete_samples)} complete samples")
    print(f"Removed {len(incomplete_samples)} incomplete samples")
    
    return len(complete_samples)

if __name__ == "__main__":
    count = fix_dataset()
    print(f"Dataset now has {count} valid samples ready for training!") 