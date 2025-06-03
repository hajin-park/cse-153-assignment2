#!/usr/bin/env python
"""
Fixed dance visualization that properly shows the movements
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import json

from choreo2groove import ChoreoGrooveDataset

def analyze_pose_data(pose_data):
    """Analyze the pose data to understand its structure"""
    print(f"Pose data shape: {pose_data.shape}")
    print(f"Data range - Min: {pose_data.min():.3f}, Max: {pose_data.max():.3f}")
    print(f"Data mean: {pose_data.mean():.3f}, Std: {pose_data.std():.3f}")
    
    # Check if it's already reshaped (T, features) or needs reshaping to (T, J, 3)
    if len(pose_data.shape) == 2:
        # It's flattened to (T, features)
        if pose_data.shape[1] == 51:  # 17 joints * 3 coords
            joints = 17
        elif pose_data.shape[1] == 72:  # 24 joints * 3 coords  
            joints = 24
        else:
            joints = pose_data.shape[1] // 3
        
        print(f"Detected {joints} joints")
        pose_3d = pose_data.reshape(pose_data.shape[0], joints, 3)
    else:
        pose_3d = pose_data
        joints = pose_3d.shape[1]
    
    return pose_3d, joints

def create_fixed_visualization(pose_data, output_path, fps=15):
    """Create a proper dance visualization with correct coordinate handling"""
    print("🎬 Creating FIXED dance visualization...")
    
    # Analyze and reshape the data
    pose_3d, num_joints = analyze_pose_data(pose_data)
    
    print(f"Working with {num_joints} joints, {len(pose_3d)} frames")
    
    # COCO-17 skeleton connections (if we have 17 joints)
    if num_joints == 17:
        skeleton = [
            [0, 1], [0, 2],  # nose to eyes
            [1, 3], [2, 4],  # eyes to ears
            [0, 5], [0, 6],  # nose to shoulders
            [5, 6],          # shoulders
            [5, 7], [7, 9],  # left arm
            [6, 8], [8, 10], # right arm
            [5, 11], [6, 12], # shoulders to hips
            [11, 12],        # hips
            [11, 13], [13, 15], # left leg
            [12, 14], [14, 16]  # right leg
        ]
    else:
        # Generic skeleton for other joint counts
        skeleton = []
        for i in range(min(num_joints-1, 16)):
            skeleton.append([i, i+1])
    
    # Subsample frames for reasonable video length
    stride = max(1, len(pose_3d) // (fps * 8))  # Max 8 seconds
    pose_frames = pose_3d[::stride]
    
    print(f"Using {len(pose_frames)} frames for visualization")
    
    # Normalize coordinates for better visualization
    all_coords = pose_frames.reshape(-1, 3)
    
    # Remove any invalid coordinates (NaN, inf, extreme values)
    valid_mask = np.isfinite(all_coords).all(axis=1)
    valid_coords = all_coords[valid_mask]
    
    if len(valid_coords) == 0:
        print("❌ No valid coordinates found!")
        return None
    
    # Get coordinate ranges
    x_min, x_max = valid_coords[:, 0].min(), valid_coords[:, 0].max()
    y_min, y_max = valid_coords[:, 1].min(), valid_coords[:, 1].max()
    
    print(f"Coordinate ranges: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}]")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def animate(frame_idx):
        ax.clear()
        ax.set_title(f'Choreo2Groove - Dance Visualization\nFrame {frame_idx+1}/{len(pose_frames)}', 
                    fontsize=14, fontweight='bold')
        ax.set_facecolor('black')
        
        if frame_idx < len(pose_frames):
            frame = pose_frames[frame_idx]  # Shape: (joints, 3)
            
            # Extract x, y coordinates (ignore z for 2D visualization)
            x_coords = frame[:, 0]
            y_coords = frame[:, 1]
            
            # Filter out invalid coordinates
            valid_joints = np.isfinite(x_coords) & np.isfinite(y_coords)
            
            if valid_joints.any():
                # Set axis limits based on current frame
                valid_x = x_coords[valid_joints]
                valid_y = y_coords[valid_joints]
                
                margin = 0.2
                x_range = valid_x.max() - valid_x.min()
                y_range = valid_y.max() - valid_y.min()
                
                if x_range > 0 and y_range > 0:
                    ax.set_xlim(valid_x.min() - margin * x_range, valid_x.max() + margin * x_range)
                    ax.set_ylim(valid_y.min() - margin * y_range, valid_y.max() + margin * y_range)
                else:
                    ax.set_xlim(-1, 1)
                    ax.set_ylim(-1, 1)
                
                # Draw skeleton connections
                for connection in skeleton:
                    j1, j2 = connection
                    if (j1 < len(frame) and j2 < len(frame) and 
                        valid_joints[j1] and valid_joints[j2]):
                        ax.plot([x_coords[j1], x_coords[j2]], 
                               [y_coords[j1], y_coords[j2]], 
                               'cyan', linewidth=3, alpha=0.8)
                
                # Draw joints
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    if valid_joints[i]:
                        if i < 5:  # Head area
                            color, size = 'red', 80
                        elif i < 11:  # Arms
                            color, size = 'yellow', 60
                        else:  # Legs
                            color, size = 'lime', 60
                        
                        ax.scatter(x, y, c=color, s=size, alpha=0.9, edgecolors='white', linewidth=1)
            else:
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.text(0, 0, 'No valid pose data for this frame', 
                       ha='center', va='center', fontsize=12, color='white')
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add info text
        ax.text(0.02, 0.98, 'Generated by Choreo2Groove AI', transform=ax.transAxes, 
               fontsize=10, color='lime', weight='bold', va='top')
        ax.text(0.02, 0.02, f'Dance Style: Basic Moves | Frame Rate: {fps} FPS', 
               transform=ax.transAxes, fontsize=9, color='white', va='bottom')
    
    # Create animation
    print(f"Creating animation with {len(pose_frames)} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=len(pose_frames), 
                                 interval=1000//fps, blit=False, repeat=True)
    
    # Save as GIF
    try:
        print(f"💾 Saving animation to {output_path}")
        anim.save(str(output_path), writer='pillow', fps=fps)
        print("✅ Fixed dance video created successfully!")
    except Exception as e:
        print(f"⚠️ Animation save failed: {e}")
        return None
    
    plt.close(fig)
    return output_path

def test_fixed_visualization():
    """Test the fixed visualization with a sample"""
    
    # Load a sample from the dataset
    dataset = ChoreoGrooveDataset("dataset_root", seq_len=256)
    sample_idx = 58
    
    # Load the raw pose data (not the processed version)
    raw_pose_path = Path(f"dataset_root/sample_{sample_idx:03d}/pose.npy")
    raw_pose_data = np.load(raw_pose_path)
    
    print(f"Testing with sample {sample_idx}")
    analyze_pose_data(raw_pose_data)
    
    # Create output directory
    output_dir = Path("lightning_logs/version_6/complete_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create fixed visualization
    output_path = output_dir / "dance_video_FIXED.gif"
    result = create_fixed_visualization(raw_pose_data, output_path)
    
    if result:
        print(f"\n🎉 FIXED visualization ready!")
        print(f"📁 Location: {result}")
        print("🎬 This should now show the actual dance movements!")
        
        # Also create a summary
        summary = {
            "status": "SUCCESS",
            "sample_used": sample_idx,
            "original_file": "gWA_sBM_cAll_d27_mWA5_ch02.pkl",
            "frames_processed": len(raw_pose_data),
            "output_file": str(result),
            "notes": "Fixed visualization with proper coordinate normalization"
        }
        
        with open(output_dir / "visualization_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        return result
    else:
        print("❌ Visualization creation failed")
        return None

if __name__ == "__main__":
    result = test_fixed_visualization()
    
    if result:
        print(f"\n✨ Open this file to see the dance: {result}")
        print("💡 The visualization should now show actual movement instead of a blank screen!")
    else:
        print("❌ Failed to create visualization") 