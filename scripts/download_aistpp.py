#!/usr/bin/env python
"""
Windows-compatible download script for AIST++ dataset
Fetches annotations + audio without dependency conflicts
"""
import os
import sys
import subprocess
import zipfile
from pathlib import Path
import urllib.request
import shutil

def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    if check:
        return subprocess.run(cmd, check=True, capture_output=True, text=True)
    else:
        return subprocess.run(cmd, capture_output=True, text=True)

def download_file(url, output_path, description="file"):
    """Download a file with progress"""
    print(f"[AIST++] downloading {description}...")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\r{description}: {percent}% [{downloaded}/{total_size} bytes]", end="")
    
    urllib.request.urlretrieve(url, output_path, show_progress)
    print()  # New line after progress

def main():
    ROOT = sys.argv[1] if len(sys.argv) > 1 else "aist_raw"
    AIST_API_DIR = Path(ROOT) / "aistplusplus_api"
    ANN_DIR = Path(ROOT) / "annotations"
    AUDIO_DIR = Path(ROOT) / "audio"
    
    # Create root directory
    Path(ROOT).mkdir(exist_ok=True)
    
    # 1) Clone API repo (shallow)
    if not AIST_API_DIR.exists():
        print("[AIST++] cloning API repo...")
        try:
            run_command(["git", "clone", "--depth", "1", 
                        "https://github.com/google/aistplusplus_api.git", 
                        str(AIST_API_DIR)])
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repo: {e}")
            print("Make sure git is installed and available in PATH")
            return 1
    
    print("[AIST++] requirements installation skipped (handled separately)")
    
    # 2) Install the AIST++ API
    if AIST_API_DIR.exists():
        print("[AIST++] installing aistplusplus_api...")
        try:
            # Install the API package in development mode
            run_command([sys.executable, "-m", "pip", "install", "-e", str(AIST_API_DIR)])
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not install aistplusplus_api: {e}")
            print("Continuing anyway...")
    
    # 3) Download annotations (≈5 GB)
    if not ANN_DIR.exists():
        ANN_DIR.mkdir(parents=True, exist_ok=True)
        print("[AIST++] downloading 3-D key-points...")
        
        try:
            # Try to use the API download
            import aistplusplus_api.cli.download as dl
            original_argv = sys.argv
            sys.argv = ["", "--type", "annotations", "--out_dir", str(ANN_DIR)]
            dl.main()
            sys.argv = original_argv
        except Exception as e:
            print(f"API download failed: {e}")
            print("You may need to download annotations manually from:")
            print("https://aistdancedb.ongaaccel.jp/")
            return 1
    else:
        print("[AIST++] annotations already present – skipping")
    
    # 4) Download audio (~9 GB) from HF mirror
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_PATH = AUDIO_DIR / "aist_audio.zip"
    
    if not (AUDIO_DIR / "wav").exists():
        print("[AIST++] downloading audio...")
        try:
            download_file(
                "https://huggingface.co/datasets/AK391/aistplusplus_audio/resolve/main/aist_audio.zip",
                str(ZIP_PATH),
                "audio"
            )
            
            print("[AIST++] extracting audio...")
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(AUDIO_DIR)
            
            # Clean up zip file
            ZIP_PATH.unlink()
            
        except Exception as e:
            print(f"Error downloading audio: {e}")
            print("You may need to download audio manually from:")
            print("https://huggingface.co/datasets/AK391/aistplusplus_audio")
            return 1
    else:
        print("[AIST++] audio already present – skipping")
    
    print(f"✓ AIST++ assets ready in {ROOT}")
    return 0

if __name__ == "__main__":
    exit(main()) 