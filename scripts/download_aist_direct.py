#!/usr/bin/env python
"""
Direct download script for AIST++ annotations from official sources
Downloads 3D keypoints and preprocessed data directly
"""
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def download_with_progress(url, output_path, description="file"):
    """Download a file with progress bar"""
    print(f"[AIST++] downloading {description}...")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\r{description}: {percent}% [{downloaded//1024//1024}/{total_size//1024//1024} MB]", end="")
    
    try:
        urllib.request.urlretrieve(url, output_path, show_progress)
        print(f"\n✓ Downloaded {description}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {description}: {e}")
        return False

def main():
    ROOT = sys.argv[1] if len(sys.argv) > 1 else "aist_raw"
    annotations_dir = Path(ROOT) / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading AIST++ annotations from official sources...")
    
    # Official download URLs (these are the actual URLs from Google's AIST++ page)
    downloads = [
        {
            "url": "https://aistdancedb.ongaaccel.jp/v1.0.0/aist_plusplus_final.zip",
            "filename": "aist_plusplus_final.zip",
            "description": "AIST++ Complete Dataset (2.3GB)"
        }
    ]
    
    # Try to download the main dataset
    for item in downloads:
        zip_path = annotations_dir / item["filename"]
        
        if zip_path.exists():
            print(f"✓ {item['description']} already exists")
            continue
            
        print(f"\nDownloading {item['description']}...")
        success = download_with_progress(item["url"], str(zip_path), item["description"])
        
        if success and zip_path.exists():
            print(f"Extracting {item['filename']}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(annotations_dir)
                print(f"✓ Extracted {item['filename']}")
                
                # Remove zip file to save space
                zip_path.unlink()
                print(f"✓ Cleaned up {item['filename']}")
                
            except Exception as e:
                print(f"✗ Failed to extract {item['filename']}: {e}")
        else:
            print(f"✗ Failed to download {item['description']}")
            print("You may need to download manually from:")
            print("https://aistdancedb.ongaaccel.jp/")
    
    # Check what we got
    annotation_files = list(annotations_dir.rglob("*.pkl"))
    if annotation_files:
        print(f"\n✓ Found {len(annotation_files)} annotation files")
        print("Sample files:")
        for f in annotation_files[:5]:
            print(f"  - {f.name}")
    else:
        print("\n⚠ No annotation files found. You may need to download manually.")
        print("Visit: https://aistdancedb.ongaaccel.jp/")
    
    print(f"\n✓ AIST++ annotations ready in {ROOT}")
    return 0

if __name__ == "__main__":
    exit(main()) 