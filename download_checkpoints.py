#!/usr/bin/env python
"""
Download and extract SAE checkpoints from Google Drive
Run with: python download_checkpoints.py
"""

import subprocess
import os
import zipfile
import sys

os.chdir(r'd:\SAE')

print("=" * 60)
print("DOWNLOADING SAE CHECKPOINTS")
print("=" * 60)

# File IDs from Google Drive
files = {
    'out.zip': '1NJzF8PriKz_mopBY4l8_44R0FVi2uw2g',
    'data.zip': '1reuDjXsiMkntf1JJPLC5a3CcWuJ6Ji3Z'
}

for filename, file_id in files.items():
    print(f"\nDownloading {filename}...")
    
    # Use gdown to download from Google Drive
    cmd = [
        sys.executable, '-m', 'gdown',
        f'https://drive.google.com/uc?id={file_id}',
        '-O', filename,
        '--quiet'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        file_size = os.path.getsize(filename) / (1024**2)
        print(f"[OK] Downloaded {filename} ({file_size:.1f} MB)")
    else:
        print(f"[ERROR] Failed to download {filename}")
        if result.stderr:
            print(result.stderr[:500])
        
print("\n" + "=" * 60)
print("EXTRACTING FILES")
print("=" * 60)

# Extract both files
for filename in files.keys():
    if os.path.exists(filename):
        print(f"\nExtracting {filename}...")
        
        try:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(r'd:\SAE')
            print(f"[OK] Extracted {filename}")
        except Exception as e:
            print(f"[ERROR] Error extracting {filename}: {e}")
    else:
        print(f"[ERROR] {filename} not found")

print("\n" + "=" * 60)
print("VERIFYING EXTRACTED FILES")
print("=" * 60)

# Check what was extracted
for folder in ['out', 'data']:
    path = os.path.join(r'd:\SAE', folder)
    if os.path.exists(path):
        contents = os.listdir(path)
        print(f"\n[OK] {folder}/ created with {len(contents)} items")
        for item in contents[:3]:
            print(f"      - {item}")
        if len(contents) > 3:
            print(f"      ... and {len(contents) - 3} more")
    else:
        print(f"\n[ERROR] {folder}/ not found")

print("\n" + "=" * 60)
print("COMPLETE!")
print("=" * 60)
