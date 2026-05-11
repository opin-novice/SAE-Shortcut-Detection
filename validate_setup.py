#!/usr/bin/env python
"""
Validation script: Test SAE, CLIP, and dataset loading
Run with: d:\SAE\venv\Scripts\python.exe validate_setup.py
"""

import sys
import os

# Add patchsae to path
sys.path.insert(0, r'd:\SAE\patchsae\src')
sys.path.insert(0, r'd:\SAE\patchsae')

os.chdir(r'd:\SAE')

print("=" * 70)
print("VALIDATING SAE PROJECT SETUP")
print("=" * 70)

# Test 1: PyTorch & CUDA
print("\n[1] Checking PyTorch & CUDA...")
try:
    import torch
    print(f"    PyTorch version: {torch.__version__}")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print("    [OK]")
except Exception as e:
    print(f"    [ERROR] {e}")
    sys.exit(1)

# Test 2: Core ML libraries
print("\n[2] Checking core ML libraries...")
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from transformers import CLIPModel
    import open_clip
    print("    numpy, pandas, matplotlib, transformers, open_clip: OK")
    print("    [OK]")
except Exception as e:
    print(f"    [ERROR] {e}")
    sys.exit(1)

# Test 3: Dataset loading
print("\n[3] Loading Brain Tumor MRI dataset...")
try:
    from datasets import load_dataset
    print("    Loading from HuggingFace...")
    ds = load_dataset("Hemg/Brain-Tumor-MRI-Dataset")
    print(f"    Splits available: {list(ds.keys())}")
    split = list(ds.keys())[0]
    num_samples = len(ds[split])
    print(f"    First split ({split}): {num_samples} samples")
    print("    [OK]")
except Exception as e:
    print(f"    [ERROR] {e}")
    sys.exit(1)

# Test 4: CLIP model
print("\n[4] Loading CLIP model...")
try:
    from transformers import CLIPProcessor, CLIPModel
    model_name = "openai/clip-vit-base-patch32"
    print(f"    Loading: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    print(f"    Model loaded, moving to GPU...")
    if torch.cuda.is_available():
        model = model.cuda()
        print("    [OK] on CUDA")
    else:
        print("    [OK] on CPU")
except Exception as e:
    print(f"    [ERROR] {e}")
    sys.exit(1)

# Test 5: SAE checkpoint
print("\n[5] Checking SAE checkpoint files...")
try:
    sae_path = r'd:\SAE\data\sae_weight\base\out.pt'
    if os.path.exists(sae_path):
        size = os.path.getsize(sae_path) / (1024**2)
        print(f"    SAE weights found: {sae_path}")
        print(f"    Size: {size:.1f} MB")
        print("    [OK]")
    else:
        print(f"    [ERROR] SAE weights not found at {sae_path}")
        sys.exit(1)
except Exception as e:
    print(f"    [ERROR] {e}")
    sys.exit(1)

# Test 6: Feature data
print("\n[6] Checking feature data...")
try:
    feature_base = r'd:\SAE\out\feature_data\sae_base\base'
    if os.path.exists(feature_base):
        datasets_available = os.listdir(feature_base)
        print(f"    Feature datasets available: {len(datasets_available)}")
        for ds_name in datasets_available[:3]:
            print(f"      - {ds_name}")
        if len(datasets_available) > 3:
            print(f"      ... and {len(datasets_available) - 3} more")
        print("    [OK]")
    else:
        print(f"    [ERROR] Feature data not found at {feature_base}")
        sys.exit(1)
except Exception as e:
    print(f"    [ERROR] {e}")
    sys.exit(1)

# Test 7: PatchSAE imports
print("\n[7] Checking PatchSAE imports...")
try:
    from tasks.utils import load_sae
    print("    tasks.utils.load_sae: OK")
    print("    [OK]")
except Exception as e:
    print(f"    [ERROR] {e}")
    print("    This is expected if PatchSAE structure is different")
    print("    The analysis script will still work")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE!")
print("=" * 70)
print("\nReady to run SAE analysis!")
print("Next: python sae_shortcut_detection_run.py")
print("=" * 70)
