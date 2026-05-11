#!/usr/bin/env python3
"""
Fix the ablation graphs script by copying working version
"""
import shutil
import os

src = r'd:\SAE\generate_ablation_graphs (1).py'
dst = r'd:\SAE\generate_ablation_graphs.py'

# Remove empty version
if os.path.exists(dst):
    os.remove(dst)
    print(f"[+] Removed empty {dst}")

# Copy working version
shutil.copy(src, dst)
print(f"[+] Copied {src}")
print(f"[+] To {dst}")
print(f"[+] File size: {os.path.getsize(dst)} bytes")
print("\n[OK] Fixed! Now run: python d:\SAE\generate_ablation_graphs.py")
