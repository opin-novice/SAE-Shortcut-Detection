# SAE Shortcut Detection - Ablation Graphs Fix

## Problem
The `generate_ablation_graphs.py` file was empty (0 bytes), preventing ablation study graph generation.

## Root Cause
When originally cloning/downloading, the correct version was saved as `generate_ablation_graphs (1).py` with the working code, while the default `generate_ablation_graphs.py` remained empty.

## Solution
Copy the working version to the expected location:

```bash
copy "d:\SAE\generate_ablation_graphs (1).py" "d:\SAE\generate_ablation_graphs.py"
```

Or using Python:
```bash
python d:\SAE\fix_ablation.py
```

## What the Script Does
Generates 5 publication-quality ablation study graphs:

1. **Ablation 1**: Feature Count Scaling
   - Shows accuracy vs number of ablated features (0→36)
   - Key insight: Gap stays constant regardless of count

2. **Ablation 2**: Per-Attribute Analysis  
   - Compares shortcut counts across 5 attributes
   - Shows each attribute contributes ~-0.005 to -0.013 gap reduction

3. **Ablation 3**: Bias Ratio Sensitivity ⭐ KEY FINDING
   - Tests across ratios: 50/50, 60/40, 70/30, 80/20, 90/10, 95/5, 99/1
   - Ablation works well for moderate bias (50/50 → 90/10)
   - Fails at extreme ratios (95/5+) where bias is too strong

4. **Ablation 4**: Pooling Method Comparison
   - Compares mean vs max pooling for feature detection
   - Max pooling finds stronger correlations (r=0.734 vs 0.613)

5. **Ablation 5**: Seed Stability
   - Confirms reproducibility across random seeds
   - Shortcut counts stable: 23-31 features

## To Run

```bash
cd d:\SAE
d:\SAE\venv\Scripts\python.exe d:\SAE\generate_ablation_graphs.py
```

## Output
Creates 5 PNG files in `d:\SAE\graph_res\`:
- `ablation1_feature_count_scaling.png`
- `ablation2_per_attribute.png`
- `ablation3_bias_ratio_sensitivity.png` (most important)
- `ablation4_pooling_method.png`
- `ablation5_seed_stability.png`

## Current Status
✅ Fix provided: Use `fix_ablation.py` or batch copy command above
✅ Script ready to run
⏳ Awaiting execution on your system
