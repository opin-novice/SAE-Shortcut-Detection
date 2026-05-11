"""
Generate publication-quality graphs for Ablation Studies 1-5.
Saves all figures to d:/SAE/graph_res/
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# ── Global style ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SAVE_DIR = r"d:\SAE\graph_res"
os.makedirs(SAVE_DIR, exist_ok=True)

# Color palette
C_ALIGNED  = "#4C72B0"   # steel blue
C_SHIFTED  = "#DD8452"   # warm orange
C_GAP      = "#55A868"   # green
C_BAR1     = "#8172B3"   # muted purple
C_BAR2     = "#C44E52"   # muted red
C_ACCENT   = "#CCB974"   # gold
C_HIGHLIGHT = "#E74C3C"  # bright red for emphasis


# ═══════════════════════════════════════════════════════════════════════════
# Ablation 1: Feature Count Scaling
# ═══════════════════════════════════════════════════════════════════════════
def plot_ablation1():
    n_features = [0, 5, 10, 20, 36]
    labels     = ["0\n(baseline)", "5", "10", "20", "36\n(all)"]
    aligned    = [0.996, 0.993, 0.993, 0.993, 0.993]
    shifted    = [0.940, 0.928, 0.931, 0.926, 0.932]
    gap        = [0.056, 0.066, 0.063, 0.067, 0.061]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # Plot Aligned & Shifted accuracy
    ax1.plot(n_features, aligned, 'o-', color=C_ALIGNED, linewidth=2.2,
             markersize=8, label='Aligned Accuracy', zorder=3)
    ax1.plot(n_features, shifted, 's-', color=C_SHIFTED, linewidth=2.2,
             markersize=8, label='Shifted Accuracy', zorder=3)
    ax1.set_xlabel("Number of Features Ablated")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.90, 1.005)
    ax1.set_xticks(n_features)
    ax1.set_xticklabels(labels)

    # Secondary axis for Gap
    ax2 = ax1.twinx()
    ax2.bar(n_features, gap, width=2.5, alpha=0.30, color=C_GAP, label='Gap', zorder=1)
    ax2.set_ylabel("Gap (Aligned − Shifted)", color=C_GAP)
    ax2.tick_params(axis='y', labelcolor=C_GAP)
    ax2.set_ylim(0, 0.12)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(C_GAP)

    # Annotate gap values
    for x, g in zip(n_features, gap):
        ax2.text(x, g + 0.003, f"+{g:.3f}", ha='center', fontsize=9, color=C_GAP, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', framealpha=0.9)

    ax1.set_title("Ablation 1: Feature Count Scaling", fontweight='bold', pad=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add insight annotation
    fig.text(0.5, -0.02,
             "Ablating more features doesn't help — the gap stays similar.\n"
             "The spurious signal is distributed across many features.",
             ha='center', fontsize=10, style='italic', color='#555555')

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "ablation1_feature_count_scaling.png"), pad_inches=0.3)
    plt.close(fig)
    print("  [OK] Ablation 1 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Ablation 2: Per-Attribute
# ═══════════════════════════════════════════════════════════════════════════
def plot_ablation2():
    attributes    = ["contrast", "texture_\nuniformity", "dark_\nborder", "freq_\nratio", "edge_\ndensity"]
    shortcuts     = [12, 24, 27, 13, 21]
    gap_reduction = [-0.013, -0.008, -0.007, -0.007, -0.005]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(attributes))
    width = 0.38

    bars1 = ax1.bar(x - width/2, shortcuts, width, color=C_BAR1, alpha=0.85,
                    label='Shortcuts Found', edgecolor='white', linewidth=0.8)
    ax1.set_ylabel("Shortcuts Found", color=C_BAR1)
    ax1.tick_params(axis='y', labelcolor=C_BAR1)
    ax1.set_ylim(0, 35)

    # Annotate bar values
    for bar, v in zip(bars1, shortcuts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                 str(v), ha='center', fontsize=10, fontweight='bold', color=C_BAR1)

    # Secondary axis for gap reduction
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, gap_reduction, width, color=C_BAR2, alpha=0.85,
                    label='Gap Reduction', edgecolor='white', linewidth=0.8)
    ax2.set_ylabel("Gap Reduction", color=C_BAR2)
    ax2.tick_params(axis='y', labelcolor=C_BAR2)
    ax2.set_ylim(-0.020, 0.005)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(C_BAR2)
    ax2.axhline(0, color='gray', linewidth=0.8, linestyle='-')

    for bar, v in zip(bars2, gap_reduction):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.001,
                 f"{v:.3f}", ha='center', va='top', fontsize=9, fontweight='bold', color=C_BAR2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(attributes)
    ax1.set_xlabel("Attribute")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)

    ax1.set_title("Ablation 2: Per-Attribute Analysis", fontweight='bold', pad=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    fig.text(0.5, -0.02,
             "No single attribute's features improve robustness when ablated —\n"
             "they all carry mixed (task + spurious) information.",
             ha='center', fontsize=10, style='italic', color='#555555')

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "ablation2_per_attribute.png"), pad_inches=0.3)
    plt.close(fig)
    print("  [OK] Ablation 2 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Ablation 3: Bias Ratio Sensitivity  (THE KEY FINDING)
# ═══════════════════════════════════════════════════════════════════════════
def plot_ablation3():
    ratios         = ["50/50", "60/40", "70/30", "80/20", "90/10", "95/5", "99/1"]
    baseline_gap   = [-0.006, 0.004, 0.009, 0.019, 0.027, 0.044, 0.054]
    ablated_gap    = [-0.020, -0.009, -0.002, 0.004, 0.012, 0.045, 0.052]
    gap_reduction  = [0.014, 0.012, 0.010, 0.015, 0.015, -0.001, 0.002]
    shifted_abl    = [0.989, 0.986, 0.983, 0.980, 0.977, 0.946, 0.938]

    x = np.arange(len(ratios))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9), height_ratios=[1.1, 1])

    # ── Top panel: Baseline Gap vs Ablated Gap ──
    ax_top.plot(x, baseline_gap, 'o-', color=C_ALIGNED, linewidth=2.2, markersize=8,
                label='Baseline Gap')
    ax_top.plot(x, ablated_gap, 's-', color=C_SHIFTED, linewidth=2.2, markersize=8,
                label='Ablated Gap')
    ax_top.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(ratios)
    ax_top.set_ylabel("Gap (Aligned − Shifted)")
    ax_top.set_xlabel("Bias Ratio")
    ax_top.legend(loc='upper left', framealpha=0.9)
    ax_top.set_title("Ablation 3: Bias Ratio Sensitivity — THE KEY FINDING",
                     fontweight='bold', pad=12, color='#222')
    ax_top.grid(axis='y', alpha=0.3, linestyle='--')

    # Highlight effective region (50/50 → 90/10)
    ax_top.axvspan(-0.5, 4.5, alpha=0.08, color=C_GAP, label='Ablation effective')
    ax_top.axvspan(4.5, 6.5, alpha=0.08, color=C_HIGHLIGHT, label='Ablation fails')
    ax_top.legend(loc='upper left', framealpha=0.9)

    # ── Bottom panel: Gap Reduction & Shifted Accuracy ──
    colors_bar = [C_GAP if g > 0.005 else '#AAAAAA' for g in gap_reduction]
    bars = ax_bot.bar(x, gap_reduction, 0.55, color=colors_bar, alpha=0.85,
                      edgecolor='white', linewidth=0.8, label='Gap Reduction')
    ax_bot.axhline(0, color='gray', linewidth=0.8)
    ax_bot.set_ylabel("Gap Reduction", color=C_GAP)
    ax_bot.tick_params(axis='y', labelcolor=C_GAP)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(ratios)
    ax_bot.set_xlabel("Bias Ratio")

    for bar, v in zip(bars, gap_reduction):
        offset = 0.001 if v >= 0 else -0.002
        ax_bot.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                    f"+{v:.3f}" if v > 0 else f"{v:.3f}",
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=9, fontweight='bold', color=C_GAP)

    # Overlay shifted accuracy on secondary axis
    ax3 = ax_bot.twinx()
    ax3.plot(x, shifted_abl, 'D-', color=C_HIGHLIGHT, linewidth=2, markersize=7,
             label='Shifted Acc (ablated)', alpha=0.85)
    ax3.set_ylabel("Shifted Accuracy (Ablated)", color=C_HIGHLIGHT)
    ax3.tick_params(axis='y', labelcolor=C_HIGHLIGHT)
    ax3.set_ylim(0.92, 1.00)
    ax3.spines['right'].set_visible(True)
    ax3.spines['right'].set_color(C_HIGHLIGHT)

    lines_b, labels_b = ax_bot.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    ax_bot.legend(lines_b + lines_3, labels_b + labels_3, loc='lower left', framealpha=0.9)

    ax_bot.set_title("Gap Reduction & Shifted Accuracy After Ablation", fontweight='bold', pad=10)
    ax_bot.grid(axis='y', alpha=0.3, linestyle='--')

    fig.text(0.5, -0.01,
             "Ablation WORKS for bias ratios 50/50 → 90/10 (gap reductions +0.010 to +0.015).\n"
             "It only fails at extreme ratios (95/5, 99/1) where bias is too strong for feature-level ablation.",
             ha='center', fontsize=10, style='italic', color='#555555')

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "ablation3_bias_ratio_sensitivity.png"), pad_inches=0.3)
    plt.close(fig)
    print("  [OK] Ablation 3 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Ablation 4: Pooling Method
# ═══════════════════════════════════════════════════════════════════════════
def plot_ablation4():
    methods    = ["Mean Pooling", "Max Pooling"]
    shortcuts  = [27, 24]
    max_r_spur = [0.613, 0.734]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    x = np.arange(len(methods))
    width = 0.32

    bars1 = ax1.bar(x - width/2, shortcuts, width, color=C_BAR1, alpha=0.85,
                    label='dark_border Shortcuts', edgecolor='white', linewidth=0.8)
    ax1.set_ylabel("Shortcuts Found (dark_border)", color=C_BAR1)
    ax1.tick_params(axis='y', labelcolor=C_BAR1)
    ax1.set_ylim(0, 35)

    for bar, v in zip(bars1, shortcuts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(v), ha='center', fontsize=12, fontweight='bold', color=C_BAR1)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, max_r_spur, width, color=C_BAR2, alpha=0.85,
                    label='Max r_spur', edgecolor='white', linewidth=0.8)
    ax2.set_ylabel("Max Spurious Correlation (r_spur)", color=C_BAR2)
    ax2.tick_params(axis='y', labelcolor=C_BAR2)
    ax2.set_ylim(0, 1.0)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(C_BAR2)

    for bar, v in zip(bars2, max_r_spur):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f"{v:.3f}", ha='center', fontsize=12, fontweight='bold', color=C_BAR2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=13, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

    ax1.set_title("Ablation 4: Pooling Method Comparison", fontweight='bold', pad=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    fig.text(0.5, -0.02,
             "Max pooling finds stronger correlations (r=0.734 vs 0.613)\n"
             "but slightly fewer shortcuts. Both methods detect bias.",
             ha='center', fontsize=10, style='italic', color='#555555')

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "ablation4_pooling_method.png"), pad_inches=0.3)
    plt.close(fig)
    print("  [OK] Ablation 4 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Ablation 5: Seed Stability
# ═══════════════════════════════════════════════════════════════════════════
def plot_ablation5():
    seeds      = [0, 42, 123, 456, 789]
    seed_labels = ["0", "42", "123", "456", "789"]
    shortcuts  = [25, 27, 23, 27, 31]
    probe_acc  = [0.890, 0.890, 0.890, 0.890, 0.890]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    x = np.arange(len(seeds))

    # Bar chart for shortcuts found
    bars = ax1.bar(x, shortcuts, 0.55, color=C_ALIGNED, alpha=0.80,
                   edgecolor='white', linewidth=0.8, label='Shortcuts Found')
    ax1.set_ylabel("Shortcuts Found", color=C_ALIGNED)
    ax1.tick_params(axis='y', labelcolor=C_ALIGNED)
    ax1.set_ylim(0, 40)

    for bar, v in zip(bars, shortcuts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(v), ha='center', fontsize=11, fontweight='bold', color=C_ALIGNED)

    # Mean ± std band
    mean_sc = np.mean(shortcuts)
    std_sc  = np.std(shortcuts)
    ax1.axhline(mean_sc, color=C_ALIGNED, linewidth=1.5, linestyle='--', alpha=0.7,
                label=f'Mean = {mean_sc:.1f}')
    ax1.axhspan(mean_sc - std_sc, mean_sc + std_sc, alpha=0.10, color=C_ALIGNED,
                label=f'±1 σ ({std_sc:.1f})')

    # Probe accuracy on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, probe_acc, 'D-', color=C_HIGHLIGHT, linewidth=2.2, markersize=9,
             label='Probe Accuracy', zorder=5)
    ax2.set_ylabel("Probe Accuracy", color=C_HIGHLIGHT)
    ax2.tick_params(axis='y', labelcolor=C_HIGHLIGHT)
    ax2.set_ylim(0.80, 0.95)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(C_HIGHLIGHT)

    # Annotate probe accuracy
    for xi, pa in zip(x, probe_acc):
        ax2.text(xi, pa + 0.003, f"{pa:.3f}", ha='center', fontsize=10,
                 fontweight='bold', color=C_HIGHLIGHT)

    ax1.set_xticks(x)
    ax1.set_xticklabels(seed_labels)
    ax1.set_xlabel("Random Seed")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

    ax1.set_title("Ablation 5: Seed Stability", fontweight='bold', pad=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    fig.text(0.5, -0.02,
             "Stable across seeds — 23 to 31 shortcuts, probe accuracy identical.\n"
             "The detection is robust.",
             ha='center', fontsize=10, style='italic', color='#555555')

    fig.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "ablation5_seed_stability.png"), pad_inches=0.3)
    plt.close(fig)
    print("  [OK] Ablation 5 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating ablation graphs...")
    plot_ablation1()
    plot_ablation2()
    plot_ablation3()
    plot_ablation4()
    plot_ablation5()
    print(f"\nAll 5 graphs saved to: {SAVE_DIR}")
