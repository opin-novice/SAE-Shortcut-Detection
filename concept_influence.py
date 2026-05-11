"""
concept_influence.py
====================
MedPatchSAE-style concept influence scoring for the Brain Tumor SAE project.

What this does:
  For each SAE latent in the top-50 shortcut features, compute delta_k:
    delta_k = baseline_accuracy - accuracy_when_latent_k_is_zeroed
  High delta_k  = this latent is causally important to the prediction.
  High delta_k AND non-pathological concept label = spurious / shortcut latent.

Inputs:
  - d:\SAE\results\candidate_features.json   (your existing top-50 output)
  - SAE activations extracted live from CLIP + SAE encoder
  - Brain Tumor MRI dataset (HuggingFace, same split as your ablation studies)

Outputs:
  - d:\SAE\results\concept_influence_scores.json
  - d:\SAE\results\flagged_spurious_latents.json
  - d:\SAE\results\concept_influence_plot.png

Usage:
  python concept_influence.py
  python concept_influence.py --bias_ratio 0.95 --top_k 50 --device cuda
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths (match your existing project layout exactly) ────────────────────────
BASE_DIR     = Path(r"d:\SAE")
RESULTS_DIR  = BASE_DIR / "results"
GRAPH_DIR    = BASE_DIR / "graph_res"
SAE_CKPT_DIR = BASE_DIR / "out"          # where your out.zip was extracted

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# ── Non-pathological concept keywords (used for spurious flagging) ────────────
# These are the artifact / background labels that indicate a shortcut.
# Extend this list if you add LLaVA-Med concept labels later.
SPURIOUS_KEYWORDS = {
    "dark_border", "border", "background", "artifact", "skull",
    "intensity", "edge", "texture", "frequency", "contrast",
    "aspect", "scanner", "noise", "coil", "watermark", "equipment",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — LOAD YOUR EXISTING ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════

def load_candidate_features(path: Path) -> list[dict]:
    """
    Load the top-50 shortcut features from candidate_features.json.

    Handles both shapes that exist in the repo:
      Shape A (original): {"top50_indices": [...], "top50_deltas": [...]}
      Shape B (expected): [{"feature_id": ..., "r_spurious": ..., ...}, ...]
    """
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        features = raw
    elif isinstance(raw, dict) and "top50_indices" in raw:
        idxs   = raw["top50_indices"]
        deltas = raw.get("top50_deltas", [0.0] * len(idxs))
        features = [
            {"feature_id": int(i), "delta": float(d),
             "r_spurious": 0.0, "r_label": 0.0, "attribute": "unknown"}
            for i, d in zip(idxs, deltas)
        ]
    else:
        raise ValueError(f"Unrecognised candidate_features.json shape in {path}")

    log.info(f"Loaded {len(features)} candidate features from {path}")
    return features


def load_concept_labels(path: Optional[Path]) -> dict[int, str]:
    """
    Load optional concept labels (from LLaVA-Med or manual annotation).
    Format: { "36070": "dark_border_artifact", "5982": "skull_boundary", ... }
    Returns empty dict if file does not exist yet — flagging will fall back
    to the 'attribute' field in candidate_features.json.
    """
    if path is None or not path.exists():
        log.warning("concept_labels.json not found — will use attribute names for flagging.")
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — EXTRACT SAE ACTIVATIONS
#  (Re-uses the same CLIP + SAE setup from your existing pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def load_clip_and_sae(sae_ckpt_dir: Path, device: str):
    """
    Load the frozen CLIP ViT-L/14 encoder and your pre-trained SAE.
    Matches your existing setup: d_in=768, d_sae=49152, hook at layer -2.
    """
    import open_clip
    from patchsae.models import SparseAutoencoder  # adjust import to your actual path

    log.info("Loading CLIP ViT-L/14 (frozen)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.visual.eval().to(device)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # Find the SAE checkpoint — expects a .pt or .pth file in sae_ckpt_dir
    ckpt_files = list(sae_ckpt_dir.glob("*.pt")) + list(sae_ckpt_dir.glob("*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt/.pth checkpoint found in {sae_ckpt_dir}")
    ckpt_path = ckpt_files[0]
    log.info(f"Loading SAE checkpoint: {ckpt_path}")

    sae = SparseAutoencoder(d_in=768, d_sae=49152)
    state = torch.load(ckpt_path, map_location=device)
    # Handle both raw state_dict and wrapped checkpoints
    sae.load_state_dict(state.get("state_dict", state))
    sae = sae.eval().to(device)
    for p in sae.parameters():
        p.requires_grad_(False)

    return clip_model, sae, preprocess


def extract_hook_activations(clip_model, images: torch.Tensor, device: str) -> torch.Tensor:
    """
    Extract patch token activations from CLIP layer -2 (your hook layer).
    Returns: (N, 768) — CLS token representation from the hook layer.
    """
    activations = {}

    def hook_fn(module, input, output):
        # output shape: (batch, seq_len, d_model) — take CLS token [0]
        activations["hook"] = output[:, 0, :].detach()

    # Register hook on the second-to-last transformer block
    target_layer = clip_model.transformer.resblocks[-2]
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = clip_model(images.to(device))

    handle.remove()
    return activations["hook"]  # (N, 768)


@torch.no_grad()
def extract_all_sae_activations(
    clip_model,
    sae,
    dataset,
    preprocess,
    device: str,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the full dataset through CLIP → SAE and return:
      activations : (N, 49152)  SAE latent activations
      labels      : (N,)        binary tumor labels
    """
    log.info(f"Extracting SAE activations for {len(dataset)} images...")
    all_acts, all_labels = [], []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Extracting"):
        batch = dataset[i : i + batch_size]

        # Handle HuggingFace dataset dict format
        if isinstance(batch, dict):
            imgs   = [preprocess(img.convert("RGB")) for img in batch["image"]]
            labels = torch.tensor(batch["label"], dtype=torch.long)
        else:
            imgs, labels = zip(*batch)
            imgs = [preprocess(img.convert("RGB")) for img in imgs]
            labels = torch.tensor(labels, dtype=torch.long)

        imgs_tensor = torch.stack(imgs).to(device)

        # CLIP hook → SAE encode
        clip_feats = extract_hook_activations(clip_model, imgs_tensor, device)
        sae_acts   = sae.encode(clip_feats)   # (batch, 49152)  ReLU sparse

        all_acts.append(sae_acts.cpu())
        all_labels.append(labels)

    activations = torch.cat(all_acts,   dim=0)  # (N, 49152)
    labels      = torch.cat(all_labels, dim=0)  # (N,)

    sparsity = (activations == 0).float().mean().item()
    log.info(f"Activations: {activations.shape}  |  Sparsity: {sparsity:.1%}")
    return activations, labels


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — LINEAR PROBE (BASELINE ACCURACY)
# ══════════════════════════════════════════════════════════════════════════════

def train_linear_probe(
    activations: torch.Tensor,  # (N, 49152)
    labels: torch.Tensor,       # (N,)
    test_size: float = 0.3,
    seed: int = 42,
) -> tuple[LogisticRegression, float, np.ndarray, np.ndarray]:
    """
    Train a logistic regression probe on SAE activations.
    Returns: (probe, baseline_accuracy, X_test, y_test)

    This is the same probe your ablation studies use — consistent baseline.
    """
    X = activations.numpy()
    y = labels.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    log.info("Training linear probe on SAE activations...")
    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    probe.fit(X_train, y_train)

    baseline_acc = accuracy_score(y_test, probe.predict(X_test))
    log.info(f"Baseline probe accuracy: {baseline_acc:.4f}")

    return probe, baseline_acc, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — COMPUTE DELTA_K  (core MedPatchSAE contribution)
# ══════════════════════════════════════════════════════════════════════════════

def compute_delta_k(
    probe: LogisticRegression,
    X_test: np.ndarray,              # (N_test, 49152)
    y_test: np.ndarray,              # (N_test,)
    baseline_acc: float,
    feature_ids: list[int],          # latent indices to ablate
    batch_ablate: bool = True,       # True = ablate all at once (fast), False = one-by-one
) -> dict[int, float]:
    """
    Compute concept influence score delta_k for each latent:
        delta_k = baseline_accuracy - accuracy_when_column_k_zeroed

    Positive delta_k: zeroing this latent hurts accuracy — it's causally important.
    Negative delta_k: zeroing this latent improves accuracy — it's a harmful shortcut.
    Near-zero delta_k: latent doesn't matter much for classification.

    Args:
        batch_ablate: if True, ablate all features simultaneously (MedPatchSAE Ablation 3 style).
                      if False, ablate each feature independently (one-by-one, slower, more precise).
    """
    delta_k_scores: dict[int, float] = {}

    log.info(f"Computing delta_k for {len(feature_ids)} features (batch={batch_ablate})...")

    if batch_ablate:
        # ── Batch ablation: zero ALL shortcut features simultaneously ──────────
        X_ablated = X_test.copy()
        X_ablated[:, feature_ids] = 0.0
        ablated_acc = accuracy_score(y_test, probe.predict(X_ablated))
        batch_delta = baseline_acc - ablated_acc

        log.info(f"Batch ablation:  baseline={baseline_acc:.4f}  "
                 f"ablated={ablated_acc:.4f}  delta={batch_delta:.4f}")

        # Distribute proportionally by each feature's weight magnitude in the probe
        probe_weights = np.abs(probe.coef_[0])  # (49152,)
        selected_weights = probe_weights[feature_ids]
        total_weight = selected_weights.sum() + 1e-9

        for fid, w in zip(feature_ids, selected_weights):
            delta_k_scores[fid] = float(batch_delta * (w / total_weight))

    else:
        # ── Per-feature ablation: zero one latent at a time ───────────────────
        for fid in tqdm(feature_ids, desc="Ablating features"):
            X_ablated = X_test.copy()
            X_ablated[:, fid] = 0.0
            ablated_acc = accuracy_score(y_test, probe.predict(X_ablated))
            delta_k_scores[fid] = float(baseline_acc - ablated_acc)

    log.info(f"delta_k range: [{min(delta_k_scores.values()):.4f}, "
             f"{max(delta_k_scores.values()):.4f}]")
    return delta_k_scores


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — FLAG SPURIOUS LATENTS
# ══════════════════════════════════════════════════════════════════════════════

def flag_spurious_latents(
    delta_k_scores: dict[int, float],
    candidate_features: list[dict],
    concept_labels: dict[int, str],
    delta_k_threshold: float = 0.0,   # flag if delta_k >= this value
    top_k: int = 20,                  # return at most this many flagged latents
) -> list[dict]:
    """
    Flag latents that are:
      (a) causally important OR harmful (delta_k != ~0), AND
      (b) associated with a non-pathological / spurious concept

    Returns a ranked list of flagged latents with full metadata.
    """
    # Build a lookup: feature_id → attribute name from your existing bias scan
    attr_lookup = {f["feature_id"]: f.get("attribute", "unknown")
                   for f in candidate_features}
    r_spur_lookup = {f["feature_id"]: f.get("r_spurious", 0.0)
                     for f in candidate_features}
    r_label_lookup = {f["feature_id"]: f.get("r_label", 0.0)
                      for f in candidate_features}

    flagged = []
    for fid, dk in delta_k_scores.items():
        # Determine concept label: prefer LLaVA-Med labels, fall back to attribute
        concept = concept_labels.get(fid, attr_lookup.get(fid, "unknown"))

        # Check if this concept is spurious (non-pathological)
        is_spurious = any(kw in concept.lower() for kw in SPURIOUS_KEYWORDS)

        flagged.append({
            "feature_id":     fid,
            "delta_k":        round(dk, 6),
            "concept_label":  concept,
            "is_spurious":    is_spurious,
            "r_spurious":     round(r_spur_lookup.get(fid, 0.0), 4),
            "r_label":        round(r_label_lookup.get(fid, 0.0), 4),
            "shortcut_score": round(
                # Combined score: high delta_k + high r_spurious + low r_label = worst shortcut
                abs(dk) * r_spur_lookup.get(fid, 0.0) / max(r_label_lookup.get(fid, 0.01), 0.01),
                4
            ),
        })

    # Sort: spurious first, then by |delta_k| descending
    flagged.sort(key=lambda x: (not x["is_spurious"], -abs(x["delta_k"])))

    log.info(f"Flagged {sum(f['is_spurious'] for f in flagged)} spurious latents "
             f"out of {len(flagged)} total")
    return flagged[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_concept_influence(
    flagged_latents: list[dict],
    baseline_acc: float,
    save_path: Path,
) -> None:
    """
    Publication-quality bar chart of delta_k scores.
    - Red bars: spurious/shortcut latents (harmful)
    - Teal bars: potentially useful latents
    - Dashed line: baseline accuracy reference
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Concept Influence Scores (Δk) — MedPatchSAE Analysis",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── Left: delta_k bar chart ───────────────────────────────────────────────
    ax = axes[0]
    fids    = [str(f["feature_id"]) for f in flagged_latents]
    deltas  = [f["delta_k"] for f in flagged_latents]
    colors  = ["#D85A30" if f["is_spurious"] else "#1D9E75" for f in flagged_latents]
    labels  = [f["concept_label"][:18] for f in flagged_latents]

    bars = ax.barh(range(len(fids)), deltas, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.7)
    ax.axvline(0, color="#888780", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(fids)))
    ax.set_yticklabels([f"F{f}  [{l}]" for f, l in zip(fids, labels)], fontsize=8)
    ax.set_xlabel("Δk  (baseline acc − ablated acc)", fontsize=10)
    ax.set_title("Per-feature influence score\n(negative = ablating helps accuracy)", fontsize=10)
    ax.invert_yaxis()

    red_patch  = mpatches.Patch(color="#D85A30", label="Spurious / shortcut")
    teal_patch = mpatches.Patch(color="#1D9E75", label="Useful feature")
    ax.legend(handles=[red_patch, teal_patch], fontsize=9, loc="lower right")

    # ── Right: shortcut_score scatter (r_spurious vs r_label, sized by |delta_k|) ──
    ax2 = axes[1]
    for f in flagged_latents:
        size  = max(abs(f["delta_k"]) * 2000, 30)
        color = "#D85A30" if f["is_spurious"] else "#1D9E75"
        ax2.scatter(f["r_spurious"], f["r_label"], s=size, c=color,
                    alpha=0.75, edgecolors="white", linewidths=0.5)
        ax2.annotate(str(f["feature_id"]), (f["r_spurious"], f["r_label"]),
                     fontsize=7, ha="center", va="bottom")

    ax2.axline((0, 0), slope=1, color="#888780", linestyle="--",
               linewidth=1.0, alpha=0.5, label="r_spurious = r_label")
    ax2.set_xlabel("r_spurious  (correlation with spurious attribute)", fontsize=10)
    ax2.set_ylabel("r_label  (correlation with true tumor label)", fontsize=10)
    ax2.set_title("Shortcut scatter\n(top-right = spurious, bottom-right = shortcuts)", fontsize=10)
    ax2.legend(fontsize=9)

    # Shade the dangerous quadrant: high spurious, low label
    ax2.axhspan(0, 0.3, alpha=0.06, color="#D85A30")
    ax2.text(0.55, 0.05, "shortcut zone", color="#D85A30",
             fontsize=8, alpha=0.7, transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved concept influence plot → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_concept_influence_pipeline(
    top_k: int = 50,
    batch_ablate: bool = False,   # False = per-feature (more informative)
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42,
) -> dict:
    """
    Full pipeline:
      1. Load your existing candidate_features.json
      2. Load CLIP + SAE (frozen)
      3. Extract SAE activations for the full dataset
      4. Train linear probe → get baseline accuracy
      5. Compute delta_k per feature
      6. Flag spurious latents
      7. Save JSON outputs + plot

    Returns the flagged latents dict for downstream use.
    """
    log.info(f"Device: {device}  |  top_k={top_k}  |  batch_ablate={batch_ablate}")

    # ── 1. Load existing artefacts ────────────────────────────────────────────
    candidate_path = RESULTS_DIR / "candidate_features.json"
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"{candidate_path} not found.\n"
            "Run your bias scan pipeline first to generate candidate_features.json."
        )

    candidate_features = load_candidate_features(candidate_path)
    concept_labels     = load_concept_labels(RESULTS_DIR / "concept_labels.json")
    feature_ids        = [f["feature_id"] for f in candidate_features[:top_k]]

    # ── 2. Load models ────────────────────────────────────────────────────────
    clip_model, sae, preprocess = load_clip_and_sae(SAE_CKPT_DIR, device)

    # ── 3. Load dataset (same HuggingFace split you already use) ─────────────
    log.info("Loading Brain Tumor MRI dataset from HuggingFace...")
    from datasets import load_dataset
    dataset = load_dataset("Hemg/Brain-Tumor-MRI-Dataset", split="train")
    # Adjust dataset name/split if yours differs — this matches the HF path
    # commonly used in PatchSAE brain tumor setups

    # ── 4. Extract SAE activations ────────────────────────────────────────────
    activations, labels = extract_all_sae_activations(
        clip_model, sae, dataset, preprocess, device
    )

    # ── 5. Train linear probe ─────────────────────────────────────────────────
    probe, baseline_acc, X_test, y_test = train_linear_probe(
        activations, labels, seed=seed
    )

    # ── 6. Compute delta_k ────────────────────────────────────────────────────
    delta_k_scores = compute_delta_k(
        probe, X_test, y_test, baseline_acc,
        feature_ids=feature_ids,
        batch_ablate=batch_ablate,
    )

    # ── 7. Flag spurious latents ──────────────────────────────────────────────
    flagged = flag_spurious_latents(
        delta_k_scores, candidate_features, concept_labels, top_k=top_k
    )

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    # Full delta_k scores
    influence_out = RESULTS_DIR / "concept_influence_scores.json"
    with open(influence_out, "w") as f:
        json.dump({
            "baseline_accuracy": round(baseline_acc, 6),
            "device": device,
            "top_k": top_k,
            "batch_ablate": batch_ablate,
            "scores": {str(k): v for k, v in delta_k_scores.items()},
        }, f, indent=2)
    log.info(f"Saved influence scores → {influence_out}")

    # Flagged spurious latents
    spurious_out = RESULTS_DIR / "flagged_spurious_latents.json"
    with open(spurious_out, "w") as f:
        json.dump({
            "baseline_accuracy": round(baseline_acc, 6),
            "n_flagged_spurious": sum(x["is_spurious"] for x in flagged),
            "latents": flagged,
        }, f, indent=2)
    log.info(f"Saved flagged latents → {spurious_out}")

    # Plot
    plot_concept_influence(
        flagged, baseline_acc,
        save_path=GRAPH_DIR / "concept_influence_plot.png"
    )

    # ── 9. Print summary ──────────────────────────────────────────────────────
    spurious_latents = [f for f in flagged if f["is_spurious"]]
    print("\n" + "═" * 60)
    print(f"  CONCEPT INFLUENCE SUMMARY")
    print("═" * 60)
    print(f"  Baseline accuracy      : {baseline_acc:.4f}")
    print(f"  Features analysed      : {len(delta_k_scores)}")
    print(f"  Spurious latents found : {len(spurious_latents)}")
    print()
    print(f"  {'Feature':<10} {'delta_k':>8}  {'Concept':<22}  {'r_spur':>7}  {'r_label':>7}")
    print(f"  {'-'*10} {'-'*8}  {'-'*22}  {'-'*7}  {'-'*7}")
    for f in spurious_latents[:10]:
        marker = "  <-- TOP SHORTCUT" if f["r_spurious"] > 0.5 else ""
        print(f"  {f['feature_id']:<10} {f['delta_k']:>+8.4f}  "
              f"{f['concept_label']:<22}  {f['r_spurious']:>7.3f}  "
              f"{f['r_label']:>7.3f}{marker}")
    print("═" * 60 + "\n")

    return {
        "baseline_accuracy":  baseline_acc,
        "delta_k_scores":     delta_k_scores,
        "flagged_latents":    flagged,
        "spurious_latents":   spurious_latents,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Compute MedPatchSAE concept influence scores.")
    p.add_argument("--top_k",        type=int,   default=50,
                   help="Number of candidate features to analyse (default: 50)")
    p.add_argument("--batch_ablate", action="store_true",
                   help="Ablate all features at once (faster). Default: per-feature.")
    p.add_argument("--device",       type=str,   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda or cpu")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_concept_influence_pipeline(
        top_k=args.top_k,
        batch_ablate=args.batch_ablate,
        device=args.device,
        seed=args.seed,
    )
