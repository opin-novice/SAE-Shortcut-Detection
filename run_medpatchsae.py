"""
run_medpatchsae.py
==================
Standalone runner for the MedPatchSAE layer.
Picks up exactly where sae_shortcut_detection_run.py left off.

Prerequisites (already on disk from the main pipeline run):
    results/sae_pooled_activations.pt  -- pooled SAE acts + labels (N, 49152)
    results/candidate_features.json    -- top-50 shortcut feature ids + deltas

What this script does:
    1. concept_influence  -- delta_k per latent (from blob, no GPU needed)
    2. group_dro          -- SAE-group assignment + Group DRO probe (from blob)
    3. Load CLIP + SAE + dataset  (GPU, ~30s)
    4. query_planner      -- BiomedCLIP semantic scoring on 3 sample images
    5. evidence_reporter  -- per-finding JSON + heatmap overlays

All outputs land in results/ and graph_res/ exactly as the wiring block
inside sae_shortcut_detection_run.py would have produced.
"""

import sys, json, logging
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"d:\SAE")
sys.path.insert(0, r"d:\SAE\patchsae")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("medpatchsae")

RESULTS_DIR = Path(r"d:\SAE\results")
GRAPH_DIR   = Path(r"d:\SAE\graph_res")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ── CLIP normalisation (same as main pipeline) ────────────────────────────────
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 0 — Load saved artefacts
# ══════════════════════════════════════════════════════════════════════════════

log.info("Loading saved artefacts...")

blob_path = RESULTS_DIR / "sae_pooled_activations.pt"
if not blob_path.exists():
    raise FileNotFoundError(
        f"{blob_path} not found.\n"
        "Run sae_shortcut_detection_run.py first (or at least until the "
        "MedPatchSAE layer fires the torch.save call)."
    )
blob = torch.load(blob_path, map_location="cpu")
pooled = blob["X"]   # (N, 49152)
y_all  = blob["y"]   # (N,)
log.info(f"Loaded pooled SAE activations: {tuple(pooled.shape)}  labels: {tuple(y_all.shape)}")

from evidence_reporter import load_candidates, load_influence
candidates   = load_candidates(RESULTS_DIR / "candidate_features.json")
shortcut_ids = [int(c["feature_id"]) for c in candidates[:50]]
log.info(f"Loaded {len(shortcut_ids)} candidate shortcut features.")

# Enrich candidates with attribute / r_spurious / r_label from shortcut_features.json
# (candidate_features.json only stores indices + deltas; the bias-scan results
#  in shortcut_features.json carry the full correlation metadata we need for overlays)
sf_path = RESULTS_DIR / "shortcut_features.json"
if sf_path.exists():
    with open(sf_path) as _f:
        _sf_raw = json.load(_f)
    # Build fid → {attribute, r_spurious, r_label} keeping highest r_spurious per feature
    shortcut_meta: dict[int, dict] = {}
    for _attr, _data in _sf_raw.items():
        for _s in _data.get("shortcuts", []):
            _fid = int(_s["feature"])
            if _fid not in shortcut_meta or _s["r_spurious"] > shortcut_meta[_fid]["r_spurious"]:
                shortcut_meta[_fid] = {
                    "attribute":  _attr,
                    "r_spurious": float(_s["r_spurious"]),
                    "r_label":    float(_s["r_label"]),
                }
    for c in candidates:
        _fid = int(c["feature_id"])
        if _fid in shortcut_meta:
            c.update(shortcut_meta[_fid])
    log.info(f"Enriched {sum(1 for c in candidates if c.get('attribute','unknown') != 'unknown')} "
             f"/ {len(candidates)} candidates with shortcut_features.json metadata.")
else:
    log.warning(f"{sf_path} not found — attribute/r_spurious will be 'unknown'/0.0 in overlays.")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — concept_influence  (CPU-only, uses pooled blob)
# ══════════════════════════════════════════════════════════════════════════════

log.info("=== Step 1: concept_influence ===")
try:
    from concept_influence import compute_delta_k
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_np, y_np = pooled.numpy(), y_all.numpy()
    Xtr, Xte, ytr, yte = train_test_split(
        X_np, y_np, test_size=0.3, stratify=y_np, random_state=42
    )
    log.info("Training linear probe for concept_influence...")
    probe        = LogisticRegression(max_iter=1000, C=1.0, random_state=42).fit(Xtr, ytr)
    baseline_acc = accuracy_score(yte, probe.predict(Xte))

    # Accuracy-based ablation is too coarse on ~57 test samples (granularity 1/57).
    # Use logit-sensitivity instead: mean |Δlogit| when feature k is zeroed.
    # For a linear probe this equals |coef[k]| × mean|X_test[:, k]|.
    # Positive → feature pushes toward tumor (causal); negative → pushes away (spurious).
    base_logits = probe.decision_function(Xte)   # (N_test,)
    coef        = probe.coef_[0]                 # (49152,)
    delta_k = {}
    for fid in shortcut_ids:
        contribution = coef[fid] * Xte[:, fid]  # signed logit contribution (N_test,)
        delta_k[fid] = float(np.mean(contribution))   # mean signed influence

    out_path = RESULTS_DIR / "concept_influence_scores.json"
    with open(out_path, "w") as f:
        json.dump({
            "baseline_accuracy": round(baseline_acc, 6),
            "scores": {str(k): round(v, 6) for k, v in delta_k.items()},
        }, f, indent=2)
    log.info(f"concept_influence: baseline={baseline_acc:.4f}  "
             f"delta_k range=[{min(delta_k.values()):.4f}, {max(delta_k.values()):.4f}]  "
             f"saved -> {out_path}")
except Exception as e:
    log.error(f"[concept_influence failed] {type(e).__name__}: {e}")
    delta_k = {}


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — group_dro  (CPU/GPU, uses pooled blob)
# ══════════════════════════════════════════════════════════════════════════════

log.info("=== Step 2: group_dro ===")
try:
    from group_dro import run as run_group_dro
    dro_out = run_group_dro(pooled, y_all, shortcut_ids, epochs=40, save=True)
    log.info(f"group_dro: worst={dro_out['final_worst_acc']:.3f}  "
             f"avg={dro_out['final_avg_acc']:.3f}  gap={dro_out['gap']:.3f}")
except Exception as e:
    log.error(f"[group_dro failed] {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 — Load CLIP + SAE + dataset  (needed for steps 4 & 5)
# ══════════════════════════════════════════════════════════════════════════════

log.info("=== Step 3: loading CLIP + SAE + dataset ===")

import einops
from transformers import CLIPModel
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from datasets import load_dataset
from tasks.utils import load_sae

# ── SAE ───────────────────────────────────────────────────────────────────────
sae, cfg = load_sae(r"d:\SAE\data\sae_weight\base\out.pt", device=DEVICE)
sae.eval()
HOOK_INDEX = cfg.block_layer
log.info(f"SAE loaded: d_in={cfg.d_in}  d_sae={cfg.d_sae}  hook={HOOK_INDEX}")

# ── CLIP ──────────────────────────────────────────────────────────────────────
_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).eval()

class _Wrapper:
    def encode_image(self, x):
        out = _clip.get_image_features(pixel_values=x)
        if not isinstance(out, torch.Tensor):
            out = out.image_embeds if hasattr(out, "image_embeds") else out[0]
        return out

model = _Wrapper()
HOOK_LAYER = _clip.vision_model.encoder.layers[HOOK_INDEX]

class Hook:
    def __init__(self, module):
        self.activation = None
        self.handle = module.register_forward_hook(self._hook)
    def _hook(self, module, inp, out):
        self.activation = (out[0] if isinstance(out, tuple) else out).detach()
    def remove(self):
        self.handle.remove()

hook = Hook(HOOK_LAYER)

def sae_encode(h):
    B, seq, d = h.shape
    _, feature_acts, _ = sae(h)
    return feature_acts.reshape(B, seq, -1)

def sae_decode(feature_acts):
    return (einops.einsum(feature_acts, sae.W_dec,
                          "... d_sae, d_sae d_in -> ... d_in") + sae.b_dec)

def denorm(x):
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std  = torch.tensor(CLIP_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)

log.info(f"CLIP + SAE loaded on {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(CLIP_MEAN, CLIP_STD),
])

raw_ds = load_dataset("Hemg/Brain-Tumor-MRI-Dataset")
splits  = list(raw_ds.keys())
if len(splits) == 1:
    split = raw_ds[splits[0]].train_test_split(test_size=0.2, seed=42)
    val_hf = split["test"]
else:
    val_hf = raw_ds.get("test") or raw_ds.get("Testing") or raw_ds[splits[1]]

class BrainDataset(Dataset):
    def __init__(self, hf_split):
        self.ds = hf_split
        class_names = hf_split.features["label"].names
        no_tumor_idx = next(
            i for i, n in enumerate(class_names)
            if any(k in n.lower() for k in ("no", "healthy", "normal"))
        )
        self.no_tumor_idx = no_tumor_idx

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = transform(row["image"].convert("RGB"))
        lbl = 0 if row["label"] == self.no_tumor_idx else 1
        return img, lbl

val_data = BrainDataset(val_hf)
log.info(f"Validation set: {len(val_data)} images")

# Use first 3 images as sample — no need to reconstruct split_b exactly
SAMPLE_IDX = [0, 1, 2]
SAMPLE_REPORTS = [
    "Right frontal lobe enhancing mass; no midline shift",
    "Left temporal lobe lesion with surrounding edema",
    "No intracranial mass; ventricles normal",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — query_planner
# ══════════════════════════════════════════════════════════════════════════════

log.info("=== Step 4: query_planner ===")
try:
    from query_planner import QueryPlanner, make_clip_patch_extractor

    extractor = make_clip_patch_extractor(model, hook, DEVICE)
    planner   = QueryPlanner(patch_feature_fn=extractor, device=DEVICE)

    query_plans = []
    for idx, report in zip(SAMPLE_IDX, SAMPLE_REPORTS):
        img, _ = val_data[idx]
        plan   = planner.run(img, report)
        query_plans.append({"image_index": idx, **plan.to_dict()})
        log.info(f"  img {idx}: conf={plan.confidence:.3f}  iters={plan.iterations}")

    out_path = RESULTS_DIR / "query_plan.json"
    with open(out_path, "w") as f:
        json.dump(query_plans, f, indent=2)
    log.info(f"query_planner: {len(query_plans)} plans saved -> {out_path}")
except Exception as e:
    log.error(f"[query_planner failed] {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    query_plans = []


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5 — evidence_reporter  (per-finding JSON + heatmap overlays)
# ══════════════════════════════════════════════════════════════════════════════

log.info("=== Step 5: evidence_reporter ===")
try:
    from evidence_reporter import build_finding, render_overlay, write_evidence_report

    influence = load_influence(RESULTS_DIR / "concept_influence_scores.json")
    top5      = shortcut_ids[:5]
    findings  = []

    for i, idx in enumerate(SAMPLE_IDX):
        img, _ = val_data[idx]
        # Run CLIP + SAE to get per-patch activations
        with torch.no_grad():
            _ = model.encode_image(img.unsqueeze(0).to(DEVICE))
            z = sae_encode(hook.activation)[0, 1:, :]   # (196, 49152)

        # Attach query plan for this image (both for overlay patches and finding JSON)
        qp_plan    = query_plans[i] if i < len(query_plans) else None
        qp_patches = [p["bbox_224"] for p in (qp_plan or {}).get("top_patches", [])]

        for fid in top5:
            grid     = z[:, fid].reshape(14, 14).cpu().numpy()
            cand_row = next(c for c in candidates if int(c["feature_id"]) == fid)
            finding  = build_finding(
                image_id        = str(idx),
                feature_id      = fid,
                activation_grid = grid,
                candidate_row   = cand_row,
                delta_k         = float(influence.get(fid, 0.0)),
                query_plan      = qp_plan,
            )
            findings.append(finding)

            img_chw  = denorm(img).numpy()
            ov_path  = GRAPH_DIR / f"overlay_img{idx}_F{fid}.png"
            render_overlay(img_chw, grid, finding,
                           query_patches=qp_patches, save_path=ov_path)

    report_path = write_evidence_report(findings, RESULTS_DIR / "evidence_report.json")
    log.info(f"evidence_reporter: {len(findings)} findings saved -> {report_path}")

except Exception as e:
    log.error(f"[evidence_reporter failed] {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()


hook.remove()


# ══════════════════════════════════════════════════════════════════════════════
#  Step 6 — Regenerate concept_influence_plot with real delta_k values
# ══════════════════════════════════════════════════════════════════════════════

log.info("=== Step 6: concept_influence_plot ===")
try:
    from concept_influence import plot_concept_influence

    # Reload delta_k (logit-sensitivity scores) and build flagged_latents
    with open(RESULTS_DIR / "concept_influence_scores.json") as _f:
        _ci = json.load(_f)
    _scores = {int(k): float(v) for k, v in _ci["scores"].items()}
    _baseline_acc = float(_ci["baseline_accuracy"])

    _flagged = []
    for c in candidates[:50]:
        fid  = int(c["feature_id"])
        dk   = _scores.get(fid, 0.0)
        attr = c.get("attribute", "unknown")
        is_spur = any(kw in attr.lower() for kw in {
            "dark_border","border","background","artifact","skull",
            "intensity","edge","texture","frequency","contrast",
            "aspect","scanner","noise","coil","watermark","equipment",
        })
        _flagged.append({
            "feature_id":    fid,
            "delta_k":       round(dk, 6),
            "concept_label": attr,
            "is_spurious":   is_spur,
            "r_spurious":    round(c.get("r_spurious", 0.0), 4),
            "r_label":       round(c.get("r_label",    0.0), 4),
        })
    # Sort: spurious first, then by |delta_k| descending
    _flagged.sort(key=lambda x: (not x["is_spurious"], -abs(x["delta_k"])))

    plot_path = GRAPH_DIR / "concept_influence_plot.png"
    plot_concept_influence(_flagged[:20], _baseline_acc, save_path=plot_path)
    log.info(f"concept_influence_plot saved -> {plot_path}")
except Exception as e:
    log.error(f"[concept_influence_plot failed] {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
#  Step 7 — Summary table  (top-10 findings by shortcut score)
# ══════════════════════════════════════════════════════════════════════════════

log.info("=== Step 7: summary table ===")
try:
    with open(RESULTS_DIR / "evidence_report.json") as _f:
        _report = json.load(_f)

    # Deduplicate by feature: keep worst-case (highest r_spurious) per latent
    _seen: dict[int, dict] = {}
    for fn in _report["findings"]:
        fid = fn["sae_latent_id"]
        if fid not in _seen or fn["r_spurious"] > _seen[fid]["r_spurious"]:
            _seen[fid] = fn

    # Rank by |r_spurious| * |delta_k| (shortcut severity), fallback to r_spurious
    def _shortcut_score(fn):
        return abs(fn["r_spurious"]) * max(abs(fn["delta_k"]), 1e-9)

    _ranked = sorted(_seen.values(), key=_shortcut_score, reverse=True)[:10]

    _hdr = f"{'Feature':>9}  {'Concept':<18}  {'r_spur':>6}  {'r_lbl':>6}  {'delta_k':>9}  {'Spur?':>5}"
    _sep = "-" * len(_hdr)
    print("\n" + "=" * len(_hdr))
    print("  TOP SHORTCUT FINDINGS  (ranked by r_spurious x |delta_k|)")
    print("=" * len(_hdr))
    print(_hdr)
    print(_sep)
    for fn in _ranked:
        flag = "YES" if fn["is_spurious"] else "no"
        print(f"  F{fn['sae_latent_id']:<7}  {fn['concept_label']:<18}  "
              f"{fn['r_spurious']:>6.3f}  {fn['r_label']:>6.3f}  "
              f"{fn['delta_k']:>+9.5f}  {flag:>5}")
    print("=" * len(_hdr))
    print(f"  Baseline probe acc: {_ci['baseline_accuracy']:.4f}  |  "
          f"Total findings: {_report['n_findings']}  |  "
          f"Spurious: {_report['n_spurious']}/{_report['n_findings']}\n")
except Exception as e:
    log.error(f"[summary table failed] {type(e).__name__}: {e}")


log.info("Done. MedPatchSAE layer complete.")
log.info("Outputs:")
for p in sorted((RESULTS_DIR / "..").glob("results/*.json")) + \
         sorted((RESULTS_DIR / "..").glob("graph_res/overlay_*.png")):
    try:
        log.info(f"  {p.relative_to(Path('d:/SAE'))}  ({p.stat().st_size:,} bytes)")
    except Exception:
        pass
