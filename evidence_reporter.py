"""
evidence_reporter.py
====================
Build structured JSON evidence reports + upgraded heatmap overlays for each
finding produced by the MedPatchSAE pipeline.

A "finding" is one (image, SAE_latent, concept_label) tuple.  For each
finding we record:
    {
        "image_id":          str | int,
        "concept_label":     str,       # e.g. "dark_border_artifact"
        "sae_latent_id":     int,       # e.g. 36070
        "activation_score":  float,     # max activation over patches
        "patch_coords":      [[r,c], ...],   # patches above threshold
        "bbox_224":          [x0,y0,x1,y1], # bbox over those patches
        "spurious_attribute":str,       # "dark_border" | "edge_density" | ...
        "r_spurious":        float,
        "r_label":           float,
        "delta_k":           float,
        "query_plan":        {...}      # optional, from query_planner.py
    }

CLI:
    python evidence_reporter.py \
        --influence results/concept_influence_scores.json \
        --candidates results/candidate_features.json \
        --out results/evidence_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("evidence")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  [er] %(message)s",
                    datefmt="%H:%M:%S")

RESULTS_DIR = Path(r"d:\SAE\results")
GRAPH_DIR   = Path(r"d:\SAE\graph_res")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

PATCH_GRID = 14
IMG_SIZE   = 224
CELL       = IMG_SIZE // PATCH_GRID

# Same vocabulary used in concept_influence.py for "spurious" gating
SPURIOUS_KEYWORDS = {
    "dark_border", "border", "background", "artifact", "skull",
    "intensity", "edge", "texture", "frequency", "contrast",
    "aspect", "scanner", "noise", "coil", "watermark", "equipment",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Finding:
    image_id:           str
    concept_label:      str
    sae_latent_id:      int
    activation_score:   float
    patch_coords:       list[list[int]]
    bbox_224:           list[int]
    spurious_attribute: str
    r_spurious:         float = 0.0
    r_label:            float = 0.0
    delta_k:            float = 0.0
    is_spurious:        bool  = False
    query_plan:         Optional[dict] = None
    extra:              dict  = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers — robust loaders that cope with BOTH candidate_features.json shapes
# ══════════════════════════════════════════════════════════════════════════════

def load_candidates(path: Path) -> list[dict]:
    """
    Returns a canonical list of dicts with at least feature_id.

    The repo currently ships TWO incompatible shapes for candidate_features.json:
      1) {"top50_indices": [..], "top50_deltas": [..]}      (actual format)
      2) [{"feature_id": ..., "r_spurious": ..., ...}, ...] (what
         concept_influence.py expects)

    This loader normalises both into shape (2) so downstream code is simple.
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "top50_indices" in raw:
        idxs   = raw["top50_indices"]
        deltas = raw.get("top50_deltas", [0.0] * len(idxs))
        return [
            {"feature_id": int(i), "delta": float(d),
             "r_spurious": 0.0, "r_label": 0.0, "attribute": "unknown"}
            for i, d in zip(idxs, deltas)
        ]
    raise ValueError(f"Unrecognised candidate_features.json shape in {path}")


def load_influence(path: Path) -> dict[int, float]:
    """concept_influence_scores.json → {feature_id: delta_k}."""
    if not path.exists():
        log.warning(f"{path} not found — delta_k will be 0.0 for all findings.")
        return {}
    with open(path) as f:
        raw = json.load(f)
    scores = raw.get("scores", raw)
    return {int(k): float(v) for k, v in scores.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  Section 1 — Build per-image, per-latent findings
# ══════════════════════════════════════════════════════════════════════════════

def patches_above_threshold(activation_grid: np.ndarray,
                            threshold_pct: float = 0.6) -> list[tuple[int, int]]:
    """
    Returns list of (row, col) patches whose activation >= threshold * max.
    Operates on a (14, 14) numpy array.
    """
    if activation_grid.max() <= 0:
        return []
    thr = threshold_pct * activation_grid.max()
    rs, cs = np.where(activation_grid >= thr)
    return list(zip(rs.tolist(), cs.tolist()))


def coords_to_bbox(coords: list[tuple[int, int]]) -> list[int]:
    """Tight bbox in 224-pixel coords (x0, y0, x1, y1) over a set of patches."""
    if not coords:
        return [0, 0, 0, 0]
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return [min(cs) * CELL, min(rs) * CELL,
            (max(cs) + 1) * CELL, (max(rs) + 1) * CELL]


def concept_label_for(feature_id: int,
                      concept_labels: dict[int, str],
                      attribute: str) -> tuple[str, bool]:
    """Resolve a human-readable label and flag whether it's spurious."""
    label = concept_labels.get(feature_id, attribute or "unknown")
    is_spurious = any(kw in label.lower() for kw in SPURIOUS_KEYWORDS)
    return label, is_spurious


def build_finding(
    image_id:        str,
    feature_id:      int,
    activation_grid: np.ndarray,         # (14, 14)
    candidate_row:   dict,
    delta_k:         float = 0.0,
    concept_labels:  Optional[dict[int, str]] = None,
    query_plan:      Optional[dict]            = None,
    threshold_pct:   float = 0.6,
) -> Finding:
    concept_labels = concept_labels or {}
    coords = patches_above_threshold(activation_grid, threshold_pct)
    bbox   = coords_to_bbox(coords)
    attribute = candidate_row.get("attribute", "unknown")
    label, is_spurious = concept_label_for(feature_id, concept_labels, attribute)

    return Finding(
        image_id           = str(image_id),
        concept_label      = label,
        sae_latent_id      = int(feature_id),
        activation_score   = float(activation_grid.max()),
        patch_coords       = [[r, c] for r, c in coords],
        bbox_224           = bbox,
        spurious_attribute = attribute,
        r_spurious         = float(candidate_row.get("r_spurious", 0.0)),
        r_label            = float(candidate_row.get("r_label", 0.0)),
        delta_k            = float(delta_k),
        is_spurious        = is_spurious,
        query_plan         = query_plan,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Section 2 — Heatmap overlay (matplotlib only — no extra deps)
# ══════════════════════════════════════════════════════════════════════════════

def render_overlay(
    image_chw:        np.ndarray,        # (3, 224, 224) DENORMALISED in [0,1]
    activation_grid:  np.ndarray,        # (14, 14)
    finding:          Finding,
    query_patches:    Optional[list[list[int]]] = None,
    save_path:        Optional[Path]            = None,
):
    """
    Renders MRI image + SAE activation heatmap + finding bbox (red) + optional
    query-planner top-K patches (cyan) + concept label text at the top.

    Saves to save_path if given.  Returns the matplotlib Figure either way.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import torch.nn.functional as F
    import torch

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    img_hwc = np.transpose(np.clip(image_chw, 0, 1), (1, 2, 0))
    ax.imshow(img_hwc)

    # Upsample 14×14 → 224×224 with bilinear interp via torch
    heat = torch.from_numpy(activation_grid).float().unsqueeze(0).unsqueeze(0)
    heat_up = F.interpolate(heat, size=(IMG_SIZE, IMG_SIZE),
                            mode="bilinear", align_corners=False)[0, 0].numpy()
    if heat_up.max() > heat_up.min():
        heat_up = (heat_up - heat_up.min()) / (heat_up.max() - heat_up.min())
    ax.imshow(heat_up, cmap="hot", alpha=0.45)

    # Red bbox for the SAE finding
    x0, y0, x1, y1 = finding.bbox_224
    if (x1 - x0) > 0 and (y1 - y0) > 0:
        rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  linewidth=2.0, edgecolor="#FF3333",
                                  facecolor="none")
        ax.add_patch(rect)

    # Cyan boxes for query-planner top patches (if provided)
    if query_patches:
        for qb in query_patches:
            qx0, qy0, qx1, qy1 = qb
            ax.add_patch(mpatches.Rectangle(
                (qx0, qy0), qx1 - qx0, qy1 - qy0,
                linewidth=1.0, edgecolor="#22D3EE", facecolor="none",
                linestyle="--"))

    label = (f"F{finding.sae_latent_id}  •  {finding.concept_label}"
             f"  •  act={finding.activation_score:.2f}"
             f"  Δk={finding.delta_k:+.3f}")
    sub   = (f"r_spur={finding.r_spurious:.2f}  r_label={finding.r_label:.2f}"
             f"  spurious={finding.is_spurious}")
    ax.set_title(label, fontsize=9, loc="left")
    ax.text(2, 218, sub, color="white", fontsize=7,
            bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))
    ax.axis("off")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"saved overlay → {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Section 3 — Aggregate writer
# ══════════════════════════════════════════════════════════════════════════════

def write_evidence_report(findings: list[Finding],
                          out_path: Path = RESULTS_DIR / "evidence_report.json"
                          ) -> Path:
    payload = {
        "n_findings":   len(findings),
        "n_spurious":   sum(1 for f in findings if f.is_spurious),
        "n_unique_latents": len({f.sae_latent_id for f in findings}),
        "findings":     [asdict(f) for f in findings],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"wrote {len(findings)} findings → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  Section 4 — CLI: dry-run that builds findings from existing artefacts
# ══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default=str(RESULTS_DIR / "candidate_features.json"))
    p.add_argument("--influence",  default=str(RESULTS_DIR / "concept_influence_scores.json"))
    p.add_argument("--concepts",   default=str(RESULTS_DIR / "concept_labels.json"),
                   help="Optional concept_labels.json")
    p.add_argument("--out",        default=str(RESULTS_DIR / "evidence_report.json"))
    p.add_argument("--top_k",      type=int, default=20)
    args = p.parse_args()

    candidates = load_candidates(Path(args.candidates))
    influence  = load_influence(Path(args.influence))
    concept_labels: dict[int, str] = {}
    cpath = Path(args.concepts)
    if cpath.exists():
        with open(cpath) as f:
            concept_labels = {int(k): v for k, v in json.load(f).items()}

    # Without real activations we synthesise an empty grid — this CLI run is
    # only useful as a JSON-schema preview.  For real overlays call
    # write_evidence_report() and render_overlay() from the main pipeline.
    findings = []
    for row in candidates[:args.top_k]:
        fid = int(row["feature_id"])
        empty_grid = np.zeros((PATCH_GRID, PATCH_GRID), dtype=np.float32)
        findings.append(build_finding(
            image_id        = "schema_preview",
            feature_id      = fid,
            activation_grid = empty_grid,
            candidate_row   = row,
            delta_k         = influence.get(fid, 0.0),
            concept_labels  = concept_labels,
        ))
    write_evidence_report(findings, Path(args.out))
    print(f"Schema preview written to {args.out}  "
          f"({len(findings)} stub findings).  "
          "Call render_overlay() from the main pipeline for real images.")


if __name__ == "__main__":
    _cli()
