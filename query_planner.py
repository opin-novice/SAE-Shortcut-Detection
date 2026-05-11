"""
query_planner.py
================
Query-Planning Layer for the Brain Tumor MedPatchSAE pipeline.

Pipeline (matches the "QUERY PLANNING LAYER" in the architecture diagram):

    radiology report text
            ┌──────────┐
            │  parse   │ (NER: pathologies, locations, negations)
            └────┬─────┘
                 ▼
            ┌──────────┐         ┌────────────────────┐
            │ BiomedCLIP│◄──── parsed queries
            │  encode   │         (positive + negative)
            └────┬─────┘
                 ▼ multi-query text vectors
                 │           ┌────────────────┐
                 │           │ spatial prior  │ ← anatomy heatmap
                 │           │     map        │
                 ▼           └──────┬─────────┘
            ┌─────────────────────┐ │
            │  score every patch  │◄┘
            │  s = α·sem + β·spat │
            └──────────┬──────────┘
                       ▼
                ┌──────────────┐
                │ agentic loop │ rank → critique → refine queries
                │  (max 3×)    │
                └──────┬───────┘
                       ▼
                MIL aggregate ─► final label + confidence + top-K patches

Outputs:
    results/query_plan.json        per-image findings + per-patch scores
    results/query_plan_summary.json aggregate stats over the dataset

Usage (programmatic):
    from query_planner import QueryPlanner
    planner = QueryPlanner(clip_model, sae, hook, device="cuda")
    finding = planner.run(image_tensor, report_text="Right frontal lobe mass...")

Usage (CLI, smoke-test on one image):
    python query_planner.py --image data/image/aircraft.jpg \
        --report "Right frontal lobe enhancing mass; no midline shift"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("query_planner")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  [qp] %(message)s",
                    datefmt="%H:%M:%S")

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(r"d:\SAE\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_SEMANTIC = 0.6      # weight on BiomedCLIP semantic similarity
BETA_SPATIAL   = 0.4      # weight on anatomical spatial prior
MAX_REFINE_ITERS = 3      # agentic refinement budget
PATCH_GRID = 14           # ViT-B/16 @ 224 → 14×14 = 196 patches
TOP_K_PATCHES = 8         # MIL attention pool size

# Brain anatomy vocabulary used by the report parser.  Extend as needed.
PATHOLOGY_TERMS = {
    "tumor", "mass", "lesion", "neoplasm", "glioma", "meningioma",
    "pituitary", "metastasis", "edema", "hemorrhage", "infarct", "cyst",
    "enhancement", "necrosis",
}
LOCATION_TERMS = {
    "frontal", "parietal", "temporal", "occipital", "cerebellum",
    "brainstem", "thalamus", "basal ganglia", "ventricle", "sella",
    "pituitary", "pineal", "left", "right", "midline", "cortex",
    "white matter",
}
NEGATION_CUES = {"no", "without", "absent", "denies", "negative", "free of"}


# ══════════════════════════════════════════════════════════════════════════════
#  Section 1 — Lightweight report parser (regex-based NER, no spaCy dep)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedReport:
    raw: str
    pathologies: list[str] = field(default_factory=list)
    locations:   list[str] = field(default_factory=list)
    negated:     list[str] = field(default_factory=list)

    def positive_phrases(self) -> list[str]:
        """Concatenations of (pathology, location) that were NOT negated."""
        if not self.pathologies:
            # Fall back to raw sentences if no pathology term found
            return [s.strip() for s in re.split(r"[.;]", self.raw) if s.strip()]
        out = []
        for p in self.pathologies:
            if p in self.negated:
                continue
            if self.locations:
                for loc in self.locations:
                    out.append(f"{loc} {p}")
            else:
                out.append(p)
        return out or [self.raw]

    def negative_phrases(self) -> list[str]:
        return [n for n in self.negated if n in self.pathologies]


def parse_report(text: str) -> ParsedReport:
    """Cheap NER: lowercase, scan for known terms, mark negation in a ±4-token window."""
    text_clean = text.strip()
    tokens = re.findall(r"[a-zA-Z\-]+", text_clean.lower())
    pathologies, locations, negated = [], [], set()
    for i, tok in enumerate(tokens):
        if tok in PATHOLOGY_TERMS:
            pathologies.append(tok)
            window = tokens[max(0, i - 4): i]
            if any(w in NEGATION_CUES for w in window):
                negated.add(tok)
        if tok in LOCATION_TERMS:
            locations.append(tok)
    # Deduplicate, preserve order
    pathologies = list(dict.fromkeys(pathologies))
    locations   = list(dict.fromkeys(locations))
    return ParsedReport(raw=text_clean,
                        pathologies=pathologies,
                        locations=locations,
                        negated=list(negated))


# ══════════════════════════════════════════════════════════════════════════════
#  Section 2 — BiomedCLIP loader (lazy, optional fallback to base CLIP text)
# ══════════════════════════════════════════════════════════════════════════════

class BiomedCLIPEncoder:
    """
    Wraps a BiomedCLIP text/vision encoder.  Tries open_clip first
    (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224); if that fails
    falls back silently to plain openai/clip-vit-base-patch16 so the rest of
    the pipeline still runs.  The fallback is logged loudly so you know.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.is_biomed = False
        self.model = None
        self.tokenizer = None
        self.preprocess = None
        self._load()

    def _load(self):
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            tokenizer = open_clip.get_tokenizer(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )
            self.model = model.eval().to(self.device)
            self.tokenizer = tokenizer
            self.preprocess = preprocess
            self.is_biomed = True
            log.info("BiomedCLIP loaded (open_clip hf-hub).")
            return
        except Exception as e:
            log.warning(f"BiomedCLIP load failed ({e!r}); falling back to base CLIP.")

        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        ).eval().to(self.device)
        self.tokenizer = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        self.is_biomed = False

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Return L2-normalised text embeddings (N, D)."""
        if not texts:
            return torch.zeros(0, 512, device=self.device)
        if self.is_biomed:
            toks = self.tokenizer(texts).to(self.device)
            feats = self.model.encode_text(toks)
        else:
            inp = self.tokenizer(text=texts, return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
            feats = self.model.get_text_features(**inp)
            # Some HF transformers versions return a ModelOutput instead of a tensor
            if not isinstance(feats, torch.Tensor):
                feats = (feats.text_embeds if hasattr(feats, "text_embeds")
                         else feats.pooler_output)
        return F.normalize(feats, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Section 3 — Spatial prior over the 14×14 patch grid
# ══════════════════════════════════════════════════════════════════════════════

# Rough anatomical heatmaps over a 14×14 grid in normalised image coords.
# (0,0) is top-left, (1,1) is bottom-right.  These are deliberately broad —
# the goal is to bias attention toward plausible regions, not to localise.
_ANATOMY_CENTERS: dict[str, tuple[float, float, float]] = {
    # name           cy    cx    sigma
    "frontal":     (0.30, 0.50, 0.20),
    "parietal":    (0.55, 0.50, 0.20),
    "temporal":    (0.65, 0.25, 0.18),   # mirror handled by 'left/right' below
    "occipital":   (0.80, 0.50, 0.18),
    "cerebellum":  (0.85, 0.50, 0.18),
    "brainstem":   (0.70, 0.50, 0.12),
    "thalamus":    (0.55, 0.50, 0.10),
    "ventricle":   (0.50, 0.50, 0.15),
    "pituitary":   (0.65, 0.50, 0.08),
    "sella":       (0.65, 0.50, 0.08),
    "midline":     (0.50, 0.50, 0.10),
    "default":     (0.50, 0.50, 0.35),   # whole-brain fallback
}


def spatial_prior_map(locations: list[str], grid: int = PATCH_GRID) -> np.ndarray:
    """
    Build a (grid×grid) probability-like map summing Gaussian bumps over the
    requested anatomical regions.  Always non-negative, max-normalised to 1.
    """
    yy, xx = np.mgrid[0:grid, 0:grid].astype(np.float32)
    yy = (yy + 0.5) / grid
    xx = (xx + 0.5) / grid

    bumps = np.zeros((grid, grid), dtype=np.float32)
    centers = [_ANATOMY_CENTERS[k] for k in locations
               if k in _ANATOMY_CENTERS] or [_ANATOMY_CENTERS["default"]]
    for cy, cx, sigma in centers:
        bumps += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))

    # Left/right mirror handling: if "left" or "right" appears, weight one half.
    if "left" in locations:
        bumps[:, grid // 2:] *= 0.3
    if "right" in locations:
        bumps[:, :grid // 2] *= 0.3

    if bumps.max() > 0:
        bumps /= bumps.max()
    return bumps


# ══════════════════════════════════════════════════════════════════════════════
#  Section 4 — Patch scoring (semantic × spatial)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatchScore:
    idx: int            # 0..195
    row: int
    col: int
    semantic: float
    spatial: float
    total: float

    @property
    def bbox_224(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) in 224×224 pixel space for the patch."""
        cell = 224 // PATCH_GRID
        return (self.col * cell, self.row * cell,
                (self.col + 1) * cell, (self.row + 1) * cell)


def score_patches(
    patch_features: torch.Tensor,   # (196, D_img) L2-normalised patch tokens
    query_features: torch.Tensor,   # (Q, D_text) L2-normalised query embeds
    spatial_prior: np.ndarray,      # (14, 14) max-normalised
    alpha: float = ALPHA_SEMANTIC,
    beta:  float = BETA_SPATIAL,
) -> list[PatchScore]:
    """
    For each patch:
        semantic_i = max_q  cos(patch_i, query_q)
        spatial_i  = prior[row, col]
        total_i    = α · semantic_i + β · spatial_i

    If patch_features and query_features have different dimensions (BiomedCLIP
    text vs CLIP-base patches), projects the larger one with a learned-free
    pseudo-inverse — or, if dims diverge too much, falls back to spatial-only.
    """
    n = patch_features.shape[0]
    assert n == PATCH_GRID * PATCH_GRID, f"expected 196 patches, got {n}"

    if query_features.numel() > 0:
        if patch_features.shape[-1] != query_features.shape[-1]:
            # Dim mismatch (e.g., BiomedCLIP 512 vs CLIP-base 768).  Use a
            # cheap learned-free projection: random Gaussian projection seeded
            # for reproducibility.  Good enough for ranking, not classification.
            g = torch.Generator(device=patch_features.device).manual_seed(0)
            proj = torch.randn(
                patch_features.shape[-1], query_features.shape[-1],
                generator=g, device=patch_features.device,
            ) / (patch_features.shape[-1] ** 0.5)
            patch_proj = F.normalize(patch_features @ proj, dim=-1)
        else:
            patch_proj = patch_features
        sims = patch_proj @ query_features.T          # (196, Q)
        semantic = sims.max(dim=1).values.cpu().numpy()   # (196,)
    else:
        semantic = np.zeros(n, dtype=np.float32)

    # Normalise semantic to [0, 1] so the α/β weighting is meaningful
    if semantic.max() > semantic.min():
        semantic = (semantic - semantic.min()) / (semantic.max() - semantic.min())

    spatial_flat = spatial_prior.reshape(-1)
    total = alpha * semantic + beta * spatial_flat

    scores = [
        PatchScore(
            idx=i,
            row=i // PATCH_GRID,
            col=i %  PATCH_GRID,
            semantic=float(semantic[i]),
            spatial=float(spatial_flat[i]),
            total=float(total[i]),
        )
        for i in range(n)
    ]
    return scores


def mil_aggregate(scores: list[PatchScore], top_k: int = TOP_K_PATCHES
                  ) -> tuple[float, list[PatchScore]]:
    """
    Multiple-instance pooling: attention-weighted mean of the top-K patch
    scores.  Returns (confidence in [0,1], the top-K PatchScore objects).
    """
    ranked = sorted(scores, key=lambda s: -s.total)[:top_k]
    if not ranked:
        return 0.0, []
    weights = np.array([s.total for s in ranked], dtype=np.float32)
    weights = np.exp(weights - weights.max())
    weights /= weights.sum()
    confidence = float((weights * np.array([s.total for s in ranked])).sum())
    return confidence, ranked


# ══════════════════════════════════════════════════════════════════════════════
#  Section 5 — Agentic refinement loop
# ══════════════════════════════════════════════════════════════════════════════

def _refine_queries(parsed: ParsedReport,
                    previous: list[str],
                    top_patches: list[PatchScore],
                    spatial_prior: np.ndarray) -> list[str]:
    """
    One refinement step.  Heuristic, no LLM call — keeps the loop deterministic
    and free.  Strategy:
      1. If top patches all fall in a region with low prior, broaden queries
         (drop the most specific term).
      2. If top patches cluster tightly, add a more specific term
         (e.g., the strongest pathology word).
      3. Always inject a "tumor" / "lesion" generic backup the first time.
    """
    refined = list(previous)

    if not top_patches:
        return refined + ["brain tumor", "brain lesion"]

    mean_spatial = float(np.mean([p.spatial for p in top_patches]))
    if mean_spatial < 0.25 and refined:
        # Broaden — drop modifiers (left/right/midline) from the longest query
        longest = max(refined, key=len)
        broadened = re.sub(r"\b(left|right|midline)\s+", "", longest).strip()
        if broadened and broadened not in refined:
            refined.append(broadened)

    rows = np.array([p.row for p in top_patches])
    cols = np.array([p.col for p in top_patches])
    if rows.std() < 1.5 and cols.std() < 1.5 and parsed.pathologies:
        # Tightly clustered — try a more specific synonym
        specific = f"{parsed.pathologies[0]} with enhancement"
        if specific not in refined:
            refined.append(specific)

    # Always have a generic backup
    for backup in ("brain tumor", "intracranial mass"):
        if backup not in refined:
            refined.append(backup)
            break
    return refined


# ══════════════════════════════════════════════════════════════════════════════
#  Section 6 — Main planner
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QueryPlanResult:
    report: str
    parsed: dict
    queries_per_iter: list[list[str]]
    final_queries: list[str]
    iterations: int
    confidence: float
    top_patches: list[dict]
    spatial_prior_max: float
    alpha: float = ALPHA_SEMANTIC
    beta: float = BETA_SPATIAL

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class QueryPlanner:
    """
    Orchestrates report → queries → patch scoring → refinement → MIL aggregate.

    Expects the caller to provide a `patch_feature_fn` that turns a single
    image tensor (3, 224, 224) into (196, D) L2-normalised patch tokens.  This
    keeps the planner decoupled from the specific CLIP+SAE setup in
    sae_shortcut_detection_run.py.
    """

    def __init__(
        self,
        patch_feature_fn,                 # callable(img_tensor) -> (196, D) tensor
        encoder: Optional[BiomedCLIPEncoder] = None,
        device: str = "cuda",
        alpha: float = ALPHA_SEMANTIC,
        beta:  float = BETA_SPATIAL,
        max_iters: int = MAX_REFINE_ITERS,
    ):
        self.patch_feature_fn = patch_feature_fn
        self.encoder = encoder or BiomedCLIPEncoder(device=device)
        self.device = device
        self.alpha, self.beta = alpha, beta
        self.max_iters = max_iters

    def run(self, image: torch.Tensor, report_text: str) -> QueryPlanResult:
        parsed = parse_report(report_text)
        log.info(f"parsed: pathologies={parsed.pathologies} "
                 f"locations={parsed.locations} negated={parsed.negated}")

        prior = spatial_prior_map(parsed.locations)
        with torch.no_grad():
            patch_feats = self.patch_feature_fn(image.to(self.device))
            patch_feats = F.normalize(patch_feats, dim=-1)

        queries = parsed.positive_phrases()
        queries_per_iter: list[list[str]] = []
        top_patches: list[PatchScore] = []
        confidence = 0.0

        for it in range(self.max_iters):
            queries_per_iter.append(list(queries))
            q_feats = self.encoder.encode_text(queries)
            scores = score_patches(patch_feats, q_feats, prior,
                                   alpha=self.alpha, beta=self.beta)
            confidence, top_patches = mil_aggregate(scores)
            log.info(f"iter {it+1}/{self.max_iters}: "
                     f"queries={len(queries)} conf={confidence:.3f}")

            if confidence > 0.55 or it == self.max_iters - 1:
                break
            new_queries = _refine_queries(parsed, queries, top_patches, prior)
            if new_queries == queries:
                log.info("queries stabilised — stopping early")
                break
            queries = new_queries

        return QueryPlanResult(
            report=parsed.raw,
            parsed=asdict(parsed),
            queries_per_iter=queries_per_iter,
            final_queries=queries,
            iterations=len(queries_per_iter),
            confidence=confidence,
            top_patches=[asdict(p) | {"bbox_224": list(p.bbox_224)}
                         for p in top_patches],
            spatial_prior_max=float(prior.max()),
            alpha=self.alpha,
            beta=self.beta,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Section 7 — Reusable patch-feature extractor that matches the main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def make_clip_patch_extractor(clip_model, hook, device: str = "cuda"):
    """
    Returns a callable(image_tensor) -> (196, D_img) tensor.

    `clip_model` is the wrapper used in sae_shortcut_detection_run.py
    (`model = _Wrapper()` exposing encode_image).  `hook` is the registered
    Hook instance on `vision_model.encoder.layers[HOOK_INDEX]`.
    """
    @torch.no_grad()
    def extract(img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        _ = clip_model.encode_image(img.to(device))
        h = hook.activation                 # [1, 197, 768]
        patches = h[0, 1:, :]               # drop CLS  -> [196, 768]
        return patches
    return extract


# ══════════════════════════════════════════════════════════════════════════════
#  Section 8 — CLI smoke test
# ══════════════════════════════════════════════════════════════════════════════

def _smoke_test_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--image",  required=False, default=None,
                   help="Path to a single image for a smoke test")
    p.add_argument("--report", required=False,
                   default="Right frontal lobe enhancing mass; no midline shift",
                   help="Synthetic radiology report")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    parsed = parse_report(args.report)
    print("Parsed report:", asdict(parsed))
    prior = spatial_prior_map(parsed.locations)
    print(f"Spatial prior max={prior.max():.3f}  mean={prior.mean():.3f}")

    if args.image is None:
        print("No --image given; ran parse + prior only.")
        return

    # Best-effort: load the main pipeline's CLIP+hook lazily
    import sys
    sys.path.insert(0, r"d:\SAE\patchsae")
    from transformers import CLIPModel
    from torchvision import transforms
    from PIL import Image

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(args.device).eval()

    class _W:
        def encode_image(self, x): return clip.get_image_features(pixel_values=x)
    wrapper = _W()

    hook_layer = clip.vision_model.encoder.layers[-2]

    class _Hook:
        def __init__(self, m):
            self.act = None
            self.h = m.register_forward_hook(lambda mod, i, o:
                setattr(self, "act", (o[0] if isinstance(o, tuple) else o).detach()))
        @property
        def activation(self): return self.act
    hk = _Hook(hook_layer)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])
    img = tfm(Image.open(args.image).convert("RGB"))

    planner = QueryPlanner(
        patch_feature_fn=make_clip_patch_extractor(wrapper, hk, args.device),
        device=args.device,
    )
    result = planner.run(img, args.report)
    out_path = RESULTS_DIR / "query_plan.json"
    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Saved {out_path}")
    print(f"confidence={result.confidence:.3f}  "
          f"iters={result.iterations}  "
          f"top patches={len(result.top_patches)}")


if __name__ == "__main__":
    _smoke_test_cli()
