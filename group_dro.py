"""
group_dro.py
============
Group Distributionally Robust Optimisation (Group DRO, Sagawa et al. 2020)
applied on top of the SAE-based shortcut detector.

Pipeline:
  1. Load SAE activations + labels (computed once by the main run).
  2. Use the top-K shortcut features from results/candidate_features.json to
     bucket every training example into a *group*.  Default grouping rule:
       g = ( label,  high_activation_of_shortcut_features? )
     i.e. four groups: {(y=0, low), (y=0, high), (y=1, low), (y=1, high)}.
     This matches the "aligned vs conflicting" split used elsewhere in the
     pipeline but is derived purely from the SAE — no human attribute labels.
  3. Train a linear classifier on the SAE features with the Group DRO update:
        q_g ← q_g · exp( η_q · L_g )    (then renormalise)
        loss = Σ_g  q_g · L_g
     Reports per-epoch worst-group accuracy and gap.

Inputs:
  - results/candidate_features.json  (either shape — see evidence_reporter.load_candidates)
  - SAE activations + labels: either supplied programmatically OR a .pt file
    saved by the main pipeline at results/sae_pooled_activations.pt.

Outputs:
  - results/group_dro_results.json
  - graph_res/group_dro_curves.png

CLI smoke test:
  python group_dro.py --activations results/sae_pooled_activations.pt \
                      --candidates  results/candidate_features.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("group_dro")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  [dro] %(message)s",
                    datefmt="%H:%M:%S")

RESULTS_DIR = Path(r"d:\SAE\results")
GRAPH_DIR   = Path(r"d:\SAE\graph_res")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Section 1 — Group assignment from SAE shortcut features
# ══════════════════════════════════════════════════════════════════════════════

def assign_groups(
    sae_acts:      torch.Tensor,         # (N, d_sae) pooled SAE activations
    labels:        torch.Tensor,         # (N,)   binary
    shortcut_ids:  list[int],
    quantile:      float = 0.5,
) -> tuple[torch.Tensor, dict[int, str]]:
    """
    Group label rule:
        score_i = mean(sae_acts[i, shortcut_ids])
        bg_i    = 1 if score_i > quantile(score) else 0
        group   = 2 * label_i + bg_i           ∈ {0,1,2,3}

    Returns (group_tensor (N,), group_name_map).
    """
    assert sae_acts.dim() == 2, f"expected (N, d_sae), got {tuple(sae_acts.shape)}"
    assert len(shortcut_ids) > 0, "shortcut_ids must be non-empty"

    score = sae_acts[:, shortcut_ids].mean(dim=1)
    thr   = score.quantile(quantile).item()
    bg    = (score > thr).long()
    g     = (2 * labels.long() + bg).long()

    names = {
        0: "y=0, shortcut=low",   # aligned (no-tumor, low artifact)
        1: "y=0, shortcut=high",  # conflicting
        2: "y=1, shortcut=low",   # conflicting
        3: "y=1, shortcut=high",  # aligned (tumor, high artifact)
    }
    counts = {k: int((g == k).sum().item()) for k in range(4)}
    log.info(f"group counts: {counts}  threshold={thr:.4f}")
    return g, names


# ══════════════════════════════════════════════════════════════════════════════
#  Section 2 — Linear classifier + Group DRO training loop
# ══════════════════════════════════════════════════════════════════════════════

class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(d_in, n_classes)

    def forward(self, x):
        return self.fc(x)


def group_dro_train(
    X:               torch.Tensor,       # (N, d_sae)
    y:               torch.Tensor,       # (N,)
    g:               torch.Tensor,       # (N,) group ids 0..G-1
    n_groups:        int,
    epochs:          int   = 50,
    lr:              float = 1e-3,
    eta_q:           float = 0.01,       # adversary step size on q
    weight_decay:    float = 1e-4,
    batch_size:      int   = 128,
    device:          str   = "cuda" if torch.cuda.is_available() else "cpu",
    seed:            int   = 42,
    val_frac:        float = 0.2,
) -> dict:
    """
    Trains a linear probe with the Group DRO update.

    History dict returned:
        {"epoch_loss": [...], "epoch_worst_acc": [...],
         "epoch_per_group_acc": [[g0..g3], ...], "q_final": [...] }
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    N = X.shape[0]
    perm = rng.permutation(N)
    n_val = int(N * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr, X_va = X[tr_idx].to(device), X[val_idx].to(device)
    y_tr, y_va = y[tr_idx].to(device), y[val_idx].to(device)
    g_tr, g_va = g[tr_idx].to(device), g[val_idx].to(device)

    model = LinearProbe(X.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Adversary weights over groups (uniform initial)
    q = torch.ones(n_groups, device=device) / n_groups

    history = {"epoch_loss": [], "epoch_worst_acc": [],
               "epoch_per_group_acc": [], "epoch_avg_acc": []}

    n_tr = X_tr.shape[0]
    for ep in range(epochs):
        model.train()
        order = torch.randperm(n_tr, device=device)
        running = 0.0
        n_batches = 0
        for s in range(0, n_tr, batch_size):
            bidx = order[s: s + batch_size]
            logits = model(X_tr[bidx])
            losses_per_sample = F.cross_entropy(logits, y_tr[bidx], reduction="none")

            # Per-group mean loss (skip empty groups in this batch)
            L_g = torch.zeros(n_groups, device=device)
            present = torch.zeros(n_groups, device=device)
            for gg in range(n_groups):
                m = (g_tr[bidx] == gg)
                if m.any():
                    L_g[gg]     = losses_per_sample[m].mean()
                    present[gg] = 1.0

            # Adversary update on present groups
            q_new = q.clone()
            q_new = q_new * torch.exp(eta_q * L_g.detach() * present)
            q_new = q_new / q_new.sum().clamp_min(1e-8)
            q     = q_new

            loss = (q.detach() * L_g).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss.item()); n_batches += 1

        # ── Eval on validation set ─────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            preds = model(X_va).argmax(dim=1)
            correct = (preds == y_va)
            per_group_acc = []
            for gg in range(n_groups):
                m = (g_va == gg)
                per_group_acc.append(
                    float(correct[m].float().mean().item()) if m.any() else float("nan")
                )
            valid = [a for a in per_group_acc if not np.isnan(a)]
            worst = min(valid) if valid else float("nan")
            avg   = float(correct.float().mean().item())

        history["epoch_loss"].append(running / max(n_batches, 1))
        history["epoch_worst_acc"].append(worst)
        history["epoch_per_group_acc"].append(per_group_acc)
        history["epoch_avg_acc"].append(avg)

        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            log.info(f"ep {ep+1:3d}/{epochs}  loss={running/max(n_batches,1):.4f}  "
                     f"worst={worst:.3f}  avg={avg:.3f}  q={q.tolist()}")

    history["q_final"] = q.cpu().tolist()
    history["d_in"]    = X.shape[1]
    history["n_groups"]= n_groups
    return history


# ══════════════════════════════════════════════════════════════════════════════
#  Section 3 — Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_curves(history: dict, save_path: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(history["epoch_loss"], color="#1F2937")
    axes[0].set_title("Group DRO loss"); axes[0].set_xlabel("epoch")

    axes[1].plot(history["epoch_avg_acc"],   label="avg",   color="#1D9E75")
    axes[1].plot(history["epoch_worst_acc"], label="worst", color="#D85A30")
    axes[1].set_title("Avg vs worst-group accuracy"); axes[1].set_xlabel("epoch")
    axes[1].legend()

    pg = np.array(history["epoch_per_group_acc"])
    for gi in range(pg.shape[1]):
        axes[2].plot(pg[:, gi], label=f"g={gi}")
    axes[2].set_title("Per-group accuracy"); axes[2].set_xlabel("epoch")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"saved curves → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Section 4 — Convenience runner that ties everything together
# ══════════════════════════════════════════════════════════════════════════════

def run(
    sae_acts:      torch.Tensor,
    labels:        torch.Tensor,
    shortcut_ids:  list[int],
    epochs:        int   = 50,
    save:          bool  = True,
) -> dict:
    """High-level entry point used by the main pipeline (no I/O assumed)."""
    g, group_names = assign_groups(sae_acts, labels, shortcut_ids)
    history = group_dro_train(
        sae_acts, labels, g, n_groups=4, epochs=epochs,
    )
    out = {
        "shortcut_feature_ids": shortcut_ids,
        "group_names":          group_names,
        "history":              history,
        "final_worst_acc":      history["epoch_worst_acc"][-1],
        "final_avg_acc":        history["epoch_avg_acc"][-1],
        "gap":                  history["epoch_avg_acc"][-1]
                                - history["epoch_worst_acc"][-1],
    }
    if save:
        json_path = RESULTS_DIR / "group_dro_results.json"
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2)
        log.info(f"saved {json_path}")
        plot_curves(history, GRAPH_DIR / "group_dro_curves.png")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CLI smoke test
# ══════════════════════════════════════════════════════════════════════════════

def _cli():
    from evidence_reporter import load_candidates

    p = argparse.ArgumentParser()
    p.add_argument("--activations", required=True,
                   help="Torch .pt file with dict {'X':(N,d_sae), 'y':(N,)}")
    p.add_argument("--candidates",
                   default=str(RESULTS_DIR / "candidate_features.json"))
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--epochs", type=int, default=50)
    args = p.parse_args()

    blob = torch.load(args.activations, map_location="cpu")
    X, y = blob["X"], blob["y"]
    if X.dim() == 3:           # (N, seq, d_sae) → mean-pool over patches
        X = X.mean(dim=1)
    log.info(f"X={tuple(X.shape)}  y={tuple(y.shape)}")

    candidates = load_candidates(Path(args.candidates))
    shortcut_ids = [int(c["feature_id"]) for c in candidates[:args.top_k]]

    out = run(X, y, shortcut_ids, epochs=args.epochs)
    print(json.dumps({k: out[k] for k in ("final_worst_acc",
                                          "final_avg_acc", "gap")},
                     indent=2))


if __name__ == "__main__":
    _cli()
