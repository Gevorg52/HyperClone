"""Quantify floaters / geometry health across checkpoints (read-only, no writes).

For each checkpoint: Gaussian count, low-opacity fraction (floater candidates),
statistical-outlier fraction (isolated floaters far from the surface manifold —
same kNN test as prune.py), and giant-blob indicator (large-scale fraction).
Lets us check whether the aggressive v2 distill ADDED floaters vs the base.

Usage:  python floater_report.py <label=ckpt> [<label=ckpt> ...]
        e.g. python floater_report.py base=...\output\ckpts\ckpt_29999_rank0.pt \
                                       v2=...\distill_v2\round_3\output\ckpts\ckpt_7999_rank0.pt
"""
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree


def report(label, path):
    ck = torch.load(path, weights_only=True, map_location="cpu")
    s = ck["splats"]
    n = int(s["means"].shape[0])
    opa = torch.sigmoid(s["opacities"].float()).numpy()
    scl = torch.exp(s["scales"].float()).numpy()        # world-space sizes
    means = s["means"].float().numpy()

    low_opa = float((opa < 0.01).mean()) * 100
    # SOR outliers: mean kNN distance > mean + 2*std (prune.py's test)
    tree = cKDTree(means)
    d, _ = tree.query(means, k=9, workers=-1)
    knn = d[:, 1:].mean(axis=1)
    thr = knn.mean() + 2.0 * knn.std()
    outliers = float((knn > thr).mean()) * 100
    # giant blobs: max per-Gaussian scale far above the median
    maxscale = scl.max(axis=1)
    big = float((maxscale > 10 * np.median(maxscale)).mean()) * 100

    print(f"{label:14s} {n:>10,}  low-opa {low_opa:5.1f}%  "
          f"SOR-outlier {outliers:5.2f}%  giant {big:5.2f}%  "
          f"med-opa {np.median(opa):.3f}  knn-spread {knn.std()/knn.mean():.2f}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python floater_report.py <label=ckpt> ...")
    print(f"\n{'model':14s} {'Gaussians':>10s}  floater indicators "
          f"(lower=cleaner; compare vs base)\n")
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        report(label, path)


if __name__ == "__main__":
    main()
