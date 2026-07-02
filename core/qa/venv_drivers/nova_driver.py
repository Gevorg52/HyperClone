"""NOVA driver — RUNS INSIDE the qa venv (D:/REPOS/HYPERclone/venvs/qa).

Imports the cloned NOVA repo (timm DINOv2 + fine-tuned NOVA_NVS.pt) and computes
NOVA cosine distance for a list of (render, real-reference) pairs. FAILS HARD if
the checkpoint is missing (NOVA silently degrades to vanilla DINOv2 otherwise,
which would produce misleading scores).

Protocol: `python nova_driver.py <input.json> <output.json>` where input.json =
  {"repo": ".../NOVA", "checkpoint": ".../NOVA_NVS.pt",
   "pairs": [{"render": path, "ref": path, "index": i}, ...], "resize": 518}
output.json = {"per_view": [{index, cosine_distance, cosine_similarity}, ...],
               "mean": <mean distance>, "checkpoint_loaded": true}
"""
import json
import os
import sys

import numpy as np
import torch


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    cfg = json.load(open(in_path))
    repo = cfg["repo"]
    ckpt = cfg["checkpoint"]
    resize = int(cfg.get("resize", 518))

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"NOVA checkpoint missing: {ckpt} (refusing to run — NOVA would "
            f"silently fall back to vanilla DINOv2).")

    sys.path.insert(0, repo)
    import nova  # noqa: E402  (NOVA repo root module)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False path: NOVA's own load_model uses torch.load default; the
    # NOVA_NVS.pt is a trusted dict {base, embedding_head}. torch 2.6+ defaults to
    # weights_only=True which would reject it, so patch torch.load here.
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
    try:
        model = nova.load_model(checkpoint_path=ckpt, device=device)
    finally:
        torch.load = _orig_load

    per_view = []
    for pr in cfg["pairs"]:
        res = nova.compute_cosine_distance(
            model, image_a=pr["ref"], image_b=pr["render"],
            device=device, resize=resize)
        per_view.append({
            "index": pr["index"],
            "cosine_distance": float(res["cosine_distance"]),
            "cosine_similarity": float(res["cosine_similarity"]),
        })

    dists = [v["cosine_distance"] for v in per_view]
    out = {"per_view": per_view,
           "mean": float(np.mean(dists)) if dists else float("nan"),
           "median": float(np.median(dists)) if dists else float("nan"),
           "checkpoint_loaded": True,
           "n": len(per_view)}
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[nova_driver] scored {len(per_view)} pairs, mean_dist="
          f"{out['mean']:.4f}")


if __name__ == "__main__":
    main()
