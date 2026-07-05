"""WP-P1 frame-level [PC] eval: compare no-op render (targets/) vs enhanced render
(refined/) against held-out GT (gt/), using the SAME LPIPS as the trainer/region evals
(torchmetrics VGG normalize=True), on full frame + deck + lid boxes. Writes JSON.

[PC — frame-level]: these are per-frame 2D numbers, NOT splat-level; they measure the
enhancer's raw 2D power and gate whether a splat-level distill is worth running.

Usage: python _infusion/wpp1_frame_eval.py --work qa/wpp1_gq_frame --tag geoquery \
         --out _infusion/wpp1_geoquery_frame.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import get_lpips  # noqa: E402

DECK = (0.30, 0.72, 0.55, 0.95)
LID = (0.30, 0.72, 0.18, 0.58)


def to_t(p, device):
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def crop(t, box):
    _, _, H, W = t.shape
    x0, x1 = int(box[0] * W), int(box[1] * W)
    y0, y1 = int(box[2] * H), int(box[3] * H)
    return t[:, :, y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", default="enh")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    work = Path(args.work)
    man = json.loads((work / "manifest.json").read_text())
    lp = get_lpips(device)

    acc = {"noop": {"full": [], "deck": [], "lid": []},
           args.tag: {"full": [], "deck": [], "lid": []}}
    per_view = []
    with torch.no_grad():
        for e in man["entries"]:
            gt = to_t(work / e["gt_png"], device)
            noop = to_t(work / e["target_png"], device)
            enh = to_t(work / "refined" / f"{e['i']:04d}.png", device)
            row = {"i": e["i"], "stem": e.get("stem")}
            for arm, img in (("noop", noop), (args.tag, enh)):
                fl = float(lp(img, gt))
                dl = float(lp(crop(img, DECK), crop(gt, DECK)))
                ll = float(lp(crop(img, LID), crop(gt, LID)))
                acc[arm]["full"].append(fl)
                acc[arm]["deck"].append(dl)
                acc[arm]["lid"].append(ll)
                row[arm] = {"full": fl, "deck": dl, "lid": ll}
            per_view.append(row)

    def mean(arm):
        return {k: float(np.mean(acc[arm][k])) for k in ("full", "deck", "lid")}

    res = {"metric": "VGG-LPIPS normalize=True [PC frame-level]", "n": len(per_view),
           "boxes": {"deck": DECK, "lid": LID},
           "noop": mean("noop"), args.tag: mean(args.tag)}
    m0, m1 = res["noop"], res[args.tag]
    res["delta_enh_minus_noop"] = {
        k: {"abs": m1[k] - m0[k], "pct": (m1[k] - m0[k]) / m0[k] * 100.0}
        for k in ("full", "deck", "lid")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({**res, "per_view": per_view}, indent=2),
                              encoding="utf-8")
    print(f"[PC frame-level] {args.tag} vs no-op (LPIPS, lower=better):")
    for k in ("full", "deck", "lid"):
        d = res["delta_enh_minus_noop"][k]
        print(f"  {k:5s}: noop {m0[k]:.4f}  {args.tag} {m1[k]:.4f}  "
              f"delta {d['abs']:+.4f} ({d['pct']:+.1f}%)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
