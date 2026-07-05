"""WP-P1 SwiftVR frame-level [PC] eval: for each held-out (every-8) orbit frame, compare
the no-op render (in/<seq>.png) vs the SwiftVR-restored frame (out_full/<seq>.png, offset 0)
against GT (gt/<seq>.png), LPIPS on full + deck + lid boxes. Restored frames (1904x1072)
are resized to GT size for a fair comparison with the no-op render (GT size). [PC — frame-level]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import get_lpips  # noqa: E402

DECK = (0.30, 0.72, 0.55, 0.95); LID = (0.30, 0.72, 0.18, 0.58)


def to_t(p, device, size=None):
    im = Image.open(p).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def crop(t, box):
    _, _, H, W = t.shape
    return t[:, :, int(box[2]*H):int(box[3]*H), int(box[0]*W):int(box[1]*W)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    work = Path(args.work)
    man = json.loads((work / "orbit_manifest.json").read_text())
    lp = get_lpips(device)
    acc = {"noop": {"full": [], "deck": [], "lid": []},
           "swiftvr": {"full": [], "deck": [], "lid": []}}
    per_view = []
    with torch.no_grad():
        for h in man["heldout"]:
            seq = h["seq"]
            gtp = work / "gt" / f"{seq:04d}.png"
            gt = to_t(gtp, device)
            size = (gt.shape[3], gt.shape[2])  # (W,H)
            noop = to_t(work / "in" / f"{seq:04d}.png", device, size)
            rp = work / "out_full" / f"{seq - args.offset:04d}.png"
            if not rp.exists():
                continue
            enh = to_t(rp, device, size)
            row = {"seq": seq, "stem": h.get("stem")}
            for arm, img in (("noop", noop), ("swiftvr", enh)):
                r = {"full": float(lp(img, gt)),
                     "deck": float(lp(crop(img, DECK), crop(gt, DECK))),
                     "lid": float(lp(crop(img, LID), crop(gt, LID)))}
                for k in r: acc[arm][k].append(r[k])
                row[arm] = r
            per_view.append(row)
    mean = lambda a: {k: float(np.mean(acc[a][k])) for k in ("full", "deck", "lid")}
    res = {"metric": "VGG-LPIPS normalize=True [PC frame-level]", "n": len(per_view),
           "offset": args.offset, "boxes": {"deck": DECK, "lid": LID},
           "noop": mean("noop"), "swiftvr": mean("swiftvr")}
    m0, m1 = res["noop"], res["swiftvr"]
    res["delta_swiftvr_minus_noop"] = {k: {"abs": m1[k]-m0[k], "pct": (m1[k]-m0[k])/m0[k]*100.0}
                                       for k in ("full", "deck", "lid")}
    Path(args.out).write_text(json.dumps({**res, "per_view": per_view}, indent=2), encoding="utf-8")
    print(f"[PC frame-level] SwiftVR vs no-op (LPIPS, n={len(per_view)}):")
    for k in ("full", "deck", "lid"):
        d = res["delta_swiftvr_minus_noop"][k]
        print(f"  {k:5s}: noop {m0[k]:.4f}  swiftvr {m1[k]:.4f}  delta {d['abs']:+.4f} ({d['pct']:+.1f}%)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
