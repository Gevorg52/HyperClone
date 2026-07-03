"""Distill 2D per-view fills back into a SPLAT (WP-C1 Part 2).

Reuses the production trainer's extra-views reconstruction path (core.enhancement.
distill mechanism): the 23 filled hole views are appended to the training pool as
posed pseudo-views; the holed splat is the init; MCMC densifies the void to match
the fills across views (multiview fusion). The void-seeing camera arc is held out
via APEX_HOLDOUT so NO real view supervises the void with its true content -> the
void is filled ONLY by the generated fills (a genuine completion test, not
regrowth). Observed regions everywhere else stay real-supervised (faithful).

Chosen over the InFusion-seed path because InFusion's depth model runs in the
`difix` conda env, which does NOT exist on this migrated machine (same gap as
FLUX). MCMC densification-from-fills needs no extra env and directly tests the
decisive question: do per-view fills FUSE into a consistent 3D completion?

Usage: python _infusion/hole_distill.py --holed <pt> --fills <dir> --arcmeta <meta.json> \
          --data <dir> --out <dir> --holdout 0.30,0.36 [--steps 5000]
"""
import argparse, json, os, sys, subprocess
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holed", required=True)
    ap.add_argument("--fills", required=True, help="dir with vXXXX_fill.png")
    ap.add_argument("--arcmeta", required=True, help="arc meta.json (poses)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", default="0.30,0.36", help="APEX_HOLDOUT start,frac (void-seeing arc)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--cap_max", type=int, default=1400000)
    ap.add_argument("--factor", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = json.load(open(args.arcmeta))
    entries = []
    for stem in sorted(meta.keys()):
        fp = os.path.abspath(os.path.join(args.fills, f"{stem}_fill.png"))
        if not os.path.isfile(fp):
            print(f"[distill] WARN missing fill {fp}; skipping"); continue
        m = meta[stem]
        entries.append({"image": fp, "K": m["K"], "c2w": m["c2w"], "weight": 1.0})
    ev = os.path.join(args.out, "extra_views.json")
    Path(ev).write_text(json.dumps(entries), encoding="utf-8")
    print(f"[distill] {len(entries)} fill views -> {ev}")

    trainer = os.path.join(REPO, "core", "training", "gsplat_trainer.py")
    env = os.environ.copy()
    env["APEX_HOLDOUT"] = args.holdout      # hold out the void-seeing arc from REAL supervision
    env["PYTHONPATH"] = REPO
    cmd = [sys.executable, trainer,
           "--data_dir", args.data, "--result_dir", args.out,
           "--data_factor", str(args.factor), "--max_steps", str(args.steps),
           "--strategy", "mcmc", "--cap_max", str(args.cap_max),
           "--refine_stop_iter", str(int(args.steps * 0.8)),
           "--init_ckpt", os.path.abspath(args.holed),
           "--extra_views", ev,
           "--lpips_lambda", "0.2", "--lpips_gated", "--lpips_start", "500",
           "--antialiased", "--random_bkgd",
           "--save_steps", str(args.steps), "--seed", "0"]
    print("[distill] launching trainer:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    ck = os.path.join(args.out, "ckpts", f"ckpt_{args.steps - 1}_rank0.pt")
    print(f"[distill] done -> {ck}")


if __name__ == "__main__":
    main()
