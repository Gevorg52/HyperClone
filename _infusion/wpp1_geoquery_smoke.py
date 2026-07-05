"""WP-P1 GeoQuery feasibility smoke test (run in the geoquery venv).

Verifies on the RTX PRO 6000 / sm_120 / torch 2.11:
  1. old diffusion pins import (diffusers 0.25.1, transformers 4.38.0, peft 0.9.0)
     under torch 2.11 + numpy 2.4;
  2. GeoQuery(pretrained_path=.pkl) loads (triggers stabilityai/sd-turbo pull);
  3. one sample() forward runs end-to-end on a synthetic (target, reference,
     depth, K, poses) tuple and returns a same-size PIL image.

Usage (from D:\\REPOS\\HYPERclone\\GeoQuery):
  set PYTHONPATH=src
  <geoquery venv python> _infusion/wpp1_geoquery_smoke.py --pkl weights/geoquery_window_size3_down1.pkl
"""
import argparse
import time
import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--w", type=int, default=904)
    args = ap.parse_args()

    import transformers, diffusers, peft
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"transformers {transformers.__version__} diffusers {diffusers.__version__} "
          f"peft {peft.__version__} numpy {np.__version__}", flush=True)

    from model import GeoQuery  # src/ on PYTHONPATH

    t0 = time.time()
    gq = GeoQuery(pretrained_path=args.pkl, timestep=199,
                  neighborhood_size=args.window_size, low_res_only=True)
    gq.set_eval() if hasattr(gq, "set_eval") else None
    gq.to("cuda")
    print(f"[load] GeoQuery + sd-turbo loaded in {time.time()-t0:.1f}s", flush=True)

    # synthetic pair
    rng = np.random.default_rng(0)
    tgt = Image.fromarray(rng.integers(0, 255, (args.h, args.w, 3), dtype=np.uint8))
    ref = Image.fromarray(rng.integers(0, 255, (args.h, args.w, 3), dtype=np.uint8))
    depth = (rng.random((args.h, args.w)).astype(np.float32) * 4 + 1)  # 1..5
    fx = fy = 0.9 * args.w
    K = np.array([[fx, 0, args.w / 2], [0, fy, args.h / 2], [0, 0, 1]], dtype=np.float32)
    ref_pose = np.eye(4, dtype=np.float32)
    tgt_pose = np.eye(4, dtype=np.float32)
    tgt_pose[0, 3] = 0.05  # slight baseline

    t1 = time.time()
    out, elapsed = gq.sample(tgt, ref, depth, K, ref_pose, tgt_pose,
                             prompt="remove degradation")
    print(f"[sample] out size {out.size} fwd {elapsed:.3f}s wall {time.time()-t1:.3f}s",
          flush=True)
    assert out.size == tgt.size, f"size mismatch {out.size} vs {tgt.size}"
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
