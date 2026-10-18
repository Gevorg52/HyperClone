"""WP-F5 PPISP leak check (EXECUTION_PLAN: 'smoke-test 3k steps watching memory_allocated';
ppisp issue #6 was a ~6 MB/forward GPU leak in v1.2.x). Runs 3000 train-like steps at 1920x1080
(factor-2 Laptop size) through PPISP forward + reg loss + backward + optimizer/scheduler, with the
controller activating at 80 %, and records torch.cuda.memory_allocated every 250 steps.
  venvs/gsplat_main/Scripts/python.exe _infusion/wpf5_leakcheck.py --out _infusion/wpf5/leakcheck.json
"""
import argparse, json, time
import torch
from ppisp import PPISP, PPISPConfig

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=3000); a = ap.parse_args()
torch.manual_seed(0)
W, H, NF = 1920, 1080, 266
m = PPISP(num_cameras=1, num_frames=NF, config=PPISPConfig(use_controller=True))
opts = m.create_optimizers(); sch = m.create_schedulers(opts, a.steps)
x = torch.rand(H, W, 3, device="cuda")
torch.cuda.synchronize(); base = torch.cuda.memory_allocated()
rec = []; t0 = time.time()
for step in range(a.steps):
    rgb = (x * 1.0).requires_grad_(True)
    out = m(rgb, resolution=(W, H), camera_idx=0, frame_idx=step % NF)
    loss = (out - x).abs().mean() + m.get_regularization_loss()
    loss.backward()
    for o in opts: o.step(); o.zero_grad(set_to_none=True)
    for s in sch: s.step()
    if step % 250 == 0 or step == a.steps - 1:
        torch.cuda.synchronize()
        rec.append({"step": step, "alloc_mb": (torch.cuda.memory_allocated() - base) / 2**20})
torch.cuda.synchronize()
res = {"steps": a.steps, "resolution": [W, H], "num_frames": NF, "controller": True,
       "seconds": time.time() - t0, "ms_per_step": 1000 * (time.time() - t0) / a.steps,
       "alloc_delta_mb": rec, "growth_mb_first_to_last": rec[-1]["alloc_mb"] - rec[1]["alloc_mb"],
       "ppisp_version": __import__("ppisp").__version__ if hasattr(__import__("ppisp"), "__version__") else "1.3.0+a247368"}
open(a.out, "w").write(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
