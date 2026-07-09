"""Persisted trainer-stall (GPU-contention) measurement for UI-2.

The preview subprocess only shares the GPU with the trainer for the STRIP RENDER LOOP
(`gpu_render_burst`, measured with torch.cuda.synchronize inside gui2/preview.render_strip);
the rest of the subprocess wall time is CPU-side (torch import + COLMAP parse) and does not
contend for GPU compute. This script renders a strip from a checkpoint (cold + warm) and
lets render_strip print the burst, so the "~0.2s @ 3M gaussians" headline is reproducible
from a committed artifact.

Re-run (captures to _infusion/ui2_burst_3m.log):
  python _infusion/ui2_burst_measure.py > _infusion/ui2_burst_3m.log 2>&1
Defaults target the WP-X1 cap3m (3,000,000-Gaussian) checkpoint.
"""
import sys, time, argparse, tempfile
from pathlib import Path

REPO = Path(r"D:\REPOS\HYPERclone\apex-sfm")
sys.path.insert(0, str(REPO))


def measure(ckpt, data_dir, factor, mode, label, runs=2):
    from gui2.preview import render_strip
    out = str(Path(tempfile.gettempdir()) / f"ui2_burst_{label}.png")
    print(f"### {label}\n  ckpt={ckpt}\n  data_dir={data_dir} factor={factor} mode={mode}")
    for i in range(runs):
        t0 = time.time()
        render_strip(ckpt, data_dir, out, step=0, factor=factor, mode=mode, gpu=0)
        print(f"  [{label}] run {i + 1}: subprocess_equiv_wall={time.time() - t0:.2f}s "
              f"(gpu_render_burst printed on the line above)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        REPO / "_dataset/GPU_V4/apex_images/training/wpx1/cap3m/output/ckpts/ckpt_29999_rank0.pt"))
    ap.add_argument("--data_dir", default=str(
        REPO / "_dataset/GPU_V4/apex_images/training/gsplat_prod/data"))
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--mode", default="object")
    ap.add_argument("--label", default="cap3m")
    a = ap.parse_args()
    print(f"UI-2 GPU-contention burst measurement @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
    measure(a.ckpt, a.data_dir, a.factor, a.mode, a.label, runs=2)
    print("\nDONE — gpu_render_burst is the trainer's worst-case GPU-contention window.")
