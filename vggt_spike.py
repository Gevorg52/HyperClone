"""VGGT feasibility spike: run on a dozen OBSERVED GPU views -> dense depth + point
map. Confirms VGGT runs on the 4090 (VRAM/time) before building the render+inpaint
geometry-grounded prior. Reports shapes, point count, VRAM, runtime."""
import glob
import os
import time

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

IMG_DIR = r"F:\HYPERclone\apex-sfm\_dataset\GPU\export\images"
dev = "cuda"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def main():
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
    # take 12 from the OBSERVED arc (first ~25% of the orbit; wedge is 27.5-72.5%)
    obs = files[: int(0.25 * len(files))]
    pick = [obs[i] for i in np.linspace(0, len(obs) - 1, 12).astype(int)]
    print(f"VGGT spike on {len(pick)} observed views", flush=True)

    t = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    print(f"model loaded [{time.time()-t:.0f}s]", flush=True)

    images = load_and_preprocess_images(pick).to(dev)
    print(f"images tensor {tuple(images.shape)}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            pred = model(images)
    dt = time.time() - t
    vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\ninference {dt:.1f}s  peak VRAM {vram:.1f} GB", flush=True)
    print("prediction keys:", list(pred.keys()))
    for k, v in pred.items():
        if torch.is_tensor(v):
            print(f"  {k:18s} {tuple(v.shape)}")
    if "depth" in pred:
        d = pred["depth"]
        print(f"depth range [{float(d.min()):.3f}, {float(d.max()):.3f}], "
              f"~{d.numel():,} depth samples (dense geometry OK)")


if __name__ == "__main__":
    main()
