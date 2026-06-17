"""Score InFusion's completed depth vs ground truth inside the hole.

The hole region is fully observed by our splat, so true_depth.npy is the real
surface. InFusion only saw surrounding depth + the RGB. If its completed depth
matches truth in the hole, the depth-completion prior is faithful on our data.
Output depth is in the same metric scale as our input (normalized by the
known-region disparity min/max we supplied), so a direct comparison is valid.
"""
import os

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IO = r"F:\HYPERclone\apex-sfm\_infusion\io"

true = np.load(os.path.join(IO, "true_depth.npy"))            # (H,W) z-depth
alpha = np.load(os.path.join(IO, "alpha.npy"))
pred = np.load(os.path.join(IO, "out", "rgb_depth.npy"))      # (H,W) completed depth
mask = cv2.imread(os.path.join(IO, "mask.png"), cv2.IMREAD_GRAYSCALE) > 127

if pred.shape != true.shape:
    pred = cv2.resize(pred, (true.shape[1], true.shape[0]))

obs = alpha > 0.5
hole = mask & obs                # observed pixels inside the hole (have truth)
known = (~mask) & obs            # known region (blend should preserve)


def stats(m, label):
    t, p = true[m], pred[m]
    ae = np.abs(t - p)
    rel = ae / (t + 1e-6)
    cc = np.corrcoef(t, p)[0, 1] if m.sum() > 10 else float("nan")
    print(f"{label:16s} n={m.sum():>7d}  MAE={ae.mean():.4f}  "
          f"medAE={np.median(ae):.4f}  relMAE={rel.mean()*100:5.1f}%  "
          f"corr={cc:.3f}  (depth range {t.min():.3f}..{t.max():.3f})")
    return ae


print(f"scene depth (observed): {true[obs].min():.3f}..{true[obs].max():.3f}")
stats(known, "known (blend)")     # sanity: should be ~0 error
ae_hole = stats(hole, "HOLE (filled)")

# Visualization
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
vmin, vmax = true[obs].min(), true[obs].max()
rgb = cv2.cvtColor(cv2.imread(os.path.join(IO, "rgb.png")), cv2.COLOR_BGR2RGB)
ax[0].imshow(rgb); ax[0].set_title("RGB + hole")
ax[0].add_patch(plt.Rectangle((np.where(mask.any(0))[0][0], np.where(mask.any(1))[0][0]),
                mask.any(0).sum(), mask.any(1).sum(), fill=False, color="r", lw=2))
ax[1].imshow(np.where(obs, true, np.nan), vmin=vmin, vmax=vmax, cmap="turbo")
ax[1].set_title("TRUE depth")
pv = pred.copy(); pv[~obs] = np.nan
ax[2].imshow(pv, vmin=vmin, vmax=vmax, cmap="turbo"); ax[2].set_title("InFusion depth")
err = np.full_like(true, np.nan); err[hole] = np.abs(true - pred)[hole]
im = ax[3].imshow(err, cmap="hot"); ax[3].set_title("|err| in hole"); plt.colorbar(im, ax=ax[3])
for a in ax: a.axis("off")
out = os.path.join(IO, "score.png")
plt.tight_layout(); plt.savefig(out, dpi=90); print("viz:", out)
