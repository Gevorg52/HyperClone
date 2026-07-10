"""UI-3 programmatic smoke: exercise the Mask Review pipeline the way gui2/app.py wires it,
without a browser. Validates review.py + sam_service + delivery-flag assembly end to end.

Re-runnable:  python _infusion/ui3_ui_smoke.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from gui2 import review as RV  # noqa: E402
from gui2 import sam_service as SAM  # noqa: E402
from gui2 import defaults as D  # noqa: E402

RUN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "_dataset", "SONY_TEST")
trace = {}


def check(name, cond, extra=""):
    trace[name] = {"pass": bool(cond), "info": extra}
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    return cond


# 1) build the Gradio Blocks (structural wiring validity)
from gui2.app import create_app  # noqa: E402
_app = create_app()
check("app_builds", _app is not None)

# 2) representative-view picking (uniform orbit spread)
n_probe = 588
idxs = RV.spread_indices(n_probe, 12)
check("spread_indices", len(idxs) == 12 and idxs[0] == 0 and idxs[-1] == n_probe - 1,
      f"idxs={idxs}")

# 3) ordered images + working frames (reuses existing sam_work/frames)
images = RV.ordered_images(RUN_DIR)
check("ordered_images", len(images) == 588, f"n={len(images)}")
t0 = time.time()
fdir, stems, _ = RV.prepare_working_frames(RUN_DIR)
check("prepare_working_frames", len(stems) == 588,
      f"n={len(stems)} in {time.time()-t0:.1f}s (reuse)")

# 4) representative mattes (reuse delivery masks_2 as the current matte source)
idxs = RV.spread_indices(len(stems), 12)
matte_dir = os.path.join(RUN_DIR, "sam_work", "mattes")
existing = os.path.join(RUN_DIR, "training", "gsplat", "data", "masks_2")
RV.ensure_mattes(fdir, stems, idxs, matte_dir, existing_mask_dir=existing)
have = sum(os.path.isfile(os.path.join(matte_dir, stems[i] + ".png")) for i in idxs)
check("ensure_mattes", have == 12, f"mattes={have}/12")

# 5) gallery items (overlay rgb + caption per representative view)
items = RV.gallery_items(fdir, stems, idxs, matte_dir, tint=RV.BIREFNET_TINT)
ok_gallery = (len(items) == 12 and all(isinstance(im, np.ndarray) and im.ndim == 3
                                       for im, _cap in items))
check("gallery_items", ok_gallery, f"n={len(items)}")

# 6) live single-frame prompting (the 'Fix it' edit click path)
seed = stems.index("DSC01497")
frame = RV.load_frame(fdir, seed)
mask = SAM.prompt_frame(frame, [[430, 330], [515, 585]], [1, 0], image_key="smoke::seed")
frac = float((mask > 127).mean()) if mask is not None else 0.0
check("prompt_frame", mask is not None and 0.02 < frac < 0.6,
      f"mask fg frac={frac:.3f}")

# 7) overlay with include/exclude dots (the live edit render)
ov = RV.overlay_rgb(frame, mask, RV.INCLUDE_TINT,
                    points=[[430, 330], [515, 585]], labels=[1, 0])
check("overlay_rgb", ov.shape == frame.shape)

# 8) accept -> masks_reviewed staging (uses the already-propagated masks_sam)
src = os.path.join(RUN_DIR, "sam_work", "masks_sam")
if os.path.isdir(src):
    d, nwr = RV.accept_reviewed(RUN_DIR, src, stems)
    check("accept_reviewed", nwr == 588, f"staged={nwr} -> {d}")
else:
    check("accept_reviewed", False, "no masks_sam (run propagation first)")

# 9) delivery-flag assembly: reviewed dir -> --mask_dir, NOT --gen_masks
obj_settings = D.defaults_for("object", False)
flags_rev = D.assemble_delivery_flags(obj_settings, reviewed_mask_dir=RV.masks_reviewed_dir(RUN_DIR))
flags_def = D.assemble_delivery_flags(obj_settings)
check("delivery_flags_reviewed",
      "--mask_dir" in flags_rev and "--gen_masks" not in flags_rev and "--extract_object" in flags_rev,
      " ".join(flags_rev))
check("delivery_flags_default_unchanged",
      "--gen_masks" in flags_def and "--mask_dir" not in flags_def,
      " ".join(flags_def))

# 10) 'Looks right' clears reviewed masks (delivery falls back to BiRefNet) — then restore
RV.clear_reviewed(RUN_DIR)
check("clear_reviewed", not os.path.isdir(RV.masks_reviewed_dir(RUN_DIR)))
RV.accept_reviewed(RUN_DIR, src, stems)  # restore for downstream

n_pass = sum(v["pass"] for v in trace.values())
summary = {"n_checks": len(trace), "n_pass": n_pass,
           "all_pass": n_pass == len(trace), "checks": trace}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui3_ui_smoke.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(f"\n=== UI-3 UI smoke: {n_pass}/{len(trace)} checks passed ===")
sys.exit(0 if summary["all_pass"] else 1)
