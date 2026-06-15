"""Headless run of Stages 1-7 for the Laptop video-frame dataset.

Drives the real pipeline stage handlers (no Gradio GUI). Keeps all extracted
frames (the still-tuned sharpness metric is unreliable on 4K video frames),
uses NetVLAD retrieval matching (305-frame continuous orbit), incremental SfM,
final BA, and 3DGS export. Prints per-stage timing + stats.

Run from F:\\HYPERclone\\apex-sfm with the SfM conda env (anaconda base).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: F401  (stage handlers import/raise gr.*)

from core.pipeline_state import PipelineState
from core.utils import create_project_dir, get_image_list, get_reconstruction_stats
from gui.stages.extraction import run_extraction
from gui.stages.matching import run_matching
from gui.stages.reconstruction import run_reconstruction
from gui.stages.export import run_export, run_refinement

NOOP = lambda *a, **k: None  # noqa: E731  (stand-in for gr.Progress())

IMAGE_DIR = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\images"
EXPORT_DIR = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\export"


def banner(msg: str):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def main():
    t_start = time.time()

    # ---- Stage 1: import (keep ALL frames; video sharpness metric unreliable) ----
    banner("STAGE 1 — import (no cull)")
    state = PipelineState()
    state.image_dir = IMAGE_DIR
    state.project_dir = str(create_project_dir(str(Path(IMAGE_DIR).parent),
                                               Path(IMAGE_DIR).name))
    all_imgs = get_image_list(IMAGE_DIR)
    state.image_list = [str(p) for p in all_imgs]
    state.excluded_images = []
    state.num_images = len(all_imgs)
    state.save()
    print(f"project_dir = {state.project_dir}")
    print(f"kept {state.num_images} frames (no culling)")

    # ---- Stage 2: ALIKED feature extraction ----
    banner("STAGE 2 — ALIKED extraction (8192 kpts, resize 1600)")
    t = time.time()
    run_extraction("aliked", 8192, 3, False, 1600, False, state, NOOP)
    print(f"feature_path = {state.feature_path}")
    print(f"avg keypoints = {getattr(state, 'avg_keypoints', '?')}")
    print(f"[stage 2] {time.time() - t:.0f}s")

    # ---- Stage 3: LightGlue retrieval matching (NetVLAD top-50) ----
    banner("STAGE 3 — LightGlue matching (retrieval, top-50, adaptive off)")
    t = time.time()
    run_matching("retrieval", True, 0.1, 50, state, NOOP)
    print(f"pairs_path = {state.pairs_path}")
    print(f"match_path = {state.match_path}")
    print(f"num_pairs = {getattr(state, 'num_pairs', '?')}, "
          f"avg_matches = {getattr(state, 'avg_matches_per_pair', '?')}")
    print(f"[stage 3] {time.time() - t:.0f}s")

    # ---- Stages 4-5: incremental reconstruction ----
    banner("STAGE 4-5 — incremental SfM (OPENCV, single camera)")
    t = time.time()
    run_reconstruction("incremental", "OPENCV", True, 15, 40, 100,
                       True, True, True, state, NOOP)
    reg = state.num_registered or 0
    print(f"registered {reg}/{state.num_images} "
          f"({100 * reg / max(state.num_images, 1):.1f}%)")
    print(f"3D points = {state.num_points3d}, "
          f"mean reproj = {state.mean_reprojection_error:.3f} px")
    print(f"[stage 4-5] {time.time() - t:.0f}s")

    # ---- Stage 6: final BA ----
    banner("STAGE 6 — final bundle adjustment (200 iters)")
    t = time.time()
    run_refinement(200, True, True, True, state, NOOP)
    print(f"[stage 6] {time.time() - t:.0f}s")

    # ---- Stage 7: export to 3DGS layout ----
    banner("STAGE 7 — export (alignment on, centering on)")
    t = time.time()
    run_export(EXPORT_DIR, "binary", True, "y", True, True, False, "bw_masks",
               state, NOOP)
    print(f"export_dir = {state.export_dir}")
    stats = get_reconstruction_stats(str(Path(state.export_dir) / "sparse" / "0"))
    print(f"export stats: {stats}")
    print(f"[stage 7] {time.time() - t:.0f}s")

    banner(f"SfM COMPLETE in {(time.time() - t_start) / 60:.1f} min — "
           f"export ready at {state.export_dir}")


if __name__ == "__main__":
    main()
