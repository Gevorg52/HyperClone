"""Stage 1: Image Import & Quality Check.

Provides folder loading, thumbnail gallery, sharpness scoring,
EXIF consistency validation, and include/exclude data table.

Performance: processes images in parallel threads with disk caching
so repeat loads are near-instant.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from core.pipeline_state import PipelineState
from core.utils import (
    create_project_dir,
    get_image_list,
    read_exif,
    validate_exif_consistency,
)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

CACHE_VERSION = 2
CACHE_FILENAME = "image_cache.json"


def _cache_key(img_path: Path) -> dict:
    """Return file identity: modification time + size for cache invalidation."""
    stat = img_path.stat()
    return {"mtime": stat.st_mtime, "size": stat.st_size}


def _load_cache(project_dir: Path) -> dict:
    """Load cached image processing results from disk."""
    cache_path = project_dir / CACHE_FILENAME
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("version") != CACHE_VERSION:
            return {}
        return data.get("images", {})
    except (json.JSONDecodeError, KeyError):
        return {}


def _save_cache(project_dir: Path, cache: dict):
    """Save image processing results to disk."""
    cache_path = project_dir / CACHE_FILENAME
    data = {"version": CACHE_VERSION, "images": cache}
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Single-image processing (one read, three results)
# ---------------------------------------------------------------------------

def _process_single_image(img_path: Path, thumb_size: int = 256) -> dict:
    """Process one image: read EXIF, load pixels once, compute sharpness + thumbnail.

    Returns dict with keys: sharpness, exif, thumbnail.
    """
    # EXIF is cheap — reads only header bytes, not pixel data
    exif = read_exif(img_path)

    # Read the full image ONCE
    img = cv2.imread(str(img_path))
    if img is None:
        return {
            "sharpness": 0.0,
            "exif": exif,
            "thumbnail": np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8),
        }

    # Fill in dimensions from actual pixels if EXIF missed them
    if exif["width"] is None or exif["height"] is None:
        exif["height"], exif["width"] = img.shape[:2]

    # Sharpness: 90th percentile of 8x8 tile Laplacian variances
    # Robust to depth-of-field (ignores bokeh tiles), sensitive to motion blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tile_grid = 8
    th, tw = gray.shape[0] // tile_grid, gray.shape[1] // tile_grid
    variances = []
    for r in range(tile_grid):
        for c in range(tile_grid):
            tile = gray[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            variances.append(float(cv2.Laplacian(tile, cv2.CV_64F).var()))
    sharpness = float(np.percentile(variances, 90))

    # Thumbnail: resize from the already-loaded image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = thumb_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    thumbnail = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return {"sharpness": sharpness, "exif": exif, "thumbnail": thumbnail}


def _process_cached_image(img_path: Path, cached: dict, thumb_size: int = 256) -> dict:
    """Use cached sharpness/exif — only generate thumbnail (fast at 256px)."""
    exif = cached["exif"]

    # Thumbnail still needs pixel data but we can read at reduced scale
    # OpenCV doesn't support direct reduced-read for JPEG, so read full + resize
    img = cv2.imread(str(img_path))
    if img is None:
        thumb = np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        scale = thumb_size / max(h, w)
        thumb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    return {"sharpness": cached["sharpness"], "exif": exif, "thumbnail": thumb}


# ---------------------------------------------------------------------------
# Browse folder (tkinter dialog)
# ---------------------------------------------------------------------------

def _browse_folder() -> str:
    """Open a native folder selection dialog. Returns path or empty string."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select Image Folder")
    root.destroy()
    return folder or ""


# ---------------------------------------------------------------------------
# Main load function
# ---------------------------------------------------------------------------

# Number of threads for parallel image processing.
# OpenCV JPEG decode releases the GIL, so threads are effective.
_NUM_WORKERS = min(8, (os.cpu_count() or 4))


def load_images(folder_path: str, state: PipelineState, progress=gr.Progress()):
    """Load images from folder with parallel processing and disk caching."""
    if not folder_path or not Path(folder_path).is_dir():
        raise gr.Error("Please enter a valid folder path.")

    progress(0, desc="Scanning folder...")
    images = get_image_list(folder_path)
    if not images:
        raise gr.Error(f"No supported images found in: {folder_path}")

    # Create project directory alongside the image folder
    project_dir = create_project_dir(str(Path(folder_path).parent), Path(folder_path).name)

    # Load cache
    cache = _load_cache(project_dir)

    # Classify images: cache-hit vs. needs-processing
    to_process = []  # (index, img_path) — need full processing
    cached_hits = []  # (index, img_path, cached_data) — cache valid
    for i, img_path in enumerate(images):
        name = img_path.name
        key = _cache_key(img_path)
        if (name in cache
                and cache[name].get("mtime") == key["mtime"]
                and cache[name].get("size") == key["size"]):
            cached_hits.append((i, img_path, cache[name]))
        else:
            to_process.append((i, img_path))

    total = len(images)
    results = [None] * total  # ordered results
    new_cache = {}

    cache_count = len(cached_hits)
    process_count = len(to_process)

    if cache_count > 0:
        progress(0.01, desc=f"Cache hit for {cache_count}/{total} images. "
                            f"Processing {process_count} new...")

    # --- Process cached images (just need thumbnails) ---
    # Still parallelize since we need to read pixels for thumbnails
    completed = 0
    lock = threading.Lock()

    def _do_cached(item):
        idx, img_path, cached_data = item
        return idx, _process_cached_image(img_path, cached_data)

    def _do_full(item):
        idx, img_path = item
        return idx, _process_single_image(img_path)

    # Run all work in parallel: cached (thumbnail-only) + full processing
    all_tasks = (
        [("cached", item) for item in cached_hits] +
        [("full", item) for item in to_process]
    )

    with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as pool:
        futures = {}
        for task_type, item in all_tasks:
            if task_type == "cached":
                fut = pool.submit(_do_cached, item)
            else:
                fut = pool.submit(_do_full, item)
            futures[fut] = task_type

        for fut in as_completed(futures):
            idx, result = fut.result()
            results[idx] = result
            with lock:
                completed += 1
            if completed % 10 == 0 or completed == total:
                progress(completed / total,
                         desc=f"Processed {completed}/{total} images...")

    # --- Build outputs ---
    thumbnails = []
    table_rows = []
    exif_list = []
    updated_cache = {}

    for i, img_path in enumerate(images):
        r = results[i]
        exif = r["exif"]
        sharpness = r["sharpness"]
        exif_list.append(exif)

        thumbnails.append((r["thumbnail"], f"{img_path.name} — {round(sharpness, 1)}"))

        table_rows.append([
            img_path.name,
            round(sharpness, 1),
            exif.get("width") or 0,
            exif.get("height") or 0,
            exif.get("focal_length_mm") or "?",
            exif.get("camera_model") or "unknown",
            True,  # included by default
        ])

        # Update cache entry
        key = _cache_key(img_path)
        updated_cache[img_path.name] = {
            "mtime": key["mtime"],
            "size": key["size"],
            "sharpness": round(sharpness, 2),
            "exif": exif,
        }

    # Sort by sharpness ascending (blurriest first for easy identification)
    sort_indices = sorted(range(len(table_rows)), key=lambda i: table_rows[i][1])
    thumbnails = [thumbnails[i] for i in sort_indices]
    table_rows = [table_rows[i] for i in sort_indices]

    # Save cache for next time
    _save_cache(project_dir, updated_cache)

    # Validate EXIF consistency
    warnings = validate_exif_consistency(exif_list)
    warning_text = ("\n".join(f"WARNING: {w}" for w in warnings)
                    if warnings else "All images consistent.")

    # Representative resolution and focal length
    rep_exif = exif_list[0] if exif_list else {}
    img_w = rep_exif.get("width") or 0
    img_h = rep_exif.get("height") or 0
    focal = rep_exif.get("focal_length_mm")

    # Restore previous pipeline progress if it exists (survives app restarts)
    saved = PipelineState.load(str(project_dir))
    if saved.project_dir:
        # Carry forward progress from completed stages
        state.feature_path = saved.feature_path
        state.extractor_name = saved.extractor_name
        state.max_keypoints_used = saved.max_keypoints_used
        state.avg_keypoints = saved.avg_keypoints
        state.match_path = saved.match_path
        state.pairs_path = saved.pairs_path
        state.matcher_name = saved.matcher_name
        state.num_pairs = saved.num_pairs
        state.avg_matches_per_pair = saved.avg_matches_per_pair
        state.sfm_dir = saved.sfm_dir
        state.num_registered = saved.num_registered
        state.num_points3d = saved.num_points3d
        state.mean_reprojection_error = saved.mean_reprojection_error
        state.refined = saved.refined
        state.pre_ba_error = saved.pre_ba_error
        state.post_ba_error = saved.post_ba_error
        state.export_dir = saved.export_dir

    # Update state with current stage 1 results (must set project_dir before discover)
    state.image_dir = folder_path
    state.project_dir = str(project_dir)
    state.image_list = [str(p) for p in images]
    state.excluded_images = []
    state.num_images = len(images)
    state.image_width = img_w
    state.image_height = img_h
    state.focal_length_mm = focal

    # Discover outputs on disk that may not be in the saved state
    state.discover_outputs()

    # Save state to disk
    state.save()

    cache_msg = ""
    if cache_count > 0:
        cache_msg = f"\n(Cached: {cache_count}, reprocessed: {process_count})"

    # Show what stages are already complete
    resumed = []
    if state.feature_path:
        resumed.append("Stage 2 (features)")
    if state.match_path:
        resumed.append("Stage 3 (matches)")
    if state.sfm_dir:
        resumed.append("Stages 4-5 (reconstruction)")
    if state.refined:
        resumed.append("Stage 6 (refinement)")
    if state.export_dir:
        resumed.append("Stage 7 (export)")
    resume_msg = f"\nResumed: {', '.join(resumed)}" if resumed else ""

    stats_text = (
        f"Images: {len(images)}\n"
        f"Resolution: {img_w} x {img_h}\n"
        f"Focal length: {focal} mm\n"
        f"Camera: {rep_exif.get('camera_make', '?')} {rep_exif.get('camera_model', '?')}\n"
        f"Project dir: {project_dir}"
        f"{cache_msg}"
        f"{resume_msg}"
    )

    # Initial sharpness summary at default threshold (200)
    sharpness_summary = _build_sharpness_summary(table_rows, 200.0)

    return thumbnails, table_rows, warning_text, stats_text, sharpness_summary, state


# ---------------------------------------------------------------------------
# Sharpness summary helpers
# ---------------------------------------------------------------------------

def _build_sharpness_summary(table_data, threshold: float) -> str:
    """Build HTML summary of sharpness distribution relative to threshold."""
    scores = [float(row[1]) for row in table_data]
    blurry = sum(1 for s in scores if s < threshold)
    borderline = sum(1 for s in scores if threshold <= s < threshold * 1.5)
    sharp = sum(1 for s in scores if s >= threshold * 1.5)
    total = len(scores)

    return (
        f'<div style="display:flex;gap:16px;padding:8px 0;font-family:monospace">'
        f'<span style="color:#e74c3c">Blurry (&lt;{threshold:.0f}): <b>{blurry}</b></span>'
        f'<span style="color:#f39c12">Borderline ({threshold:.0f}\u2013{threshold*1.5:.0f}): <b>{borderline}</b></span>'
        f'<span style="color:#27ae60">Sharp (&gt;{threshold*1.5:.0f}): <b>{sharp}</b></span>'
        f'<span>Total: {total}</span>'
        f'</div>'
    )


def _update_threshold_preview(table_data, threshold):
    """Update the sharpness group summary when slider moves."""
    if isinstance(table_data, pd.DataFrame):
        if table_data.empty:
            return "No images loaded."
        table_data = table_data.values.tolist()
    elif not table_data:
        return "No images loaded."
    return _build_sharpness_summary(table_data, threshold)


# ---------------------------------------------------------------------------
# Auto-cull and apply table changes
# ---------------------------------------------------------------------------

def auto_cull(table_data, threshold: float, state: PipelineState):
    """Mark images below sharpness threshold as excluded."""
    # Gradio Dataframe passes pandas DataFrame, convert to list of lists
    if isinstance(table_data, pd.DataFrame):
        if table_data.empty:
            raise gr.Error("No images loaded. Load a folder first.")
        table_data = table_data.values.tolist()
    elif not table_data:
        raise gr.Error("No images loaded. Load a folder first.")

    culled = 0
    updated_rows = []
    included_images = []
    excluded_images = []

    for row in table_data:
        name, sharpness = row[0], row[1]
        included = float(sharpness) >= threshold
        updated_rows.append([row[0], row[1], row[2], row[3], row[4], row[5], included])
        if included:
            included_images.append(name)
        else:
            excluded_images.append(name)
            culled += 1

    # Update state image list based on include/exclude
    if state.image_dir:
        base = Path(state.image_dir)
        state.image_list = [str(base / name) for name in included_images]
        state.excluded_images = [str(base / name) for name in excluded_images]
        state.num_images = len(included_images)

    cull_msg = (f"Auto-cull: {culled} images below threshold {threshold}. "
                f"{len(included_images)} images remaining.")

    summary_html = _build_sharpness_summary(updated_rows, threshold)

    state.save()
    return updated_rows, cull_msg, summary_html, state


def apply_table_changes(table_data, state: PipelineState):
    """Sync the include/exclude checkboxes back to pipeline state."""
    # Gradio Dataframe passes pandas DataFrame, convert to list of lists
    if isinstance(table_data, pd.DataFrame):
        if table_data.empty or not state.image_dir:
            return "No images loaded.", state
        table_data = table_data.values.tolist()
    elif not table_data or not state.image_dir:
        return "No images loaded.", state

    base = Path(state.image_dir)
    included = []
    excluded = []

    for row in table_data:
        name = row[0]
        is_included = row[6]
        if is_included:
            included.append(str(base / name))
        else:
            excluded.append(str(base / name))

    state.image_list = included
    state.excluded_images = excluded
    state.num_images = len(included)

    status = f"Table changes applied. {len(included)} images included, {len(excluded)} excluded."
    state.save()
    return status, state


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

def build_stage1_tab():
    """Build the Stage 1 Import & QC tab. Returns (tab, components dict)."""
    with gr.Tab("1. Import & QC") as tab:
        gr.Markdown("### Stage 1: Image Import & Quality Check")

        with gr.Row():
            folder_input = gr.Textbox(
                label="Image Folder Path",
                placeholder="F:/path/to/your/images",
                scale=4,
            )
            browse_btn = gr.Button("Browse...", scale=1)
            load_btn = gr.Button("Load Images", variant="primary", scale=1)

        with gr.Row():
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Image Thumbnails",
                    columns=4,
                    height=400,
                    object_fit="contain",
                )
            with gr.Column(scale=1):
                stats_box = gr.Textbox(label="Summary", lines=6, interactive=False)
                warnings_box = gr.Textbox(label="EXIF Warnings", lines=3, interactive=False)

        sharpness_summary = gr.HTML(value="", label="Sharpness Distribution")

        with gr.Row():
            threshold_slider = gr.Slider(
                minimum=0, maximum=500, value=200, step=5,
                label="Sharpness Threshold",
            )
            cull_btn = gr.Button("Auto-Cull Blurry Images")
            apply_btn = gr.Button("Apply Table Changes")

        cull_info = gr.Textbox(label="Cull Status", interactive=False)

        image_table = gr.Dataframe(
            headers=["Filename", "Sharpness", "Width", "Height", "Focal (mm)", "Camera", "Include"],
            datatype=["str", "number", "number", "number", "str", "str", "bool"],
            label="Image Quality Table",
            interactive=True,
            wrap=True,
        )

    return tab, {
        "folder_input": folder_input,
        "browse_btn": browse_btn,
        "load_btn": load_btn,
        "gallery": gallery,
        "stats_box": stats_box,
        "warnings_box": warnings_box,
        "sharpness_summary": sharpness_summary,
        "threshold_slider": threshold_slider,
        "cull_btn": cull_btn,
        "apply_btn": apply_btn,
        "cull_info": cull_info,
        "image_table": image_table,
    }


def wire_stage1(components: dict, state: gr.State):
    """Connect Stage 1 UI events to handler functions."""
    c = components

    load_outputs = [
        c["gallery"], c["image_table"], c["warnings_box"], c["stats_box"],
        c["sharpness_summary"], state,
    ]

    # Browse just sets the path — user clicks "Load Images" when ready
    c["browse_btn"].click(
        fn=_browse_folder,
        inputs=[],
        outputs=[c["folder_input"]],
    )

    c["load_btn"].click(
        fn=load_images,
        inputs=[c["folder_input"], state],
        outputs=load_outputs,
    )

    c["cull_btn"].click(
        fn=auto_cull,
        inputs=[c["image_table"], c["threshold_slider"], state],
        outputs=[c["image_table"], c["cull_info"], c["sharpness_summary"], state],
    )

    c["apply_btn"].click(
        fn=apply_table_changes,
        inputs=[c["image_table"], state],
        outputs=[c["cull_info"], state],
    )

    # Live threshold preview — update group counts when slider moves
    c["threshold_slider"].release(
        fn=_update_threshold_preview,
        inputs=[c["image_table"], c["threshold_slider"]],
        outputs=[c["sharpness_summary"]],
    )
