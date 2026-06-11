"""Stage 8: Object Isolation & Masking.

Auto-segment objects with BiRefNet, then interactively correct
individual masks with SAM 2.1 point prompts. Masks are saved to
<project_dir>/masks/ as grayscale PNGs for 3DGS training.
"""

import logging
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from core.masking import (
    BIREFNET_DEFAULT,
    BIREFNET_VARIANTS,
    generate_overlay,
    load_birefnet,
    load_mask,
    load_sam2,
    load_sam2_amg,
    load_segments,
    merge_selected_segments,
    predict_mask_sam2,
    render_segment_map,
    run_birefnet_batch,
    run_birefnet_single,
    run_sam2_auto_batch,
    sam2_available,
    save_mask,
    unload_sam2_amg,
)
from core.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

# Max edge size for review display — Gradio will resize anyway
_REVIEW_MAX_EDGE = 1600

# Module-level cache for the currently viewed review image
_review_cache_path: str | None = None
_review_cache_img: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Helper: build overlay with point annotations
# ---------------------------------------------------------------------------

def _load_image_rgb(image_path: str) -> np.ndarray:
    """Load image as RGB, cached + downscaled to _REVIEW_MAX_EDGE for review."""
    global _review_cache_path, _review_cache_img

    if _review_cache_path == image_path and _review_cache_img is not None:
        return _review_cache_img

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Downscale for review display if larger than threshold
    h, w = img_rgb.shape[:2]
    max_edge = max(h, w)
    if max_edge > _REVIEW_MAX_EDGE:
        scale = _REVIEW_MAX_EDGE / max_edge
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    _review_cache_path = image_path
    _review_cache_img = img_rgb
    return img_rgb


def _render_overlay_with_points(
    image_path: str,
    mask_path: str | None,
    points: list[tuple[int, int, int]],
) -> np.ndarray:
    """Render image with mask overlay and annotated click points.

    Args:
        image_path: Path to source image.
        mask_path: Path to mask PNG, or None if no mask exists.
        points: List of (x, y, label) tuples. label: 1=fg, 0=bg.

    Returns:
        RGB overlay image with point annotations.
    """
    image = _load_image_rgb(image_path)

    if mask_path and Path(mask_path).exists():
        mask = load_mask(mask_path)
        # Resize mask if dimensions don't match
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        overlay = generate_overlay(image, mask)
    else:
        overlay = image.copy()

    # Draw click points
    for x, y, label in points:
        color = (0, 255, 0) if label == 1 else (255, 0, 0)  # green=fg, red=bg
        cv2.circle(overlay, (int(x), int(y)), 8, color, -1)
        cv2.circle(overlay, (int(x), int(y)), 8, (255, 255, 255), 2)

    return overlay


def _get_mask_path(state: PipelineState, image_path: str) -> str | None:
    """Get the mask path for a given source image."""
    if not state.mask_dir:
        return None
    stem = Path(image_path).stem
    mask_path = Path(state.mask_dir) / f"{stem}.png"
    return str(mask_path) if mask_path.exists() else None


def _get_image_at_index(state: PipelineState, index: int) -> str | None:
    """Get image path at the given index (0-based) from the state image list."""
    if not state.image_list or index < 0 or index >= len(state.image_list):
        return None
    return state.image_list[index]


def _load_thumb_for_gallery(
    img_path_str: str, thumb_size: int, thumbs_dir: Path | None,
) -> np.ndarray | None:
    """Load a thumbnail for gallery display: prefer disk cache, fallback to PIL draft mode."""
    img_path = Path(img_path_str)

    # Try the Stage 1 cached thumbnail first (~2ms)
    if thumbs_dir is not None:
        thumb_file = thumbs_dir / f"{img_path.stem}.jpg"
        if thumb_file.exists():
            t = cv2.imread(str(thumb_file))
            if t is not None:
                return cv2.cvtColor(t, cv2.COLOR_BGR2RGB)

    # Fallback: PIL draft mode for fast downscaled JPEG read
    try:
        from PIL import Image
        pil_img = Image.open(img_path_str)
        pil_img.draft("RGB", (thumb_size, thumb_size))
        pil_img = pil_img.resize(
            (thumb_size, int(thumb_size * pil_img.height / max(pil_img.width, 1))),
            Image.LANCZOS,
        )
        return np.array(pil_img.convert("RGB"))
    except Exception:
        pass

    # Last resort: full cv2 read
    img = cv2.imread(str(img_path_str))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = thumb_size / max(h, w)
    return cv2.resize(img_rgb, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def _build_gallery_items(state: PipelineState) -> list[tuple[np.ndarray, str]]:
    """Build gallery thumbnails showing mask overlays.

    Uses cached thumbnails from Stage 1's thumbs/ directory for speed.
    """
    items = []
    thumb_size = 192
    thumbs_dir = Path(state.project_dir) / "thumbs" if state.project_dir else None

    for img_path_str in state.image_list:
        img_path = Path(img_path_str)
        try:
            thumb = _load_thumb_for_gallery(img_path_str, thumb_size, thumbs_dir)
            if thumb is None:
                continue

            mask_path = _get_mask_path(state, img_path_str)
            if mask_path:
                mask = load_mask(mask_path)
                mask_thumb = cv2.resize(mask, (thumb.shape[1], thumb.shape[0]),
                                        interpolation=cv2.INTER_AREA)
                thumb = generate_overlay(thumb, mask_thumb)

            items.append((thumb, img_path.name))
        except Exception as e:
            logger.warning("Gallery thumbnail failed for %s: %s", img_path.name, e)

    return items


def _update_gallery_item(
    gallery: list[tuple[np.ndarray, str]],
    state: PipelineState,
    image_index: int,
) -> list[tuple[np.ndarray, str]]:
    """Update a single gallery item in-place after mask correction.

    Avoids rebuilding the entire gallery for single-image operations.
    """
    if not state.image_list or image_index < 0 or image_index >= len(state.image_list):
        return gallery

    img_path_str = state.image_list[image_index]
    img_path = Path(img_path_str)
    thumb_size = 192
    thumbs_dir = Path(state.project_dir) / "thumbs" if state.project_dir else None

    try:
        thumb = _load_thumb_for_gallery(img_path_str, thumb_size, thumbs_dir)
        if thumb is None:
            return gallery

        mask_path = _get_mask_path(state, img_path_str)
        if mask_path:
            mask = load_mask(mask_path)
            mask_thumb = cv2.resize(mask, (thumb.shape[1], thumb.shape[0]),
                                    interpolation=cv2.INTER_AREA)
            thumb = generate_overlay(thumb, mask_thumb)

        # Replace the item at the correct position
        if image_index < len(gallery):
            gallery[image_index] = (thumb, img_path.name)
    except Exception as e:
        logger.warning("Gallery update failed for %s: %s", img_path.name, e)

    return gallery


# ---------------------------------------------------------------------------
# Auto-mask handler
# ---------------------------------------------------------------------------

def run_auto_mask(
    birefnet_variant: str,
    threshold: float,
    erosion_px: int,
    state: PipelineState,
    progress=gr.Progress(),
):
    """Run BiRefNet auto-masking on all images."""
    if not state.image_list:
        raise gr.Error("No images loaded. Complete Stage 1 first.")
    if not state.project_dir:
        raise gr.Error("No project directory. Complete Stage 1 first.")

    threshold = float(threshold)
    erosion_px = int(erosion_px)

    mask_dir = str(Path(state.project_dir) / "masks")

    progress(0.05, desc="Loading BiRefNet model...")
    model, transform = load_birefnet(variant=birefnet_variant)

    def progress_cb(current, total):
        pct = 0.1 + 0.85 * (current / total)
        progress(pct, desc=f"Masking image {current}/{total}...")

    try:
        count = run_birefnet_batch(
            model, transform,
            state.image_list, mask_dir,
            threshold=threshold,
            erosion_px=erosion_px,
            progress_cb=progress_cb,
        )
    except RuntimeError as e:
        raise gr.Error(str(e))

    if count == 0:
        raise gr.Error("No masks were generated. Check the console log for errors.")

    progress(0.98, desc="Building gallery...")

    # Update state only on success
    hf_id = BIREFNET_VARIANTS.get(birefnet_variant, (birefnet_variant, None))[0]
    state.masks_generated = True
    state.mask_dir = mask_dir
    state.mask_method = "birefnet"
    state.num_masks = count
    state.num_corrected = 0
    state.mask_model_variant = hf_id
    state.save()

    stats = (
        f"Masks generated: {count} / {len(state.image_list)}\n"
        f"Model: {birefnet_variant}\n"
        f"Threshold: {threshold:.2f} | Erosion: {erosion_px}px\n"
        f"Output: {mask_dir}"
    )
    log = f"Auto-masking complete. {count} masks saved."

    # Build initial review image (first image)
    review_img = None
    if state.image_list:
        mask_path = _get_mask_path(state, state.image_list[0])
        review_img = _render_overlay_with_points(state.image_list[0], mask_path, [])

    gallery = _build_gallery_items(state)

    return stats, log, review_img, gallery, state


# ---------------------------------------------------------------------------
# Navigation handlers
# ---------------------------------------------------------------------------

def navigate_image(index: int, state: PipelineState):
    """Navigate to a specific image by slider index (1-based for user display).

    Segment-aware: if SAM segments exist for this image and no final mask
    has been saved yet, shows the segment map for interactive selection.
    """
    idx = int(index) - 1  # Convert to 0-based
    img_path = _get_image_at_index(state, idx)
    if img_path is None:
        return None, "", [], [], ""

    mask_path = _get_mask_path(state, img_path)
    name = Path(img_path).name

    # Check for SAM segments
    seg_dir = _get_segments_dir(state) if state.project_dir else None
    stem = Path(img_path).stem
    segments = load_segments(seg_dir, stem) if seg_dir else None

    if segments and not mask_path:
        # Segments exist but no final mask — show segment map for selection
        overlay = _render_segment_review(img_path, segments, [])
        info = f"{name}\nMask: NO — {len(segments)} segments available"
        segment_info = f"{len(segments)} segments. Click on the object to select."
        return overlay, info, [], [], segment_info
    else:
        # Normal view: mask overlay (or plain image)
        overlay = _render_overlay_with_points(img_path, mask_path, [])
        has_mask = "YES" if mask_path else "NO"
        info = f"{name}\nMask: {has_mask}"
        return overlay, info, [], [], ""


def navigate_prev(current_index: int, state: PipelineState):
    """Navigate to previous image."""
    new_idx = max(1, int(current_index) - 1)
    overlay, info, points, selected, seg_info = navigate_image(new_idx, state)
    return new_idx, overlay, info, points, selected, seg_info


def navigate_next(current_index: int, state: PipelineState):
    """Navigate to next image."""
    max_idx = len(state.image_list) if state.image_list else 1
    new_idx = min(max_idx, int(current_index) + 1)
    overlay, info, points, selected, seg_info = navigate_image(new_idx, state)
    return new_idx, overlay, info, points, selected, seg_info


# ---------------------------------------------------------------------------
# Click-to-correct handlers
# ---------------------------------------------------------------------------

def on_image_click(
    image_index: int,
    points_state: list,
    selected_segments_state: list,
    click_mode: str,
    state: PipelineState,
    evt: gr.SelectData,
):
    """Handle click on the review image — dispatches to segment or point mode.

    If SAM segments exist for this image and no final mask saved,
    routes to segment selection. Otherwise, uses point-prompt behavior.
    """
    if not state.image_list:
        return None, "", points_state, selected_segments_state, ""

    idx = int(image_index) - 1
    img_path = _get_image_at_index(state, idx)
    if img_path is None:
        return None, "", points_state, selected_segments_state, ""

    # Check if segment mode applies
    seg_dir = _get_segments_dir(state) if state.project_dir else None
    stem = Path(img_path).stem
    segments = load_segments(seg_dir, stem) if seg_dir else None
    mask_path = _get_mask_path(state, img_path)

    if segments and not mask_path:
        # Segment selection mode
        overlay, _, selected_segments_state, seg_info = on_segment_click(
            image_index, selected_segments_state, click_mode, state, evt,
        )
        return overlay, "", points_state, selected_segments_state, seg_info

    # Normal point-prompt mode
    x, y = evt.index
    label = 1 if click_mode == "Add foreground" else 0
    points_state.append((int(x), int(y), label))

    overlay = _render_overlay_with_points(img_path, mask_path, points_state)

    point_text = "\n".join(
        f"({'fg' if l else 'bg'}) ({x}, {y})" for x, y, l in points_state
    )

    return overlay, point_text, points_state, selected_segments_state, ""


def apply_sam_correction(
    image_index: int,
    points_state: list,
    mask_gallery: list,
    state: PipelineState,
):
    """Apply SAM 2.1 correction using collected points."""
    if not points_state:
        raise gr.Error("No points added. Click on the image to add foreground/background points.")

    if not sam2_available():
        raise gr.Error(
            "SAM 2.1 is not installed. Install it from source with SAM2_BUILD_CUDA=0.\n"
            "See the installation guide for details."
        )

    idx = int(image_index) - 1
    img_path = _get_image_at_index(state, idx)
    if img_path is None:
        raise gr.Error("Invalid image index.")

    # Load SAM 2.1
    predictor = load_sam2()
    if predictor is None:
        raise gr.Error("Failed to load SAM 2.1 predictor.")

    # Prepare image and points
    image = _load_image_rgb(img_path)
    points_array = np.array([(x, y) for x, y, _ in points_state], dtype=np.float32)
    labels_array = np.array([l for _, _, l in points_state], dtype=np.int32)

    # Run prediction
    mask = predict_mask_sam2(
        predictor, image, points_array, labels_array,
        image_path=img_path,
    )

    if mask is None:
        raise gr.Error("SAM 2.1 prediction failed. Try different points.")

    # Save corrected mask
    if not state.mask_dir:
        state.mask_dir = str(Path(state.project_dir) / "masks")
    mask_path = Path(state.mask_dir) / f"{Path(img_path).stem}.png"
    save_mask(mask, str(mask_path), soft=False)

    state.num_corrected += 1
    state.save()

    # Render updated overlay (clear points after apply)
    overlay = _render_overlay_with_points(img_path, str(mask_path), [])

    # Incremental gallery update — only replace the one changed thumbnail
    gallery = _update_gallery_item(list(mask_gallery or []), state, idx)

    log = f"Mask corrected for {Path(img_path).name} using {len(points_state)} points."

    return overlay, "", [], gallery, log, state


def clear_points(image_index: int, state: PipelineState):
    """Clear all collected points without changing the mask."""
    idx = int(image_index) - 1
    img_path = _get_image_at_index(state, idx)
    if img_path is None:
        return None, "", []

    mask_path = _get_mask_path(state, img_path)
    overlay = _render_overlay_with_points(img_path, mask_path, [])
    return overlay, "", []


# ---------------------------------------------------------------------------
# SAM 2.1 auto-segment handlers
# ---------------------------------------------------------------------------

def _get_segments_dir(state: PipelineState) -> str:
    """Return path to sam_segments directory for the current project."""
    return str(Path(state.project_dir) / "sam_segments")


def _has_segments(state: PipelineState, image_path: str) -> bool:
    """Check if SAM segments exist for the given image."""
    seg_dir = _get_segments_dir(state)
    stem = Path(image_path).stem
    return Path(seg_dir).joinpath(f"{stem}.npz").exists()


def _render_segment_review(
    image_path: str,
    segments: list[dict],
    selected_indices: list[int],
) -> np.ndarray:
    """Load image and render segment map overlay for review display."""
    image = _load_image_rgb(image_path)

    # Resize segment masks to match the downscaled review image
    review_h, review_w = image.shape[:2]
    resized_segments = []
    for seg in segments:
        mask = seg["segmentation"]
        if mask.shape[:2] != (review_h, review_w):
            resized_mask = cv2.resize(
                mask.astype(np.uint8), (review_w, review_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            resized_mask = mask
        resized_segments.append({**seg, "segmentation": resized_mask})

    return render_segment_map(image, resized_segments, selected_indices)


def run_sam_auto_segment(state: PipelineState, progress=gr.Progress()):
    """Run SAM 2.1 auto-segmentation on all images.

    Generates all possible segments per image, stored for interactive selection.
    Does NOT save final masks — only stores segments for the user to pick from.
    """
    if not state.image_list:
        raise gr.Error("No images loaded. Complete Stage 1 first.")
    if not state.project_dir:
        raise gr.Error("No project directory. Complete Stage 1 first.")

    if not sam2_available():
        raise gr.Error(
            "SAM 2.1 is not installed. Install it from source with SAM2_BUILD_CUDA=0.\n"
            "See the installation guide for details."
        )

    progress(0.05, desc="Loading SAM 2.1 auto-mask generator...")
    generator = load_sam2_amg()
    if generator is None:
        raise gr.Error("Failed to load SAM 2.1 auto-mask generator.")

    segments_dir = _get_segments_dir(state)

    def progress_cb(current, total):
        pct = 0.1 + 0.85 * (current / total)
        progress(pct, desc=f"Auto-segmenting image {current}/{total}...")

    try:
        count = run_sam2_auto_batch(
            generator, state.image_list, segments_dir,
            progress_cb=progress_cb,
        )
    except RuntimeError as e:
        raise gr.Error(str(e))

    if count == 0:
        raise gr.Error("No images were segmented. Check the console log for errors.")

    progress(0.98, desc="Preparing review...")

    # Unload generator to free VRAM (user will interact now)
    unload_sam2_amg()

    # Update state
    state.mask_method = "sam2_auto"
    if not state.mask_dir:
        state.mask_dir = str(Path(state.project_dir) / "masks")
    state.save()

    stats = (
        f"SAM auto-segmentation complete: {count} / {len(state.image_list)} images\n"
        f"Segments saved to: {segments_dir}\n"
        f"Click on segments to select, then 'Save Selected Mask'."
    )
    log = f"SAM auto-segmentation done. {count} images processed. Select segments to build masks."

    # Show first image's segment map
    review_img = None
    selected = []
    segment_info = ""
    if state.image_list:
        stem = Path(state.image_list[0]).stem
        segments = load_segments(segments_dir, stem)
        if segments:
            review_img = _render_segment_review(state.image_list[0], segments, [])
            segment_info = f"{len(segments)} segments found. Click to select."

    gallery = _build_gallery_items(state)

    return stats, log, review_img, gallery, selected, segment_info, state


def on_segment_click(
    image_index: int,
    selected_segments: list,
    click_mode: str,
    state: PipelineState,
    evt: gr.SelectData,
):
    """Handle click on review image when SAM segments exist.

    Toggles segment selection: click on a segment to select/deselect.
    If no segments exist for this image, falls back to the point-prompt behavior.
    """
    if not state.image_list:
        return None, "", selected_segments, ""

    idx = int(image_index) - 1
    img_path = _get_image_at_index(state, idx)
    if img_path is None:
        return None, "", selected_segments, ""

    # Check if SAM segments exist for this image
    seg_dir = _get_segments_dir(state)
    stem = Path(img_path).stem
    segments = load_segments(seg_dir, stem)

    if segments is None:
        # No segments — this shouldn't happen in segment mode, but handle gracefully
        return None, "", selected_segments, "No segments for this image."

    x, y = evt.index

    # Find which segment contains the clicked pixel
    # Check smallest-area segments first for precision (user wants to pick specific parts)
    # Segments are stored sorted by area descending, so iterate in reverse for smallest-first
    clicked_idx = None
    image = _load_image_rgb(img_path)
    review_h, review_w = image.shape[:2]

    for i in reversed(range(len(segments))):
        mask = segments[i]["segmentation"]
        # Resize mask to match review display coordinates
        if mask.shape[:2] != (review_h, review_w):
            mask = cv2.resize(
                mask.astype(np.uint8), (review_w, review_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            clicked_idx = i
            break

    if clicked_idx is None:
        # Clicked outside any segment
        overlay = _render_segment_review(img_path, segments, selected_segments)
        return overlay, "", selected_segments, _segment_info_text(segments, selected_segments)

    # Toggle: deselect if already selected, otherwise select
    if clicked_idx in selected_segments:
        selected_segments.remove(clicked_idx)
    else:
        selected_segments.append(clicked_idx)

    # Re-render with updated selection
    overlay = _render_segment_review(img_path, segments, selected_segments)
    info = _segment_info_text(segments, selected_segments)

    return overlay, "", selected_segments, info


def _segment_info_text(segments: list[dict], selected_indices: list[int]) -> str:
    """Build info text about current segment selection."""
    if not selected_indices:
        return f"{len(segments)} segments. Click on the object to select."

    total_area = sum(
        segments[i]["area"] for i in selected_indices
        if 0 <= i < len(segments)
    )
    # Approximate coverage as percentage of image
    if segments:
        h, w = segments[0]["segmentation"].shape[:2]
        image_area = h * w
        pct = (total_area / image_area) * 100 if image_area > 0 else 0
    else:
        pct = 0

    return f"{len(selected_indices)} segments selected ({pct:.1f}% coverage)"


def save_segment_selection(
    image_index: int,
    selected_segments: list,
    mask_gallery: list,
    state: PipelineState,
):
    """Save merged selected segments as the final mask, then auto-advance."""
    if not selected_segments:
        raise gr.Error("No segments selected. Click on segments first.")

    idx = int(image_index) - 1
    img_path = _get_image_at_index(state, idx)
    if img_path is None:
        raise gr.Error("Invalid image index.")

    seg_dir = _get_segments_dir(state)
    stem = Path(img_path).stem
    segments = load_segments(seg_dir, stem)
    if segments is None:
        raise gr.Error(f"No segments found for {stem}.")

    # Merge selected segments into one mask
    mask = merge_selected_segments(segments, selected_segments)

    # Save to masks directory
    if not state.mask_dir:
        state.mask_dir = str(Path(state.project_dir) / "masks")
    mask_path = Path(state.mask_dir) / f"{stem}.png"
    save_mask(mask, str(mask_path), soft=False)

    # Update mask count
    mask_files = list(Path(state.mask_dir).glob("*.png"))
    state.num_masks = len(mask_files)
    state.save()

    # Incremental gallery update
    gallery = _update_gallery_item(list(mask_gallery or []), state, idx)

    log = f"Mask saved for {Path(img_path).name} ({len(selected_segments)} segments merged)."

    # Auto-advance to next image
    max_idx = len(state.image_list) if state.image_list else 1
    next_idx_1based = min(max_idx, idx + 2)  # +2 because idx is 0-based, slider is 1-based
    next_idx_0based = next_idx_1based - 1

    # Prepare next image's review
    next_img_path = _get_image_at_index(state, next_idx_0based)
    new_selected = []
    segment_info = ""
    if next_img_path:
        next_stem = Path(next_img_path).stem
        next_segments = load_segments(seg_dir, next_stem)
        if next_segments:
            # Show segment map for next image
            review_img = _render_segment_review(next_img_path, next_segments, [])
            segment_info = f"{len(next_segments)} segments. Click on the object to select."
        else:
            # No segments, show normal overlay
            mask_p = _get_mask_path(state, next_img_path)
            review_img = _render_overlay_with_points(next_img_path, mask_p, [])
    else:
        review_img = None

    return (
        next_idx_1based,   # image_slider
        review_img,        # review_image
        "",                # point_list
        [],                # points_state
        new_selected,      # selected_segments_state
        gallery,           # mask_gallery
        log,               # masking_log
        segment_info,      # segment_info
        state,             # state
    )


def reset_to_auto_mask(
    image_index: int, threshold: float, erosion_px: int,
    mask_gallery: list, state: PipelineState,
):
    """Re-run BiRefNet on a single image to reset its mask."""
    idx = int(image_index) - 1
    img_path = _get_image_at_index(state, idx)
    if img_path is None:
        raise gr.Error("Invalid image index.")

    if not state.mask_dir:
        state.mask_dir = str(Path(state.project_dir) / "masks")

    variant = state.mask_model_variant or BIREFNET_DEFAULT
    model, transform = load_birefnet(variant=variant)
    mask = run_birefnet_single(
        model, transform, img_path, state.mask_dir,
        threshold=float(threshold), erosion_px=int(erosion_px),
    )

    if mask is None:
        raise gr.Error(f"BiRefNet failed on {Path(img_path).name}.")

    mask_path = str(Path(state.mask_dir) / f"{Path(img_path).stem}.png")
    overlay = _render_overlay_with_points(img_path, mask_path, [])

    # Incremental gallery update
    gallery = _update_gallery_item(list(mask_gallery or []), state, idx)

    log = f"Reset mask for {Path(img_path).name} to auto-mask."

    state.save()
    return overlay, "", [], gallery, log, state


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

def build_stage8_tab():
    """Build the Stage 8 Object Masking tab. Returns (tab, components dict)."""
    with gr.Tab("8. Object Masking") as tab:
        gr.Markdown("### Stage 8: Object Isolation")
        gr.Markdown(
            "Auto-segment objects with BiRefNet, then click to correct with SAM 2.1. "
            "'general' is best for solid objects on surfaces. Use 'matting' variants for "
            "translucent or furry objects."
        )

        with gr.Row():
            birefnet_variant = gr.Dropdown(
                choices=list(BIREFNET_VARIANTS.keys()),
                value=BIREFNET_DEFAULT,
                label="Model Variant",
                scale=2,
            )
            threshold_slider = gr.Slider(
                minimum=0.0, maximum=0.95, value=0.7, step=0.05,
                label="Confidence Threshold",
                scale=2,
            )
            erosion_slider = gr.Slider(
                minimum=0, maximum=20, value=0, step=1,
                label="Edge Erosion (px)",
                scale=1,
            )
            auto_mask_btn = gr.Button("Auto-Mask All Images", variant="primary", scale=2)

        with gr.Row():
            masking_stats = gr.Textbox(label="Masking Stats", lines=4, interactive=False)
            masking_log = gr.Textbox(label="Log", lines=4, interactive=False)

        gr.Markdown("---")
        gr.Markdown("#### Alternative: SAM 2.1 Auto-Segment")
        gr.Markdown(
            "Generate ALL segments per image, then click to select which ones "
            "are the object. Gives you direct control over what's included."
        )
        with gr.Row():
            sam_auto_btn = gr.Button(
                "SAM Auto-Segment All", variant="secondary", scale=2,
            )
            segment_info = gr.Textbox(
                label="Segment Selection",
                lines=1, interactive=False, scale=2,
            )
            save_selection_btn = gr.Button(
                "Save Selected Mask", variant="primary", scale=2,
            )

        gr.Markdown("---")
        gr.Markdown("#### Review & Correct")

        with gr.Row():
            image_slider = gr.Slider(
                minimum=1, maximum=1, value=1, step=1,
                label="Image", scale=3,
            )
            prev_btn = gr.Button("Prev", scale=1)
            next_btn = gr.Button("Next", scale=1)
            click_mode = gr.Dropdown(
                choices=["Add foreground", "Remove background"],
                value="Add foreground",
                label="Click Mode",
                scale=2,
            )

        with gr.Row():
            with gr.Column(scale=3):
                review_image = gr.Image(
                    label="Click to add/remove points",
                    interactive=False,
                    type="numpy",
                )
            with gr.Column(scale=1):
                point_list = gr.Textbox(
                    label="Collected Points", lines=8, interactive=False,
                )
                apply_sam_btn = gr.Button("Apply SAM Correction", variant="primary")
                clear_points_btn = gr.Button("Clear Points")
                reset_mask_btn = gr.Button("Reset to Auto-Mask")

        with gr.Row():
            mask_gallery = gr.Gallery(
                label="Mask Overview",
                columns=8,
                height=150,
                object_fit="contain",
            )

        # Hidden state for click points: list of (x, y, label)
        points_state = gr.State([])
        # Hidden state for selected segment indices: list of int
        selected_segments_state = gr.State([])

    return tab, {
        "birefnet_variant": birefnet_variant,
        "threshold_slider": threshold_slider,
        "erosion_slider": erosion_slider,
        "auto_mask_btn": auto_mask_btn,
        "masking_stats": masking_stats,
        "masking_log": masking_log,
        "sam_auto_btn": sam_auto_btn,
        "segment_info": segment_info,
        "save_selection_btn": save_selection_btn,
        "image_slider": image_slider,
        "prev_btn": prev_btn,
        "next_btn": next_btn,
        "click_mode": click_mode,
        "review_image": review_image,
        "point_list": point_list,
        "apply_sam_btn": apply_sam_btn,
        "clear_points_btn": clear_points_btn,
        "reset_mask_btn": reset_mask_btn,
        "mask_gallery": mask_gallery,
        "points_state": points_state,
        "selected_segments_state": selected_segments_state,
    }


def wire_stage8(components: dict, state: gr.State):
    """Connect Stage 8 UI events to handler functions."""
    c = components

    # Auto-mask all images (BiRefNet)
    c["auto_mask_btn"].click(
        fn=run_auto_mask,
        inputs=[
            c["birefnet_variant"], c["threshold_slider"],
            c["erosion_slider"], state,
        ],
        outputs=[
            c["masking_stats"], c["masking_log"],
            c["review_image"], c["mask_gallery"],
            state,
        ],
    ).then(
        fn=lambda s: gr.update(maximum=max(len(s.image_list), 1), value=1),
        inputs=[state],
        outputs=[c["image_slider"]],
    )

    # SAM 2.1 auto-segment all images
    c["sam_auto_btn"].click(
        fn=run_sam_auto_segment,
        inputs=[state],
        outputs=[
            c["masking_stats"], c["masking_log"],
            c["review_image"], c["mask_gallery"],
            c["selected_segments_state"], c["segment_info"],
            state,
        ],
    ).then(
        fn=lambda s: gr.update(maximum=max(len(s.image_list), 1), value=1),
        inputs=[state],
        outputs=[c["image_slider"]],
    )

    # Save selected segments as mask
    c["save_selection_btn"].click(
        fn=save_segment_selection,
        inputs=[c["image_slider"], c["selected_segments_state"],
                c["mask_gallery"], state],
        outputs=[
            c["image_slider"], c["review_image"], c["point_list"],
            c["points_state"], c["selected_segments_state"],
            c["mask_gallery"], c["masking_log"], c["segment_info"],
            state,
        ],
    )

    # Navigation: slider change
    c["image_slider"].release(
        fn=navigate_image,
        inputs=[c["image_slider"], state],
        outputs=[c["review_image"], c["masking_log"], c["points_state"],
                 c["selected_segments_state"], c["segment_info"]],
    )

    # Navigation: prev/next buttons
    c["prev_btn"].click(
        fn=navigate_prev,
        inputs=[c["image_slider"], state],
        outputs=[c["image_slider"], c["review_image"], c["masking_log"],
                 c["points_state"], c["selected_segments_state"], c["segment_info"]],
    )

    c["next_btn"].click(
        fn=navigate_next,
        inputs=[c["image_slider"], state],
        outputs=[c["image_slider"], c["review_image"], c["masking_log"],
                 c["points_state"], c["selected_segments_state"], c["segment_info"]],
    )

    # Click on image — mode-aware dispatch (segments vs point-prompts)
    c["review_image"].select(
        fn=on_image_click,
        inputs=[c["image_slider"], c["points_state"],
                c["selected_segments_state"], c["click_mode"], state],
        outputs=[c["review_image"], c["point_list"], c["points_state"],
                 c["selected_segments_state"], c["segment_info"]],
    )

    # Apply SAM 2.1 point-prompt correction
    c["apply_sam_btn"].click(
        fn=apply_sam_correction,
        inputs=[c["image_slider"], c["points_state"], c["mask_gallery"], state],
        outputs=[
            c["review_image"], c["point_list"], c["points_state"],
            c["mask_gallery"], c["masking_log"], state,
        ],
    )

    # Clear points
    c["clear_points_btn"].click(
        fn=clear_points,
        inputs=[c["image_slider"], state],
        outputs=[c["review_image"], c["point_list"], c["points_state"]],
    )

    # Reset to auto-mask
    c["reset_mask_btn"].click(
        fn=reset_to_auto_mask,
        inputs=[c["image_slider"], c["threshold_slider"], c["erosion_slider"],
                c["mask_gallery"], state],
        outputs=[
            c["review_image"], c["point_list"], c["points_state"],
            c["mask_gallery"], c["masking_log"], state,
        ],
    )
