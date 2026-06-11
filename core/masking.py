"""Object masking engine: BiRefNet auto-segmentation, SAM 2.1 interactive correction, mask I/O.

BiRefNet produces soft alpha mattes (0-255 grayscale) for all images in a dataset.
SAM 2.1 provides point-prompt correction for individual images.
Mask I/O utilities handle saving, loading, RGBA compositing, and overlay generation.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BiRefNet model variant registry
# ---------------------------------------------------------------------------

# Maps display name → (HuggingFace model ID, input resolution)
# Ordered by recommended usage for studio object capture.
BIREFNET_VARIANTS = {
    "general": ("ZhengPeng7/BiRefNet", (1024, 1024)),
    "high-detail (DIS)": ("ZhengPeng7/BiRefNet-DIS5K", (1024, 1024)),
    "high-res SOD": ("ZhengPeng7/BiRefNet-HRSOD", (1024, 1024)),
    "massive": ("ZhengPeng7/BiRefNet-massive", (1024, 1024)),
    "matting": ("ZhengPeng7/BiRefNet-matting", (1024, 1024)),
    "HR matting (2048)": ("ZhengPeng7/BiRefNet_HR-matting", (2048, 2048)),
}

BIREFNET_DEFAULT = "general"


def _resolve_variant(variant: str) -> tuple[str, tuple[int, int]]:
    """Resolve a variant display name or HF model ID to (model_id, input_size)."""
    # Check if it's a display name
    if variant in BIREFNET_VARIANTS:
        return BIREFNET_VARIANTS[variant]
    # Check if it's a raw HF model ID
    for _name, (hf_id, size) in BIREFNET_VARIANTS.items():
        if variant == hf_id:
            return hf_id, size
    # Unknown variant — assume 1024 input
    return variant, (1024, 1024)


# ---------------------------------------------------------------------------
# Mask post-processing
# ---------------------------------------------------------------------------


def postprocess_mask(
    mask: np.ndarray,
    threshold: float = 0.0,
    erosion_px: int = 0,
) -> np.ndarray:
    """Apply threshold and morphological cleanup to a mask.

    Args:
        mask: Float32 array in [0, 1] range, or uint8 in [0, 255].
        threshold: Confidence threshold in [0, 1]. Pixels below this → 0.
                   Pixels at or above retain their value (soft edges preserved).
        erosion_px: Morphological erosion radius in pixels. 0 = no erosion.

    Returns:
        Processed uint8 mask in [0, 255].
    """
    # Normalize to float [0, 1] if needed
    if mask.dtype == np.uint8:
        mask_f = mask.astype(np.float32) / 255.0
    else:
        mask_f = mask.astype(np.float32)

    # Threshold: zero out low-confidence pixels, preserve values above
    if threshold > 0:
        mask_f[mask_f < threshold] = 0.0

    # Convert to uint8
    mask_u8 = (mask_f * 255).clip(0, 255).astype(np.uint8)

    # Morphological erosion
    if erosion_px > 0:
        kernel_size = erosion_px * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)

    return mask_u8


# ---------------------------------------------------------------------------
# Mask I/O utilities
# ---------------------------------------------------------------------------


def save_mask(mask_array: np.ndarray, path: str | Path, soft: bool = True):
    """Save mask as grayscale PNG.

    Args:
        mask_array: 2D array, expected range [0, 255] or [0, 1].
        path: Output path.
        soft: If True, preserve continuous values. If False, threshold at 128 to binary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to [0, 255] uint8
    if mask_array.dtype in (np.float32, np.float64):
        if mask_array.max() <= 1.0:
            mask_array = (mask_array * 255).clip(0, 255)
        mask_array = mask_array.astype(np.uint8)

    if not soft:
        mask_array = ((mask_array > 128) * 255).astype(np.uint8)

    cv2.imwrite(str(path), mask_array)


def load_mask(path: str | Path) -> np.ndarray:
    """Load grayscale PNG mask, return numpy array [H, W] in [0, 255] uint8.

    Raises:
        FileNotFoundError: If the mask file does not exist.
        ValueError: If the file cannot be read as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask image: {path}")

    return mask


def create_rgba_image(image_path: str | Path, mask_path: str | Path, output_path: str | Path):
    """Composite source image + mask into RGBA PNG with alpha channel.

    The RGB channels come from the source image, the alpha channel from the mask.
    """
    img = Image.open(str(image_path)).convert("RGB")
    mask = Image.open(str(mask_path)).convert("L")

    # Resize mask to match image if needed
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.LANCZOS)

    rgba = img.copy()
    rgba.putalpha(mask)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(str(output_path), "PNG")


def generate_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Generate visualization overlay: image with colored tint on masked regions.

    Args:
        image: RGB image array [H, W, 3], uint8.
        mask: Grayscale mask [H, W], uint8 (0=bg, 255=fg).
        alpha: Blending strength of the color overlay.
        color: RGB color for the mask tint.

    Returns:
        RGB overlay image [H, W, 3], uint8.
    """
    overlay = image.copy()

    # Create color layer
    color_layer = np.full_like(image, color, dtype=np.uint8)

    # Normalize mask to [0, 1] for blending
    mask_norm = mask.astype(np.float32) / 255.0
    if mask_norm.ndim == 2:
        mask_norm = mask_norm[:, :, np.newaxis]

    # Blend: where mask is foreground, tint with color
    overlay = (
        overlay * (1 - alpha * mask_norm) + color_layer * (alpha * mask_norm)
    ).clip(0, 255).astype(np.uint8)

    return overlay


# ---------------------------------------------------------------------------
# BiRefNet auto-masking
# ---------------------------------------------------------------------------

# Module-level model cache (not serializable — never put in gr.State)
_birefnet_model = None
_birefnet_transform = None
_birefnet_variant = None


def load_birefnet(
    variant: str = "general",
    device: str = "cuda",
) -> tuple:
    """Load BiRefNet model for object segmentation.

    Accepts either a display name ("general", "high-detail (DIS)", etc.)
    or a HuggingFace model ID. Caches at module level — subsequent calls
    with the same variant return the cached model instantly.

    Returns:
        (model, transform) tuple.
    """
    global _birefnet_model, _birefnet_transform, _birefnet_variant

    hf_id, input_size = _resolve_variant(variant)

    if _birefnet_model is not None and _birefnet_variant == hf_id:
        return _birefnet_model, _birefnet_transform

    # If switching variants, free the old model first
    if _birefnet_model is not None:
        unload_birefnet()

    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    logger.info("Loading BiRefNet model: %s (input %s)", hf_id, input_size)
    model = AutoModelForImageSegmentation.from_pretrained(
        hf_id, trust_remote_code=True
    )
    model = model.to(device).half().eval()

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    _birefnet_model = model
    _birefnet_transform = transform
    _birefnet_variant = hf_id

    logger.info("BiRefNet loaded (%s, input %s)", hf_id, input_size)
    return model, transform


def unload_birefnet():
    """Free BiRefNet model from GPU memory."""
    global _birefnet_model, _birefnet_transform, _birefnet_variant
    if _birefnet_model is not None:
        import torch
        del _birefnet_model
        _birefnet_model = None
        _birefnet_transform = None
        _birefnet_variant = None
        torch.cuda.empty_cache()
        logger.info("BiRefNet unloaded")


def run_birefnet_batch(
    model,
    transform,
    image_paths: list[str | Path],
    mask_dir: str | Path,
    device: str = "cuda",
    threshold: float = 0.0,
    erosion_px: int = 0,
    progress_cb=None,
) -> int:
    """Run BiRefNet inference on a batch of images, saving masks to mask_dir.

    Processes images one-at-a-time for predictable VRAM usage.

    Args:
        model: Loaded BiRefNet model.
        transform: Preprocessing transform.
        image_paths: List of source image paths.
        mask_dir: Output directory for mask PNGs.
        device: CUDA device string.
        threshold: Confidence threshold [0, 1]. Pixels below → 0.
        erosion_px: Morphological erosion radius in pixels.
        progress_cb: Optional callback(current, total) for progress reporting.

    Returns:
        Number of masks generated.
    """
    import torch
    import traceback as tb_module

    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    errors = []

    for i, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        try:
            # Load and preprocess
            pil_img = Image.open(str(img_path)).convert("RGB")
            orig_w, orig_h = pil_img.size

            input_tensor = transform(pil_img).unsqueeze(0).to(device).half()

            # Inference
            with torch.no_grad():
                preds = model(input_tensor)

            # Get the final prediction, apply sigmoid
            # .float() converts FP16→FP32 before numpy — cv2.resize doesn't support float16
            pred = preds[-1].sigmoid().squeeze().cpu().float().numpy()

            # Resize to original dimensions using PIL (more robust than cv2 for float arrays)
            pred_pil = Image.fromarray(pred)
            pred_pil = pred_pil.resize((orig_w, orig_h), Image.BILINEAR)
            pred_resized = np.array(pred_pil)

            # Post-process: threshold + erosion
            mask_uint8 = postprocess_mask(pred_resized, threshold, erosion_px)

            # Save
            out_path = mask_dir / f"{img_path.stem}.png"
            ok = cv2.imwrite(str(out_path), mask_uint8)
            if not ok:
                raise RuntimeError(f"cv2.imwrite returned False for {out_path}")
            count += 1

            if i == 0:
                logger.info(
                    "First mask stats: shape=%s, min=%d, max=%d, mean=%.1f, "
                    "threshold=%.2f, erosion=%dpx",
                    mask_uint8.shape, mask_uint8.min(), mask_uint8.max(),
                    mask_uint8.mean(), threshold, erosion_px,
                )

        except Exception as e:
            logger.error("BiRefNet failed on %s: %s\n%s", img_path.name, e, tb_module.format_exc())
            errors.append(f"{img_path.name}: {e}")

        if progress_cb is not None:
            progress_cb(i + 1, len(image_paths))

    if count == 0 and errors:
        raise RuntimeError(
            f"BiRefNet failed on all {len(errors)} images. First error: {errors[0]}"
        )

    return count


def run_birefnet_single(
    model,
    transform,
    image_path: str | Path,
    mask_dir: str | Path,
    device: str = "cuda",
    threshold: float = 0.0,
    erosion_px: int = 0,
) -> Optional[np.ndarray]:
    """Run BiRefNet on a single image. Returns the mask array or None on failure."""
    import torch

    img_path = Path(image_path)
    mask_dir = Path(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    try:
        pil_img = Image.open(str(img_path)).convert("RGB")
        orig_w, orig_h = pil_img.size

        input_tensor = transform(pil_img).unsqueeze(0).to(device).half()

        with torch.no_grad():
            preds = model(input_tensor)

        pred = preds[-1].sigmoid().squeeze().cpu().float().numpy()
        pred_pil = Image.fromarray(pred)
        pred_pil = pred_pil.resize((orig_w, orig_h), Image.BILINEAR)

        mask_uint8 = postprocess_mask(np.array(pred_pil), threshold, erosion_px)

        out_path = mask_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), mask_uint8)
        return mask_uint8

    except Exception as e:
        logger.error("BiRefNet single failed on %s: %s", img_path.name, e)
        return None


# ---------------------------------------------------------------------------
# SAM 2.1 interactive correction
# ---------------------------------------------------------------------------

# Module-level predictor cache
_sam2_predictor = None
_sam2_checkpoint = None
_sam2_current_image_path = None  # Track which image is set to avoid redundant set_image


def _check_sam2_available() -> bool:
    """Check if SAM 2.1 is installed."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return True
    except ImportError:
        return False


def _sam2_resolve_path(relative_path: str) -> str:
    """Resolve a SAM2 checkpoint/config path relative to the sam2 package root.

    build_sam2() resolves paths relative to CWD, which breaks when running
    from a different directory (e.g. apex-sfm/). This finds the sam2 package
    install location and builds an absolute path.
    """
    import sam2
    sam2_root = Path(sam2.__file__).parent.parent  # sam2/ package → repo root
    resolved = sam2_root / relative_path
    return str(resolved)


def load_sam2(
    checkpoint: str = "sam2.1_hiera_large",
    device: str = "cuda",
):
    """Load SAM 2.1 image predictor.

    Caches at module level. Returns the predictor or None if SAM 2.1 is not installed.
    """
    global _sam2_predictor, _sam2_checkpoint, _sam2_current_image_path

    if _sam2_predictor is not None and _sam2_checkpoint == checkpoint:
        return _sam2_predictor

    if not _check_sam2_available():
        logger.warning("SAM 2.1 not installed. Interactive correction unavailable.")
        return None

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    logger.info("Loading SAM 2.1: %s", checkpoint)

    # SAM 2.1 config/checkpoint mapping
    config_map = {
        "sam2.1_hiera_large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
        "sam2.1_hiera_base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
        "sam2.1_hiera_small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
        "sam2.1_hiera_tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
    }

    if checkpoint not in config_map:
        logger.error("Unknown SAM 2.1 checkpoint: %s", checkpoint)
        return None

    config_path, ckpt_path = config_map[checkpoint]
    ckpt_path = _sam2_resolve_path(ckpt_path)

    try:
        sam2_model = build_sam2(config_path, ckpt_path, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        _sam2_predictor = predictor
        _sam2_checkpoint = checkpoint
        _sam2_current_image_path = None
        logger.info("SAM 2.1 loaded: %s", checkpoint)
        return predictor
    except Exception as e:
        logger.error("Failed to load SAM 2.1: %s", e)
        return None


def unload_sam2():
    """Free SAM 2.1 predictor from GPU memory."""
    global _sam2_predictor, _sam2_checkpoint, _sam2_current_image_path
    if _sam2_predictor is not None:
        import torch
        del _sam2_predictor
        _sam2_predictor = None
        _sam2_checkpoint = None
        _sam2_current_image_path = None
        torch.cuda.empty_cache()
        logger.info("SAM 2.1 unloaded")


def predict_mask_sam2(
    predictor,
    image: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    image_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Run SAM 2.1 point-prompt prediction on an image.

    Args:
        predictor: SAM2ImagePredictor instance.
        image: RGB image array [H, W, 3], uint8.
        points: Nx2 array of (x, y) pixel coordinates.
        labels: N array of labels (1=foreground, 0=background).
        image_path: Optional path for caching set_image across calls.

    Returns:
        Binary mask [H, W], uint8 (0 or 255), or None on failure.
    """
    global _sam2_current_image_path

    if predictor is None:
        return None

    try:
        # Only call set_image if the image changed (expensive ~200ms)
        if image_path is None or image_path != _sam2_current_image_path:
            predictor.set_image(image)
            _sam2_current_image_path = image_path

        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )

        # Select the best mask (highest confidence)
        best_idx = scores.argmax()
        best_mask = masks[best_idx]

        # Convert boolean mask to uint8 [0, 255]
        return (best_mask.astype(np.uint8) * 255)

    except Exception as e:
        logger.warning("SAM 2.1 prediction failed: %s", e)
        return None


def sam2_available() -> bool:
    """Check if SAM 2.1 is installed and can be used."""
    return _check_sam2_available()


# ---------------------------------------------------------------------------
# SAM 2.1 automatic mask generation (segment-everything + pick)
# ---------------------------------------------------------------------------

# Module-level generator cache
_sam2_amg = None
_sam2_amg_checkpoint = None

# Distinct color palette for segment visualization (12 colors, cycling)
_SEGMENT_PALETTE = [
    (255, 100, 100),  # red
    (100, 100, 255),  # blue
    (255, 200, 50),   # yellow
    (200, 50, 255),   # purple
    (255, 150, 50),   # orange
    (50, 200, 200),   # cyan
    (255, 100, 200),  # pink
    (100, 200, 100),  # light green
    (200, 150, 100),  # tan
    (150, 100, 200),  # lavender
    (100, 200, 255),  # sky blue
    (200, 200, 100),  # lime
]


def load_sam2_amg(
    checkpoint: str = "sam2.1_hiera_large",
    device: str = "cuda",
):
    """Load SAM 2.1 automatic mask generator.

    Reuses the config_map from load_sam2(). Caches at module level.
    Unloads BiRefNet first to free VRAM.

    Returns:
        SAM2AutomaticMaskGenerator instance, or None on failure.
    """
    global _sam2_amg, _sam2_amg_checkpoint

    if _sam2_amg is not None and _sam2_amg_checkpoint == checkpoint:
        return _sam2_amg

    if not _check_sam2_available():
        logger.warning("SAM 2.1 not installed. Auto-segmentation unavailable.")
        return None

    # Unload BiRefNet to free VRAM — both models won't fit comfortably
    unload_birefnet()
    # Also unload the point-prompt predictor if loaded
    unload_sam2()

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    config_map = {
        "sam2.1_hiera_large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
        "sam2.1_hiera_base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
        "sam2.1_hiera_small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
        "sam2.1_hiera_tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
    }

    if checkpoint not in config_map:
        logger.error("Unknown SAM 2.1 checkpoint: %s", checkpoint)
        return None

    config_path, ckpt_path = config_map[checkpoint]
    ckpt_path = _sam2_resolve_path(ckpt_path)

    try:
        logger.info("Loading SAM 2.1 auto-mask generator: %s", checkpoint)
        sam2_model = build_sam2(config_path, ckpt_path, device=device)
        generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
        )
        _sam2_amg = generator
        _sam2_amg_checkpoint = checkpoint
        logger.info("SAM 2.1 auto-mask generator loaded: %s", checkpoint)
        return generator
    except Exception as e:
        logger.error("Failed to load SAM 2.1 auto-mask generator: %s", e)
        return None


def unload_sam2_amg():
    """Free SAM 2.1 auto-mask generator from GPU memory."""
    global _sam2_amg, _sam2_amg_checkpoint
    if _sam2_amg is not None:
        import torch
        del _sam2_amg
        _sam2_amg = None
        _sam2_amg_checkpoint = None
        torch.cuda.empty_cache()
        logger.info("SAM 2.1 auto-mask generator unloaded")


def run_sam2_auto_segments(generator, image_path: str | Path) -> list[dict]:
    """Run SAM 2.1 automatic mask generation on a single image.

    Args:
        generator: SAM2AutomaticMaskGenerator instance.
        image_path: Path to the source image.

    Returns:
        List of segment dicts with keys:
          segmentation (bool HxW), area, predicted_iou, stability_score, bbox.
        Filtered (area >= 500) and sorted by area descending.
    """
    image = np.array(Image.open(str(image_path)).convert("RGB"))
    raw_segments = generator.generate(image)

    # Filter tiny segments
    segments = [s for s in raw_segments if s["area"] >= 500]

    # Sort by area descending (object is typically among the largest)
    segments.sort(key=lambda s: s["area"], reverse=True)

    return segments


def run_sam2_auto_batch(
    generator,
    image_paths: list[str | Path],
    segments_dir: str | Path,
    progress_cb=None,
) -> int:
    """Run SAM 2.1 auto-segmentation on all images, saving segments to NPZ files.

    Args:
        generator: SAM2AutomaticMaskGenerator instance.
        image_paths: List of source image paths.
        segments_dir: Output directory for NPZ segment files.
        progress_cb: Optional callback(current, total) for progress reporting.

    Returns:
        Number of images processed successfully.
    """
    import traceback as tb_module

    segments_dir = Path(segments_dir)
    segments_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    errors = []

    for i, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        try:
            segments = run_sam2_auto_segments(generator, img_path)

            if segments:
                masks = np.stack([s["segmentation"] for s in segments])
                areas = np.array([s["area"] for s in segments], dtype=np.int64)
                ious = np.array([s["predicted_iou"] for s in segments], dtype=np.float32)
                stability = np.array([s["stability_score"] for s in segments], dtype=np.float32)

                out_path = segments_dir / f"{img_path.stem}.npz"
                np.savez_compressed(str(out_path),
                                    masks=masks, areas=areas,
                                    ious=ious, stability=stability)
            else:
                logger.warning("No segments found for %s", img_path.name)

            count += 1

            if i == 0:
                logger.info("First image segments: %d segments for %s",
                            len(segments), img_path.name)

        except Exception as e:
            logger.error("SAM auto-segment failed on %s: %s\n%s",
                         img_path.name, e, tb_module.format_exc())
            errors.append(f"{img_path.name}: {e}")

        if progress_cb is not None:
            progress_cb(i + 1, len(image_paths))

    if count == 0 and errors:
        raise RuntimeError(
            f"SAM auto-segment failed on all {len(errors)} images. "
            f"First error: {errors[0]}"
        )

    return count


def load_segments(segments_dir: str | Path, image_stem: str) -> list[dict] | None:
    """Load pre-computed segments from NPZ file.

    Args:
        segments_dir: Directory containing NPZ segment files.
        image_stem: Image filename stem (without extension).

    Returns:
        List of segment dicts with keys: segmentation, area, predicted_iou,
        stability_score. Returns None if file doesn't exist.
    """
    npz_path = Path(segments_dir) / f"{image_stem}.npz"
    if not npz_path.exists():
        return None

    data = np.load(str(npz_path))
    masks = data["masks"]      # (N, H, W) bool
    areas = data["areas"]      # (N,)
    ious = data["ious"]        # (N,)
    stability = data["stability"]  # (N,)

    segments = []
    for j in range(masks.shape[0]):
        segments.append({
            "segmentation": masks[j],
            "area": int(areas[j]),
            "predicted_iou": float(ious[j]),
            "stability_score": float(stability[j]),
        })

    return segments


def render_segment_map(
    image: np.ndarray,
    segments: list[dict],
    selected_indices: list[int],
) -> np.ndarray:
    """Render colorized segment overlay on image.

    Args:
        image: RGB image [H, W, 3], uint8.
        segments: List of segment dicts with 'segmentation' key.
        selected_indices: Indices of selected segments (bright green overlay).

    Returns:
        RGB overlay image [H, W, 3], uint8.
    """
    overlay = image.astype(np.float32).copy()
    selected_set = set(selected_indices)

    # Draw unselected segments first (low alpha), then selected (high alpha)
    for i, seg in enumerate(segments):
        mask = seg["segmentation"]
        # Resize mask if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        if i in selected_set:
            continue  # Draw selected segments on top

        # Unselected: semi-transparent colored fill
        color = _SEGMENT_PALETTE[i % len(_SEGMENT_PALETTE)]
        mask_3d = mask[:, :, np.newaxis]
        overlay = overlay * (1 - 0.2 * mask_3d) + np.array(color, dtype=np.float32) * (0.2 * mask_3d)

    # Draw selected segments on top with bright green
    for i in selected_indices:
        if i < 0 or i >= len(segments):
            continue
        mask = segments[i]["segmentation"]
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_3d = mask[:, :, np.newaxis]
        green = np.array([0, 255, 0], dtype=np.float32)
        overlay = overlay * (1 - 0.5 * mask_3d) + green * (0.5 * mask_3d)

    return overlay.clip(0, 255).astype(np.uint8)


def merge_selected_segments(
    segments: list[dict],
    selected_indices: list[int],
) -> np.ndarray:
    """OR all selected segment masks together into a single binary mask.

    Args:
        segments: List of segment dicts with 'segmentation' key.
        selected_indices: Indices to merge.

    Returns:
        uint8 mask [H, W] with values 0 or 255.
    """
    if not selected_indices or not segments:
        # Return empty mask matching first segment's shape
        if segments:
            h, w = segments[0]["segmentation"].shape[:2]
        else:
            h, w = 1, 1
        return np.zeros((h, w), dtype=np.uint8)

    merged = None
    for i in selected_indices:
        if i < 0 or i >= len(segments):
            continue
        mask = segments[i]["segmentation"]
        if merged is None:
            merged = mask.copy()
        else:
            merged = merged | mask

    if merged is None:
        h, w = segments[0]["segmentation"].shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    return (merged.astype(np.uint8) * 255)
