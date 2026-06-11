"""Pipeline state dataclass shared across all GUI tabs via gr.State.

IMPORTANT: Only store serializable types (paths as strings, primitives, lists, dicts).
Never store pycolmap objects — they cannot survive Gradio's deep copy of gr.State.

State is persisted to disk (pipeline_state.json) in the project directory so that
progress survives app restarts.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

STATE_FILENAME = "pipeline_state.json"


@dataclass
class PipelineState:
    """Holds all outputs and paths produced by each pipeline stage."""

    # --- Project ---
    project_dir: Optional[str] = None  # Root working directory for this run

    # --- Stage 1: Import & QC ---
    image_dir: Optional[str] = None  # Folder containing source images
    image_list: list[str] = field(default_factory=list)  # Paths of accepted images
    excluded_images: list[str] = field(default_factory=list)
    num_images: int = 0
    image_width: int = 0
    image_height: int = 0
    focal_length_mm: Optional[float] = None

    # --- Stage 2: Feature Extraction ---
    feature_path: Optional[str] = None  # HDF5 path from HLoc
    extractor_name: Optional[str] = None
    max_keypoints_used: int = 0
    avg_keypoints: float = 0.0

    # --- Stage 3: Feature Matching ---
    match_path: Optional[str] = None  # HDF5 path from HLoc
    pairs_path: Optional[str] = None  # Pairs txt file
    matcher_name: Optional[str] = None
    num_pairs: int = 0
    avg_matches_per_pair: float = 0.0

    # --- Stages 4-5: Reconstruction ---
    sfm_dir: Optional[str] = None  # Directory containing cameras/images/points3D.bin
    num_registered: int = 0
    num_points3d: int = 0
    mean_reprojection_error: float = 0.0

    # --- Stage 6: Refinement ---
    refined: bool = False
    pre_ba_error: float = 0.0
    post_ba_error: float = 0.0

    # --- Alignment (applied during export) ---
    aligned: bool = False
    alignment_method: Optional[str] = None       # "imu" | "estimated" | None
    alignment_imu_coverage: float = 0.0
    alignment_rotation_deg: float = 0.0
    alignment_up_axis: Optional[str] = None      # "y" or "z"
    alignment_centered: bool = False

    # --- Masking (applied during export) ---
    masks_generated: bool = False
    mask_dir: Optional[str] = None           # <project_dir>/masks/
    mask_method: Optional[str] = None        # "birefnet" | "sam2" | "manual"
    num_masks: int = 0
    num_corrected: int = 0                   # masks manually corrected via SAM
    mask_model_variant: Optional[str] = None # e.g. "BiRefNet_HR-matting"

    # --- Stage 7: Export ---
    export_dir: Optional[str] = None  # Final 3DGS-ready directory

    # --- Stage 9: 3DGS Training ---
    training_backend: Optional[str] = None         # "gsplat", "2dgs", etc.
    training_dir: Optional[str] = None             # <project_dir>/training/<backend>/
    training_data_dir: Optional[str] = None        # Prepared data directory
    training_checkpoint: Optional[str] = None      # Path to checkpoint directory
    training_ply: Optional[str] = None             # Path to exported PLY
    training_iterations: int = 0                   # Steps completed
    training_psnr: float = 0.0                     # Final PSNR on test set
    training_ssim: float = 0.0                     # Final SSIM
    training_loss: float = 0.0                     # Final training loss
    training_status: Optional[str] = None          # "prepared", "running", "completed", "failed", "stopped"

    # --- Stage 10: Enhancement ---
    enhancement_backend: Optional[str] = None      # "fixer", "difix3d+", etc.
    enhancement_dir: Optional[str] = None          # Output directory
    num_views_enhanced: int = 0

    def discover_outputs(self):
        """Scan project_dir for existing pipeline outputs and fill missing fields.

        Handles stale/incomplete pipeline_state.json by looking for known output
        file patterns on disk. Only fills fields that are currently None — never
        overwrites existing values. Safe to call multiple times.
        """
        if not self.project_dir:
            return
        project_dir = Path(self.project_dir).resolve()
        if not project_dir.is_dir():
            return

        # Feature files: feats-*.h5
        if not self.feature_path:
            feats = sorted(project_dir.glob("feats-*.h5"))
            if feats:
                self.feature_path = str(feats[-1])
                parts = feats[-1].stem.split("-")
                if len(parts) >= 2:
                    self.extractor_name = parts[1]

        # Pairs files: pairs-*.txt
        if not self.pairs_path:
            pairs = sorted(project_dir.glob("pairs-*.txt"))
            if pairs:
                self.pairs_path = str(pairs[-1])

        # Match files: *matches*.h5
        if not self.match_path:
            matches = [f for f in project_dir.glob("*matches*.h5")]
            if matches:
                self.match_path = str(sorted(matches)[-1])

        # SFM directory: sfm/ with reconstruction .bin files
        if not self.sfm_dir:
            sfm_dir = project_dir / "sfm"
            if sfm_dir.is_dir() and any(sfm_dir.glob("*.bin")):
                self.sfm_dir = str(sfm_dir)

        # Masks directory: masks/ with .png files
        if not self.mask_dir:
            mask_dir = project_dir / "masks"
            if mask_dir.is_dir():
                mask_files = list(mask_dir.glob("*.png"))
                if mask_files:
                    self.mask_dir = str(mask_dir)
                    self.masks_generated = True
                    self.num_masks = len(mask_files)

        # Training directory: training/<backend>/output/ckpts/
        if not self.training_checkpoint:
            training_dir = project_dir / "training"
            if training_dir.is_dir():
                for backend_dir in training_dir.iterdir():
                    if not backend_dir.is_dir():
                        continue
                    ckpt_dir = backend_dir / "output" / "ckpts"
                    if ckpt_dir.is_dir() and any(ckpt_dir.glob("*.pt")):
                        self.training_backend = backend_dir.name
                        self.training_dir = str(backend_dir)
                        self.training_checkpoint = str(ckpt_dir)
                        self.training_status = "completed"
                        # Look for PLY
                        ply_dir = backend_dir / "output" / "ply"
                        if ply_dir.is_dir():
                            plys = sorted(ply_dir.glob("*.ply"))
                            if plys:
                                self.training_ply = str(plys[-1])
                        break

    def save(self):
        """Persist state to project_dir/pipeline_state.json."""
        if not self.project_dir:
            return
        path = Path(self.project_dir) / STATE_FILENAME
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, project_dir: str) -> "PipelineState":
        """Restore state from project_dir/pipeline_state.json, or return fresh state.

        Filters out unknown keys so old state files with removed/renamed fields
        don't cause TypeError. New fields get their defaults.
        """
        path = Path(project_dir) / STATE_FILENAME
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            valid_keys = {f.name for f in fields(cls)}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return cls(**filtered)
        except (json.JSONDecodeError, TypeError):
            return cls()
