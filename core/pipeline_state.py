"""Pipeline state dataclass shared across all GUI tabs via gr.State.

IMPORTANT: Only store serializable types (paths as strings, primitives, lists, dicts).
Never store pycolmap objects — they cannot survive Gradio's deep copy of gr.State.

State is persisted to disk (pipeline_state.json) in the project directory so that
progress survives app restarts.
"""

import json
from dataclasses import asdict, dataclass, field
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

    # --- Stage 7: Export ---
    export_dir: Optional[str] = None  # Final 3DGS-ready directory

    # --- Stage 8 (future): 3DGS Training ---
    gaussian_model_path: Optional[str] = None
    training_iterations: int = 0
    final_loss: float = 0.0

    # --- Stage 9 (future): DiFix3D+ Enhancement ---
    difix_output_path: Optional[str] = None
    enhanced_model_path: Optional[str] = None
    num_views_enhanced: int = 0

    def save(self):
        """Persist state to project_dir/pipeline_state.json."""
        if not self.project_dir:
            return
        path = Path(self.project_dir) / STATE_FILENAME
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, project_dir: str) -> "PipelineState":
        """Restore state from project_dir/pipeline_state.json, or return fresh state."""
        path = Path(project_dir) / STATE_FILENAME
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except (json.JSONDecodeError, TypeError):
            return cls()
