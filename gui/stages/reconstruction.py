"""Stages 4-5: Geometric Verification & Reconstruction.

Uses HLoc's reconstruction.main() (verification + incremental SfM), or the
COLMAP 4.0 global mapper (merged GLOMAP) as an A/B alternative.
Presented as a single "Reconstruction" tab.
"""

import shutil
from pathlib import Path

import gradio as gr
import pycolmap

from core.pipeline_state import PipelineState
from core.utils import get_reconstruction_stats


def _run_global_reconstruction(
    sfm_dir: Path,
    image_dir: Path,
    pairs_path: Path,
    feature_path: Path,
    match_path: Path,
    camera_mode,
    camera_model_name: str,
    min_num_matches: int,
    ba_iters: int,
):
    """COLMAP 4.0 global mapping (GLOMAP) variant of hloc.reconstruction.main.

    Reuses HLoc's database import + geometric verification, then runs
    pycolmap.global_mapping instead of incremental mapping.
    """
    from hloc.reconstruction import create_empty_db, get_image_ids, import_images
    from hloc.triangulation import (
        estimation_and_geometric_verification,
        import_features,
        import_matches,
    )

    sfm_dir.mkdir(parents=True, exist_ok=True)
    database = sfm_dir / "database.db"

    create_empty_db(database)
    import_images(image_dir, database, camera_mode, None,
                  {"camera_model": camera_model_name})
    image_ids = get_image_ids(database)
    with pycolmap.Database.open(database) as db:
        import_features(image_ids, db, feature_path)
        import_matches(image_ids, db, pairs_path, match_path, None, False)
    estimation_and_geometric_verification(database, pairs_path)

    options = pycolmap.GlobalPipelineOptions()
    options.min_num_matches = int(min_num_matches)
    options.mapper.ba_num_iterations = int(ba_iters)

    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    reconstructions = pycolmap.global_mapping(
        database, image_dir, models_path, options,
    )
    if not reconstructions:
        return None

    largest = max(reconstructions, key=lambda i: reconstructions[i].num_reg_images())
    for filename in ["images.bin", "cameras.bin", "points3D.bin",
                     "frames.bin", "rigs.bin"]:
        src = models_path / str(largest) / filename
        if src.exists():
            if (sfm_dir / filename).exists():
                (sfm_dir / filename).unlink()
            shutil.move(str(src), str(sfm_dir))
    return reconstructions[largest]


def run_reconstruction(
    mapper: str,
    camera_model: str,
    single_camera: bool,
    min_num_matches: int,
    ba_local_iters: int,
    ba_global_iters: int,
    refine_focal: bool,
    refine_principal: bool,
    refine_extra: bool,
    state: PipelineState,
    progress=gr.Progress(),
):
    """Run reconstruction: HLoc incremental SfM or COLMAP 4 global mapping."""
    state.discover_outputs()
    if not state.match_path or not state.feature_path:
        raise gr.Error("No matches found. Complete Stage 3 first.")

    from hloc import reconstruction

    project_dir = Path(state.project_dir)
    image_dir = Path(state.image_dir)
    feature_path = Path(state.feature_path)
    match_path = Path(state.match_path)
    pairs_path = Path(state.pairs_path)
    sfm_dir = project_dir / "sfm"

    progress(0.1, desc="Setting up reconstruction...")

    # Camera mode
    if single_camera:
        camera_mode = pycolmap.CameraMode.SINGLE
    else:
        camera_mode = pycolmap.CameraMode.AUTO

    # Mapper options
    mapper_options = {
        "min_num_matches": int(min_num_matches),
        "ba_local_max_num_iterations": int(ba_local_iters),
        "ba_global_max_num_iterations": int(ba_global_iters),
        "ba_refine_focal_length": refine_focal,
        "ba_refine_principal_point": refine_principal,
        "ba_refine_extra_params": refine_extra,
    }

    use_global = mapper.startswith("global")
    progress(0.2, desc=f"Running {'global' if use_global else 'incremental'} "
                       "SfM (this may take a while)...")

    try:
        if use_global:
            model = _run_global_reconstruction(
                sfm_dir=sfm_dir,
                image_dir=image_dir,
                pairs_path=pairs_path,
                feature_path=feature_path,
                match_path=match_path,
                camera_mode=camera_mode,
                camera_model_name=camera_model,
                min_num_matches=min_num_matches,
                ba_iters=ba_global_iters,
            )
            if model is None:
                raise gr.Error("Global mapping could not reconstruct any model.")
        else:
            # Geometric verification + incremental mapping via HLoc
            model = reconstruction.main(
                sfm_dir=sfm_dir,
                image_dir=image_dir,
                pairs=pairs_path,
                features=feature_path,
                matches=match_path,
                camera_mode=camera_mode,
                image_options={"camera_model": camera_model},
                mapper_options=mapper_options,
            )
    except gr.Error:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise gr.Error(f"Reconstruction failed: {e}\n\n{tb}")

    progress(0.9, desc="Gathering statistics...")

    # Get stats from the saved reconstruction (not the in-memory object)
    stats = get_reconstruction_stats(str(sfm_dir))

    # Update state — store path, NOT the pycolmap object
    state.sfm_dir = str(sfm_dir)
    state.num_registered = stats["num_images"]
    state.num_points3d = stats["num_points3d"]
    state.mean_reprojection_error = stats["mean_reprojection_error"]

    reg_rate = (stats["num_images"] / state.num_images * 100) if state.num_images > 0 else 0

    stats_text = (
        f"Mapper: {mapper}\n"
        f"Camera model: {camera_model}\n"
        f"Single camera: {single_camera}\n"
        f"Registered images: {stats['num_images']} / {state.num_images} ({reg_rate:.1f}%)\n"
        f"3D points: {stats['num_points3d']:,}\n"
        f"Mean reprojection error: {stats['mean_reprojection_error']:.4f} px\n"
        f"Cameras: {stats['num_cameras']}"
    )

    log = f"Reconstruction complete. Output: {sfm_dir}"

    state.save()
    return stats_text, log, state


def build_stage45_tab():
    """Build the Stages 4-5 Reconstruction tab."""
    with gr.Tab("4-5. Reconstruction") as tab:
        gr.Markdown("### Stages 4-5: Geometric Verification & Incremental Mapping")
        gr.Markdown(
            "Combines geometric verification and incremental SfM via HLoc. "
            "This is typically the longest stage."
        )

        with gr.Row():
            with gr.Column():
                mapper_radio = gr.Radio(
                    choices=["incremental", "global (GLOMAP)"],
                    value="incremental",
                    label="Mapper",
                    info="Incremental = quality default; global = COLMAP 4 "
                         "GLOMAP, much faster, worth an A/B comparison",
                )
                camera_dropdown = gr.Dropdown(
                    choices=["OPENCV", "SIMPLE_RADIAL", "PINHOLE", "RADIAL"],
                    value="OPENCV",
                    label="Camera Model",
                    info="OPENCV recommended for general use",
                )
                single_cam_check = gr.Checkbox(
                    value=True,
                    label="Single Camera (all images share one model)",
                )
                min_matches_slider = gr.Slider(
                    minimum=5, maximum=50, value=15, step=1,
                    label="Min Matches per Pair",
                )
            with gr.Column():
                local_ba_slider = gr.Slider(
                    minimum=10, maximum=100, value=40, step=5,
                    label="Local BA Iterations",
                )
                global_ba_slider = gr.Slider(
                    minimum=10, maximum=200, value=100, step=5,
                    label="Global BA Iterations",
                )
                refine_focal_check = gr.Checkbox(value=True, label="Refine Focal Length")
                refine_principal_check = gr.Checkbox(value=True, label="Refine Principal Point")
                refine_extra_check = gr.Checkbox(value=True, label="Refine Extra Params (distortion)")

        recon_btn = gr.Button("Run Reconstruction", variant="primary")

        with gr.Row():
            stats_box = gr.Textbox(label="Reconstruction Statistics", lines=7, interactive=False)
            log_box = gr.Textbox(label="Log", lines=7, interactive=False)

    return tab, {
        "mapper_radio": mapper_radio,
        "camera_dropdown": camera_dropdown,
        "single_cam_check": single_cam_check,
        "min_matches_slider": min_matches_slider,
        "local_ba_slider": local_ba_slider,
        "global_ba_slider": global_ba_slider,
        "refine_focal_check": refine_focal_check,
        "refine_principal_check": refine_principal_check,
        "refine_extra_check": refine_extra_check,
        "recon_btn": recon_btn,
        "stats_box": stats_box,
        "log_box": log_box,
    }


def wire_stage45(components: dict, state: gr.State):
    """Connect Stages 4-5 UI events."""
    c = components

    c["recon_btn"].click(
        fn=run_reconstruction,
        inputs=[
            c["mapper_radio"],
            c["camera_dropdown"],
            c["single_cam_check"],
            c["min_matches_slider"],
            c["local_ba_slider"],
            c["global_ba_slider"],
            c["refine_focal_check"],
            c["refine_principal_check"],
            c["refine_extra_check"],
            state,
        ],
        outputs=[c["stats_box"], c["log_box"], state],
    )
