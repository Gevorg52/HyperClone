"""Stages 6-7: Final Bundle Adjustment Refinement & Export.

Stage 6: Run standalone BA with higher iterations for final quality pass.
Stage 7: Export to 3DGS-ready directory layout.
"""

import subprocess
import shutil
from pathlib import Path

import gradio as gr
import pycolmap

from core.pipeline_state import PipelineState
from core.utils import create_3dgs_directory, get_reconstruction_stats


def _run_standalone_ba(
    sfm_dir: Path,
    ba_iterations: int,
    refine_focal: bool,
    refine_principal: bool,
    refine_extra: bool,
) -> tuple[float, float]:
    """Run standalone bundle adjustment on existing reconstruction.

    Tries pycolmap API first, falls back to COLMAP CLI subprocess.
    Returns (pre_ba_error, post_ba_error).
    """
    rec = pycolmap.Reconstruction(str(sfm_dir))

    # Pre-BA error
    errors_before = [p.error for p in rec.points3D.values() if p.error > 0]
    pre_error = sum(errors_before) / len(errors_before) if errors_before else 0.0

    # Try pycolmap bundle adjustment API
    try:
        ba_options = pycolmap.BundleAdjustmentOptions()
        try:
            ba_options.solver_options.max_num_iterations = ba_iterations
        except AttributeError:
            raise RuntimeError(
                "pycolmap BA API mismatch: 'solver_options.max_num_iterations' not found. "
                "Check pycolmap version (need >= 3.13)."
            )
        ba_options.refine_focal_length = refine_focal
        ba_options.refine_principal_point = refine_principal
        ba_options.refine_extra_params = refine_extra

        pycolmap.bundle_adjustment(rec, ba_options)

        # Write refined reconstruction back
        rec.write(str(sfm_dir))

    except (AttributeError, TypeError):
        # Fallback: try COLMAP CLI
        colmap_bin = shutil.which("colmap")
        if colmap_bin is None:
            raise RuntimeError(
                "pycolmap bundle_adjustment API not available and COLMAP CLI not found. "
                "Install COLMAP or update pycolmap."
            )

        cmd = [
            colmap_bin, "bundle_adjuster",
            "--input_path", str(sfm_dir),
            "--output_path", str(sfm_dir),
            "--BundleAdjustment.max_num_iterations", str(ba_iterations),
            "--BundleAdjustment.refine_focal_length", str(refine_focal).lower(),
            "--BundleAdjustment.refine_principal_point", str(refine_principal).lower(),
            "--BundleAdjustment.refine_extra_params", str(refine_extra).lower(),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP bundle_adjuster failed:\n{result.stderr}")

    # Post-BA error
    rec_after = pycolmap.Reconstruction(str(sfm_dir))
    errors_after = [p.error for p in rec_after.points3D.values() if p.error > 0]
    post_error = sum(errors_after) / len(errors_after) if errors_after else 0.0

    return pre_error, post_error


def run_refinement(
    ba_iterations: int,
    refine_focal: bool,
    refine_principal: bool,
    refine_extra: bool,
    state: PipelineState,
    progress=gr.Progress(),
):
    """Run final bundle adjustment refinement."""
    if not state.sfm_dir:
        raise gr.Error("No reconstruction found. Complete Stages 4-5 first.")

    sfm_dir = Path(state.sfm_dir)

    progress(0.1, desc=f"Running final BA ({ba_iterations} iterations)...")

    try:
        pre_error, post_error = _run_standalone_ba(
            sfm_dir=sfm_dir,
            ba_iterations=int(ba_iterations),
            refine_focal=refine_focal,
            refine_principal=refine_principal,
            refine_extra=refine_extra,
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise gr.Error(f"Bundle adjustment failed: {e}\n\n{tb}")

    progress(0.9, desc="Done.")

    # Update state
    state.refined = True
    state.pre_ba_error = pre_error
    state.post_ba_error = post_error
    state.mean_reprojection_error = post_error

    improvement = ((pre_error - post_error) / pre_error * 100) if pre_error > 0 else 0

    stats = (
        f"BA iterations: {ba_iterations}\n"
        f"Pre-BA error:  {pre_error:.4f} px\n"
        f"Post-BA error: {post_error:.4f} px\n"
        f"Improvement: {improvement:.1f}%"
    )

    log = "Final bundle adjustment complete."

    state.save()
    return stats, log, state


def run_export(
    export_path: str,
    export_format: str,
    state: PipelineState,
    progress=gr.Progress(),
):
    """Export reconstruction in 3DGS-ready directory layout."""
    if not state.sfm_dir:
        raise gr.Error("No reconstruction found. Complete Stages 4-5 first.")
    if not export_path:
        raise gr.Error("Please specify an export path.")

    sfm_dir = Path(state.sfm_dir)
    image_dir = Path(state.image_dir)
    out_dir = Path(export_path)

    progress(0.1, desc="Creating export directory...")

    try:
        # If text format requested, write text versions first
        if export_format in ("text", "both"):
            text_dir = sfm_dir / "text"
            text_dir.mkdir(exist_ok=True)
            rec = pycolmap.Reconstruction(str(sfm_dir))
            rec.write_text(str(text_dir))

        progress(0.3, desc="Creating 3DGS directory layout...")

        # Create standard 3DGS directory
        export_dir = create_3dgs_directory(
            sfm_dir=str(sfm_dir),
            image_dir=str(image_dir),
            export_dir=str(out_dir),
        )

        # If text or both, also copy text files
        if export_format in ("text", "both"):
            text_sparse = out_dir / "sparse" / "0_text"
            text_sparse.mkdir(parents=True, exist_ok=True)
            for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
                src = sfm_dir / "text" / fname
                if src.exists():
                    shutil.copy2(str(src), str(text_sparse / fname))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise gr.Error(f"Export failed: {e}\n\n{tb}")

    progress(0.8, desc="Generating quality report...")

    # Final quality metrics
    stats = get_reconstruction_stats(str(sfm_dir))
    reg_rate = (stats["num_images"] / state.num_images * 100) if state.num_images > 0 else 0

    # Read camera intrinsics for report
    rec = pycolmap.Reconstruction(str(sfm_dir))
    cam_info = ""
    for cam_id, cam in rec.cameras.items():
        cam_info += f"  Camera {cam_id}: {cam.model_name}, {cam.width}x{cam.height}, params={list(cam.params)}\n"

    state.export_dir = str(export_dir)

    quality_report = (
        f"=== Final Quality Report ===\n"
        f"Registered images: {stats['num_images']} / {state.num_images} ({reg_rate:.1f}%)\n"
        f"3D points: {stats['num_points3d']:,}\n"
        f"Mean reprojection error: {stats['mean_reprojection_error']:.4f} px\n"
        f"Camera intrinsics:\n{cam_info}"
        f"Export format: {export_format}\n"
        f"Export directory: {export_dir}"
    )

    log = f"Export complete. 3DGS-ready directory: {export_dir}"

    state.save()
    return quality_report, log, state


def build_stage67_tab():
    """Build the Stages 6-7 Refinement & Export tab."""
    with gr.Tab("6-7. Refine & Export") as tab:
        # --- Stage 6: Refinement ---
        gr.Markdown("### Stage 6: Final Bundle Adjustment")

        with gr.Row():
            with gr.Column():
                ba_iter_slider = gr.Slider(
                    minimum=50, maximum=500, value=200, step=25,
                    label="BA Iterations",
                )
            with gr.Column():
                refine_focal_check = gr.Checkbox(value=True, label="Refine Focal Length")
                refine_principal_check = gr.Checkbox(value=True, label="Refine Principal Point")
                refine_extra_check = gr.Checkbox(value=True, label="Refine Extra Params")

        refine_btn = gr.Button("Run Final Bundle Adjustment", variant="primary")

        with gr.Row():
            refine_stats = gr.Textbox(label="Refinement Results", lines=5, interactive=False)
            refine_log = gr.Textbox(label="Log", lines=5, interactive=False)

        gr.Markdown("---")

        # --- Stage 7: Export ---
        gr.Markdown("### Stage 7: Export for Gaussian Splatting")

        with gr.Row():
            with gr.Column():
                export_path_input = gr.Textbox(
                    label="Export Directory",
                    placeholder="F:/path/to/export/output",
                )
                export_format_dropdown = gr.Dropdown(
                    choices=["binary", "text", "both"],
                    value="binary",
                    label="Export Format",
                )
            with gr.Column():
                export_btn = gr.Button("Export for 3DGS", variant="primary")

        with gr.Row():
            quality_report = gr.Textbox(label="Quality Report", lines=10, interactive=False)
            export_log = gr.Textbox(label="Log", lines=10, interactive=False)

    return tab, {
        "ba_iter_slider": ba_iter_slider,
        "refine_focal_check": refine_focal_check,
        "refine_principal_check": refine_principal_check,
        "refine_extra_check": refine_extra_check,
        "refine_btn": refine_btn,
        "refine_stats": refine_stats,
        "refine_log": refine_log,
        "export_path_input": export_path_input,
        "export_format_dropdown": export_format_dropdown,
        "export_btn": export_btn,
        "quality_report": quality_report,
        "export_log": export_log,
    }


def wire_stage67(components: dict, state: gr.State):
    """Connect Stages 6-7 UI events."""
    c = components

    c["refine_btn"].click(
        fn=run_refinement,
        inputs=[
            c["ba_iter_slider"],
            c["refine_focal_check"],
            c["refine_principal_check"],
            c["refine_extra_check"],
            state,
        ],
        outputs=[c["refine_stats"], c["refine_log"], state],
    )

    c["export_btn"].click(
        fn=run_export,
        inputs=[
            c["export_path_input"],
            c["export_format_dropdown"],
            state,
        ],
        outputs=[c["quality_report"], c["export_log"], state],
    )
