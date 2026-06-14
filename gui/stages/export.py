"""Stages 6-7: Final Bundle Adjustment Refinement & Export.

Stage 6: Run standalone BA with higher iterations for final quality pass.
Stage 7: Export to 3DGS-ready directory layout.
"""

import subprocess
import shutil
from pathlib import Path

import gradio as gr
import pycolmap

from core.masking import create_rgba_image
from core.pipeline_state import PipelineState
from core.utils import SUPPORTED_EXTENSIONS, create_3dgs_directory, get_reconstruction_stats


def _browse_export_folder() -> str:
    """Open a native folder selection dialog for export path.

    Creates a fresh Tk root each call and destroys it afterwards to avoid
    cross-thread issues (Gradio runs handlers in worker threads, but
    Tkinter objects can only be accessed from the thread that created them).
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        folder = filedialog.askdirectory(title="Select Export Folder")
    finally:
        root.destroy()
    return folder or ""


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

    # Try pycolmap bundle adjustment API (preferred)
    try:
        ba_options = pycolmap.BundleAdjustmentOptions()
        ba_options.refine_focal_length = refine_focal
        ba_options.refine_principal_point = refine_principal
        ba_options.refine_extra_params = refine_extra

        # Solver iteration cap location differs across pycolmap versions:
        #   pycolmap >= 4.0 : ba_options.ceres.solver_options.max_num_iterations
        #   pycolmap 3.x    : ba_options.solver_options.max_num_iterations
        # Best-effort; BA still runs (at the Ceres default) if neither is present.
        for _set_iters in (
            lambda: setattr(ba_options.ceres.solver_options,
                            "max_num_iterations", ba_iterations),
            lambda: setattr(ba_options.solver_options,
                            "max_num_iterations", ba_iterations),
        ):
            try:
                _set_iters()
                break
            except AttributeError:
                continue

        pycolmap.bundle_adjustment(rec, ba_options)

        # Write refined reconstruction back
        rec.write(str(sfm_dir))

    except Exception as api_err:
        # Fallback: COLMAP CLI
        colmap_bin = shutil.which("colmap")
        if colmap_bin is None:
            raise RuntimeError(
                f"pycolmap bundle_adjustment failed ({api_err}) and COLMAP CLI "
                "not found on PATH. Install COLMAP or update pycolmap."
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
    state.discover_outputs()
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
    state.aligned = False  # BA rewrites .bin files, reset alignment
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
    enable_alignment: bool,
    up_axis: str,
    enable_centering: bool,
    use_imu: bool,
    include_masks: bool,
    mask_format: str,
    state: PipelineState,
    progress=gr.Progress(),
):
    """Export reconstruction in 3DGS-ready directory layout."""
    state.discover_outputs()
    if not state.sfm_dir:
        raise gr.Error("No reconstruction found. Complete Stages 4-5 first.")
    if not export_path:
        raise gr.Error("Please specify an export path.")

    sfm_dir = Path(state.sfm_dir)
    image_dir = Path(state.image_dir)
    project_dir = Path(state.project_dir)
    out_dir = Path(export_path)

    progress(0.1, desc="Loading reconstruction...")

    # Load reconstruction once — reuse for alignment, text export, and report
    rec = pycolmap.Reconstruction(str(sfm_dir))

    # --- Alignment ---
    alignment_info_str = ""
    if enable_alignment and not state.aligned:
        progress(0.15, desc="Aligning to gravity...")
        try:
            from core.alignment import align_reconstruction
            info = align_reconstruction(
                rec, str(image_dir), str(project_dir),
                up_axis, enable_centering, use_imu,
            )
            rec.write(str(sfm_dir))  # persist aligned reconstruction
            state.aligned = True
            state.alignment_method = info["method"]
            state.alignment_imu_coverage = info["imu_coverage"]
            state.alignment_rotation_deg = info["rotation_angle_deg"]
            state.alignment_up_axis = info["up_axis"]
            state.alignment_centered = info["centered"]
            alignment_info_str = (
                f"Alignment: {info['method']} "
                f"(rotation {info['rotation_angle_deg']:.1f}°, "
                f"up={info['up_axis']}, "
                f"centered={info['centered']}"
            )
            if info["method"] == "imu":
                alignment_info_str += f", IMU coverage={info['imu_coverage']:.0%}"
            alignment_info_str += ")\n"
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Alignment failed: %s", e)
            alignment_info_str = f"Alignment: skipped (error: {e})\n"
    elif state.aligned:
        alignment_info_str = (
            f"Alignment: already applied ({state.alignment_method}, "
            f"{state.alignment_rotation_deg:.1f}°, up={state.alignment_up_axis})\n"
        )

    progress(0.3, desc="Creating export directory...")

    try:
        # If text format requested, write text versions
        if export_format in ("text", "both"):
            text_dir = sfm_dir / "text"
            text_dir.mkdir(exist_ok=True)
            rec.write_text(str(text_dir))

        progress(0.5, desc="Creating 3DGS directory layout...")

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

    # --- Mask export ---
    mask_info_str = ""
    if include_masks:
        if not state.masks_generated or not state.mask_dir:
            raise gr.Error("No masks found. Run Stage 8 (Object Masking) first.")

        progress(0.7, desc="Exporting masks...")
        mask_src = Path(state.mask_dir)

        if mask_format in ("bw_masks", "both"):
            masks_out = out_dir / "masks"
            masks_out.mkdir(exist_ok=True)
            mask_count = 0
            for mask_file in mask_src.glob("*.png"):
                shutil.copy2(str(mask_file), str(masks_out / mask_file.name))
                mask_count += 1
            mask_info_str += f"B&W masks exported: {mask_count}\n"

        if mask_format in ("rgba_images", "both"):
            rgba_out = out_dir / "rgba"
            rgba_out.mkdir(exist_ok=True)
            rgba_count = 0
            for img_file in Path(state.image_dir).iterdir():
                if img_file.suffix.lower() in SUPPORTED_EXTENSIONS and img_file.is_file():
                    mask_path = mask_src / f"{img_file.stem}.png"
                    if mask_path.exists():
                        create_rgba_image(
                            str(img_file), str(mask_path),
                            str(rgba_out / f"{img_file.stem}.png"),
                        )
                        rgba_count += 1
            mask_info_str += f"RGBA images exported: {rgba_count}\n"

    progress(0.8, desc="Generating quality report...")

    # Final quality metrics
    stats = get_reconstruction_stats(str(sfm_dir))
    reg_rate = (stats["num_images"] / state.num_images * 100) if state.num_images > 0 else 0

    # Camera intrinsics from the already-loaded rec
    cam_info = ""
    for cam_id, cam in rec.cameras.items():
        cam_info += f"  Camera {cam_id}: {cam.model.name}, {cam.width}x{cam.height}, params={list(cam.params)}\n"

    state.export_dir = str(export_dir)

    quality_report = (
        f"=== Final Quality Report ===\n"
        f"Registered images: {stats['num_images']} / {state.num_images} ({reg_rate:.1f}%)\n"
        f"3D points: {stats['num_points3d']:,}\n"
        f"Mean reprojection error: {stats['mean_reprojection_error']:.4f} px\n"
        f"Camera intrinsics:\n{cam_info}"
        f"{alignment_info_str}"
        f"{mask_info_str}"
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
            export_path_input = gr.Textbox(
                label="Export Directory",
                placeholder="F:/path/to/export/output",
                scale=4,
            )
            browse_export_btn = gr.Button("Browse...", scale=1)

        with gr.Row():
            export_format_dropdown = gr.Dropdown(
                choices=["binary", "text", "both"],
                value="binary",
                label="Export Format",
            )

        gr.Markdown("#### Alignment")
        with gr.Row():
            align_check = gr.Checkbox(value=True, label="Auto-align to gravity")
            center_check = gr.Checkbox(value=True, label="Center at origin")
        with gr.Row():
            up_axis_dropdown = gr.Dropdown(
                choices=["y", "z"],
                value="y",
                label="Up Axis",
            )
            imu_check = gr.Checkbox(value=True, label="Use IMU data when available")

        gr.Markdown("#### Masks")
        with gr.Row():
            include_masks_check = gr.Checkbox(value=False, label="Include masks in export")
            mask_format_dropdown = gr.Dropdown(
                choices=["bw_masks", "rgba_images", "both"],
                value="bw_masks",
                label="Mask Format",
            )

        with gr.Row():
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
        "browse_export_btn": browse_export_btn,
        "export_format_dropdown": export_format_dropdown,
        "align_check": align_check,
        "center_check": center_check,
        "up_axis_dropdown": up_axis_dropdown,
        "imu_check": imu_check,
        "include_masks_check": include_masks_check,
        "mask_format_dropdown": mask_format_dropdown,
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

    c["browse_export_btn"].click(
        fn=_browse_export_folder,
        inputs=[],
        outputs=[c["export_path_input"]],
    )

    c["export_btn"].click(
        fn=run_export,
        inputs=[
            c["export_path_input"],
            c["export_format_dropdown"],
            c["align_check"],
            c["up_axis_dropdown"],
            c["center_check"],
            c["imu_check"],
            c["include_masks_check"],
            c["mask_format_dropdown"],
            state,
        ],
        outputs=[c["quality_report"], c["export_log"], state],
    )
