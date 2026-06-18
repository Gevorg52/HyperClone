"""Stage 10: Post-Training Enhancement.

Renders novel views from a trained 3DGS checkpoint and enhances them using
a pluggable enhancement backend (NVIDIA Fixer by default).
"""

from __future__ import annotations

import traceback
from pathlib import Path

import gradio as gr

from core.pipeline_state import PipelineState
from core.enhancement import (
    EnhancementConfig,
    available_backends,
    get_backend,
)
from core.training import (
    available_backends as available_training_backends,
    get_backend as get_training_backend,
)


# ---------------------------------------------------------------------------
# Handler: Check enhancer backend
# ---------------------------------------------------------------------------

def check_enhancer(enhancer_name: str) -> str:
    """Validate that the selected enhancement backend is ready."""
    try:
        backend = get_backend(enhancer_name)
        ok, msg = backend.validate_environment()
        if ok:
            return f"Enhancer '{enhancer_name}' is ready.\n{msg}"
        return f"Enhancer '{enhancer_name}' not available:\n{msg}"
    except KeyError as e:
        return str(e)


# ---------------------------------------------------------------------------
# Handler: Render views from trained model
# ---------------------------------------------------------------------------

def render_views(
    num_views: int,
    state: PipelineState,
    progress=gr.Progress(),
) -> tuple[str, list, PipelineState]:
    """Render views from the trained checkpoint for enhancement."""
    if not state.training_checkpoint:
        raise gr.Error(
            "No training checkpoint found. Complete Stage 9 (Training) first."
        )
    if not state.training_backend:
        raise gr.Error("Training backend not set in state.")
    if not state.training_data_dir:
        raise gr.Error("Training data directory not set in state.")

    try:
        training_backend = get_training_backend(state.training_backend)
    except KeyError as e:
        raise gr.Error(str(e))

    progress(0.1, desc="Rendering views from checkpoint...")

    project_dir = Path(state.project_dir)
    render_output = project_dir / "enhancement" / "renders"
    render_output.mkdir(parents=True, exist_ok=True)

    try:
        rendered_paths = training_backend.render_views(
            checkpoint_dir=state.training_checkpoint,
            data_dir=state.training_data_dir,
            output_dir=str(render_output),
            max_images=int(num_views),
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise gr.Error(f"Rendering failed: {e}\n\n{tb}")

    progress(1.0, desc="Rendering complete.")

    stats = (
        f"Rendered {len(rendered_paths)} views\n"
        f"Output: {render_output}"
    )

    # Return images for gallery
    gallery_items = [(p, Path(p).stem) for p in rendered_paths[:20]]

    return stats, gallery_items, state


# ---------------------------------------------------------------------------
# Handler: Enhance rendered views
# ---------------------------------------------------------------------------

def render_and_enhance(
    enhancer_name: str,
    num_views: int,
    gate: bool,
    state: PipelineState,
    progress=gr.Progress(),
) -> tuple[str, str, list, PipelineState]:
    """Render views from checkpoint, then enhance with selected backend."""
    if not state.training_checkpoint:
        raise gr.Error(
            "No training checkpoint found. Complete Stage 9 (Training) first."
        )
    if not state.training_backend:
        raise gr.Error("Training backend not set in state.")

    # --- Step 1: Render views ---
    progress(0.05, desc="Rendering views from checkpoint...")

    try:
        training_backend = get_training_backend(state.training_backend)
    except KeyError as e:
        raise gr.Error(f"Training backend error: {e}")

    project_dir = Path(state.project_dir)
    render_dir = project_dir / "enhancement" / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    try:
        rendered_paths = training_backend.render_views(
            checkpoint_dir=state.training_checkpoint,
            data_dir=state.training_data_dir,
            output_dir=str(render_dir),
            max_images=int(num_views),
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise gr.Error(f"Rendering failed: {e}\n\n{tb}")

    if not rendered_paths:
        raise gr.Error("No images were rendered. Check training checkpoint.")

    progress(0.3, desc=f"Rendered {len(rendered_paths)} views. Starting enhancement...")

    # --- Step 2: Enhance ---
    try:
        enhancer = get_backend(enhancer_name)
    except KeyError as e:
        raise gr.Error(str(e))

    enhance_dir = project_dir / "enhancement" / enhancer_name
    enhance_dir.mkdir(parents=True, exist_ok=True)

    config = EnhancementConfig(
        backend=enhancer_name,
        num_views=int(num_views),
        extra={"gated": bool(gate)},
    )

    try:
        enhanced_paths = enhancer.enhance_images(
            image_paths=rendered_paths,
            output_dir=str(enhance_dir),
            config=config,
            progress_cb=lambda cur, tot, desc: progress(
                0.3 + 0.65 * (cur / max(tot, 1)),
                desc=f"Enhancing: {desc}",
            ),
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise gr.Error(f"Enhancement failed: {e}\n\n{tb}")

    progress(0.98, desc="Finalizing...")

    # Update state
    state.enhancement_backend = enhancer_name
    state.enhancement_dir = str(enhance_dir)
    state.num_views_enhanced = len(enhanced_paths)
    state.save()

    stats = (
        f"Enhancement complete!\n"
        f"Enhancer: {enhancer_name}\n"
        f"Views rendered: {len(rendered_paths)}\n"
        f"Views enhanced: {len(enhanced_paths)}\n"
        f"Output: {enhance_dir}"
    )
    log = "Render + enhance pipeline finished."

    # Build before/after gallery (up to 10 pairs)
    gallery_items = []
    for i, enh_path in enumerate(enhanced_paths[:10]):
        if i < len(rendered_paths):
            gallery_items.append((rendered_paths[i], f"Before #{i+1}"))
            gallery_items.append((enh_path, f"After #{i+1}"))

    return stats, log, gallery_items, state


# ---------------------------------------------------------------------------
# Handler: Distill-back loop
# ---------------------------------------------------------------------------

def run_distill(
    enhancer_name: str,
    rounds: int,
    views_per_round: int,
    steps_per_round: int,
    gate: bool,
    state: PipelineState,
    progress=gr.Progress(),
) -> tuple[str, str, PipelineState]:
    """Run the Difix3D+-style progressive distill-back loop.

    Renders pseudo-views between training cameras, enhances them, and
    resumes training with the enhanced views — repeating for N rounds.
    The improved checkpoint/PLY replace the Stage 9 results in state.
    """
    if not state.training_checkpoint:
        raise gr.Error("No training checkpoint. Complete Stage 9 first.")
    if not state.training_data_dir:
        raise gr.Error("Training data directory not set in state.")

    try:
        training_backend = get_training_backend(state.training_backend)
        enhancer = get_backend(enhancer_name)
    except KeyError as e:
        raise gr.Error(str(e))

    ok, msg = enhancer.validate_environment()
    if not ok:
        raise gr.Error(f"Enhancer '{enhancer_name}' not ready:\n{msg}")

    from core.enhancement.distill import run_distill_loop

    project_dir = Path(state.project_dir)
    distill_dir = project_dir / "enhancement" / "distill"

    progress(0.02, desc="Starting distill loop...")

    try:
        result = run_distill_loop(
            training_backend=training_backend,
            enhancer=enhancer,
            data_dir=state.training_data_dir,
            checkpoint_dir=state.training_checkpoint,
            output_dir=str(distill_dir),
            rounds=int(rounds),
            views_per_round=int(views_per_round),
            steps_per_round=int(steps_per_round),
            enhancer_extra={"gated": bool(gate)},
            progress_cb=lambda frac, desc: progress(
                0.02 + 0.96 * frac, desc=desc,
            ),
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise gr.Error(f"Distill loop failed: {e}\n\n{tb}")

    # The improved model supersedes the Stage 9 results
    state.training_checkpoint = result.checkpoint_dir
    if result.ply_path:
        state.training_ply = result.ply_path
    state.training_psnr = result.psnr
    state.training_ssim = result.ssim
    state.save()

    stats = (
        f"Distill loop complete!\n"
        f"Rounds: {int(rounds)} x {int(views_per_round)} views, "
        f"{int(steps_per_round):,} steps each\n"
        f"Final test PSNR: {result.psnr:.2f} dB\n"
        f"Final test SSIM: {result.ssim:.4f}\n"
        f"Checkpoint: {result.checkpoint_dir}\n"
        f"PLY: {result.ply_path or 'N/A'}"
    )
    log = (
        "Distill-back loop finished. Stage 9 checkpoint/PLY now point at "
        "the refined model (test metrics are computed on real, never-enhanced "
        "hold-out views)."
    )
    return stats, log, state


# ---------------------------------------------------------------------------
# Handler: Auto-complete holes (detect -> seed -> reconstruct)
# ---------------------------------------------------------------------------

def check_autocomplete() -> str:
    """Validate the auto-complete environment (InFusion + difix env + CUDA)."""
    try:
        from core.enhancement.autocomplete import AutoCompleteConfig, validate_environment
        ok, msg = validate_environment(AutoCompleteConfig())
        return msg
    except Exception as e:
        return f"Auto-complete unavailable: {e}"


def run_autocomplete_handler(
    max_rounds: int,
    inpaint_occluded: bool,
    state: PipelineState,
    progress=gr.Progress(),
) -> tuple[str, str, list, PipelineState]:
    """Detect holes in the trained splat, seed them, and reconstruct.

    Only commits fills that improve agreement with the real photos (the accept/
    reject gate), so a complete splat is left unchanged. On success the refined
    checkpoint/PLY supersede the Stage 9 results. With inpaint_occluded, holes whose
    reference photo is occluded are filled with the inpainter (plausible) instead.
    """
    if not state.training_checkpoint:
        raise gr.Error("No training checkpoint. Complete Stage 9 first.")
    if not state.training_data_dir:
        raise gr.Error("Training data directory not set in state.")

    from core.enhancement.autocomplete import (
        run_autocomplete, AutoCompleteConfig, validate_environment,
    )

    cfg = AutoCompleteConfig(max_rounds=int(max_rounds),
                             inpaint_occluded=bool(inpaint_occluded))
    ok, msg = validate_environment(cfg)
    if not ok:
        raise gr.Error(f"Auto-complete not ready:\n{msg}")

    project_dir = Path(state.project_dir)
    out_dir = project_dir / "enhancement" / "autocomplete"
    progress(0.02, desc="Starting auto-complete...")

    try:
        result = run_autocomplete(
            checkpoint_dir=state.training_checkpoint,
            data_dir=state.training_data_dir,
            output_dir=str(out_dir),
            config=cfg,
            progress_cb=lambda f, d: progress(f, desc=d),
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise gr.Error(f"Auto-complete failed: {e}\n\n{tb}")

    if result.holes_filled > 0 and result.checkpoint:
        # the reconstructed model supersedes the Stage 9 results
        state.training_checkpoint = str(Path(result.checkpoint).parents[1])
        if result.ply_path:
            state.training_ply = result.ply_path
        state.save()

    log = (
        "Auto-complete finished. "
        + ("Stage 9 checkpoint/PLY now point at the completed model."
           if result.holes_filled > 0 else "Splat left unchanged (no holes to fix).")
    )
    return result.report, log, result.sample_renders, state


def run_recreate_handler(
    n_views: int,
    extend_deg: int,
    state: PipelineState,
    progress=gr.Progress(),
) -> tuple[str, str, list, PipelineState]:
    """Recreate-the-back (PLAUSIBLE, not faithful): extend the orbit into unseen
    angles, inpaint + complete each partial gap. Saves a separate `recreated.pt`
    (does NOT supersede the faithful Stage-9 model — this is generated content)."""
    if not state.training_checkpoint:
        raise gr.Error("No training checkpoint. Complete Stage 9 first.")
    if not state.training_data_dir:
        raise gr.Error("Training data directory not set in state.")

    from core.enhancement.recreate import run_recreate
    from core.enhancement.autocomplete import AutoCompleteConfig, validate_environment
    from core.enhancement.inpaint_backend import validate_backend

    cfg = AutoCompleteConfig()
    ok, msg = validate_environment(cfg)
    if not ok:
        raise gr.Error(f"Recreate not ready (InFusion):\n{msg}")
    ok2, msg2 = validate_backend(cfg.inpaint_backend)
    if not ok2:
        raise gr.Error(f"Recreate not ready (inpainter):\n{msg2}")

    out_dir = Path(state.project_dir) / "enhancement" / "recreate"
    progress(0.02, desc="Starting recreate (plausible)...")
    try:
        result = run_recreate(
            checkpoint_dir=state.training_checkpoint,
            data_dir=state.training_data_dir,
            output_dir=str(out_dir),
            config=cfg,
            n_views=int(n_views),
            extend_deg=int(extend_deg),
            progress_cb=lambda f, d: progress(f, desc=d),
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise gr.Error(f"Recreate failed: {e}\n\n{tb}")

    log = ("Recreate finished. Output is a separate plausible model — the faithful "
           "Stage-9 model is unchanged. For faithful geometry, re-capture these angles.")
    return result["report"], log, [], state


def run_coverage_report(
    state: PipelineState,
    progress=gr.Progress(),
) -> tuple[str, "object"]:
    """Analyze the capture's angular coverage of the object (viewing hemisphere) and
    report the gaps + a 'shoot these angles' re-capture guidance. The FAITHFUL way to
    complete an object is to capture the missing angles — generative fill is plausible-
    only and OOD-limited on real captures."""
    if not state.training_data_dir:
        raise gr.Error("No training data directory in state. Complete Stage 9 first.")
    import numpy as np
    import cv2
    from core.training.gsplat_trainer import load_colmap_data
    from core.enhancement.coverage import (
        analyze_coverage, coverage_report, render_coverage_map, object_center_from_cameras,
    )
    progress(0.2, desc="Loading cameras...")
    data = load_colmap_data(state.training_data_dir, factor=4, test_every=8, normalize=True)
    n = len(data["camtoworlds"])
    cen = np.stack([data["camtoworlds"][i][:3, 3] for i in range(n)])
    fwd = np.stack([data["camtoworlds"][i][:3, 2] for i in range(n)])
    progress(0.6, desc="Analyzing coverage...")
    O = object_center_from_cameras(cen, fwd)
    res = analyze_coverage(cen, O)
    report = coverage_report(res)
    out = Path(state.project_dir) / "enhancement"
    out.mkdir(parents=True, exist_ok=True)
    map_path = str(out / "coverage_map.png")
    cv2.imwrite(map_path, render_coverage_map(res))
    progress(1.0, desc="Coverage report ready.")
    return report, map_path


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_stage10_tab():
    """Build the Stage 10 Enhancement tab."""
    enhancers = available_backends() or ["(none installed)"]

    with gr.Tab("10. Enhancement") as tab:
        gr.Markdown("### Stage 10: Post-Training Enhancement")
        gr.Markdown(
            "Enhance rendered views using diffusion-based artifact removal. "
            "Requires a completed training run (Stage 9)."
        )

        # --- Configuration ---
        with gr.Row():
            with gr.Column():
                enhancer_dropdown = gr.Dropdown(
                    choices=enhancers,
                    value=enhancers[0],
                    label="Enhancement Backend",
                )
                check_btn = gr.Button("Check Enhancer", size="sm")
                enhancer_status = gr.Textbox(
                    label="Enhancer Status", lines=3, interactive=False,
                )

            with gr.Column():
                num_views_slider = gr.Slider(
                    minimum=1, maximum=500, value=100, step=1,
                    label="Number of Views to Render & Enhance",
                    info="More views = better coverage but longer processing",
                )
                gate_check = gr.Checkbox(
                    value=True,
                    label="Gate Difix (protect already-sharp regions)",
                    info="Blends Difix in only where the render is locally soft, "
                         "leaving the well-resolved subject untouched. Recommended.",
                )

        # --- Actions ---
        with gr.Row():
            enhance_btn = gr.Button("Render & Enhance", variant="primary")

        # --- Distill-back loop ---
        with gr.Accordion("Distill-Back Loop (bake enhancement into the splat)", open=True):
            gr.Markdown(
                "Difix3D+-style progressive update: render pseudo-views between "
                "training cameras → enhance → resume training with the enhanced "
                "views. Improves the 3DGS model itself, not just exported frames."
            )
            with gr.Row():
                distill_rounds_slider = gr.Slider(
                    minimum=1, maximum=5, value=2, step=1,
                    label="Rounds",
                )
                distill_views_slider = gr.Slider(
                    minimum=20, maximum=300, value=80, step=10,
                    label="Pseudo-Views per Round",
                )
                distill_steps_slider = gr.Slider(
                    minimum=1000, maximum=10_000, value=4000, step=500,
                    label="Training Steps per Round",
                )
            distill_btn = gr.Button("Run Distill Loop", variant="primary")

        # --- Auto-complete holes ---
        with gr.Accordion("Auto-Complete Holes (detect → seed → reconstruct)", open=False):
            gr.Markdown(
                "Automatically **detect** geometry holes/defects (multi-view "
                "consistency + coverage), **seed** them (InFusion depth-completion), "
                "and **reconstruct** faithfully with the production trainer "
                "(MCMC + gated LPIPS). An accept/reject gate commits a fill only if it "
                "improves agreement with the real photos, so a complete splat is left "
                "untouched. Requires the `difix` env + InFusion checkpoint."
            )
            with gr.Row():
                ac_rounds_slider = gr.Slider(
                    minimum=1, maximum=10, value=6, step=1,
                    label="Max hole rounds",
                    info="Each round fills (or rejects) the worst remaining hole.",
                )
                ac_check_btn = gr.Button("Check Auto-Complete Env", size="sm")
            ac_inpaint_check = gr.Checkbox(
                value=False,
                label="Inpaint occluded holes (plausible, not faithful)",
                info="When a hole's reference photo is occluded, fill it with the "
                     "Z-Image inpainter instead of the real photo. The accept/reject "
                     "gate still applies. Off = faithful-only.",
            )
            ac_status = gr.Textbox(
                label="Auto-Complete Status", lines=2, interactive=False,
            )
            ac_btn = gr.Button("Run Auto-Complete", variant="primary")

        # --- Recreate unseen regions (plausible) ---
        with gr.Accordion("Recreate Unseen Regions — plausible, NOT faithful", open=False):
            gr.Markdown(
                "Extend the captured orbit into **unseen** angles and generate "
                "plausible geometry there (Z-Image inpaint → InFusion depth → seed). "
                "**This is hallucinated content where no camera looked** — for faithful "
                "geometry, re-capture those angles. Saves a separate `recreated.pt`; "
                "the faithful Stage-9 model is left unchanged."
            )
            with gr.Row():
                rc_views_slider = gr.Slider(
                    minimum=2, maximum=24, value=8, step=2,
                    label="Novel views", info="How many unseen viewpoints to attempt.",
                )
                rc_extend_slider = gr.Slider(
                    minimum=10, maximum=90, value=30, step=5,
                    label="Extend beyond orbit (deg)",
                    info="How far past the captured arc to reach.",
                )
            rc_btn = gr.Button("Recreate Unseen (plausible)")

        # --- Capture coverage / re-capture guidance (FAITHFUL completion) ---
        with gr.Accordion("Capture Coverage — re-capture guidance (faithful)", open=False):
            gr.Markdown(
                "Analyze how completely the capture covers the object's viewing "
                "hemisphere and report the **missing angles to shoot**. For a *complete, "
                "faithful* object the honest fix is to capture the gaps — generative "
                "fill of never-seen sides is plausible-only and degrades on real "
                "captures (Objaverse-trained generators are out-of-distribution)."
            )
            cov_btn = gr.Button("Analyze Coverage")
            with gr.Row():
                cov_report = gr.Textbox(label="Coverage / re-capture report", lines=6,
                                        interactive=False)
                cov_map = gr.Image(label="Coverage map (azimuth × elevation)", height=240)

        # --- Output ---
        with gr.Row():
            stats_box = gr.Textbox(
                label="Enhancement Statistics", lines=8, interactive=False,
            )
            log_box = gr.Textbox(label="Log", lines=8, interactive=False)

        gr.Markdown("**Before / After Gallery** (first 10 pairs)")
        gallery = gr.Gallery(
            label="Before / After Comparison",
            columns=4,
            height=400,
        )

    return tab, {
        "enhancer_dropdown": enhancer_dropdown,
        "check_btn": check_btn,
        "enhancer_status": enhancer_status,
        "num_views_slider": num_views_slider,
        "gate_check": gate_check,
        "enhance_btn": enhance_btn,
        "distill_rounds_slider": distill_rounds_slider,
        "distill_views_slider": distill_views_slider,
        "distill_steps_slider": distill_steps_slider,
        "distill_btn": distill_btn,
        "ac_rounds_slider": ac_rounds_slider,
        "ac_check_btn": ac_check_btn,
        "ac_inpaint_check": ac_inpaint_check,
        "ac_status": ac_status,
        "ac_btn": ac_btn,
        "rc_views_slider": rc_views_slider,
        "rc_extend_slider": rc_extend_slider,
        "rc_btn": rc_btn,
        "cov_btn": cov_btn,
        "cov_report": cov_report,
        "cov_map": cov_map,
        "stats_box": stats_box,
        "log_box": log_box,
        "gallery": gallery,
    }


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def wire_stage10(components: dict, state: gr.State):
    """Connect Stage 10 UI events."""
    c = components

    # Check enhancer
    c["check_btn"].click(
        fn=check_enhancer,
        inputs=[c["enhancer_dropdown"]],
        outputs=[c["enhancer_status"]],
    )

    # Render & Enhance
    c["enhance_btn"].click(
        fn=render_and_enhance,
        inputs=[c["enhancer_dropdown"], c["num_views_slider"], c["gate_check"], state],
        outputs=[c["stats_box"], c["log_box"], c["gallery"], state],
    )

    # Distill-back loop
    c["distill_btn"].click(
        fn=run_distill,
        inputs=[
            c["enhancer_dropdown"],
            c["distill_rounds_slider"],
            c["distill_views_slider"],
            c["distill_steps_slider"],
            c["gate_check"],
            state,
        ],
        outputs=[c["stats_box"], c["log_box"], state],
    )

    # Auto-complete holes
    c["ac_check_btn"].click(
        fn=check_autocomplete,
        inputs=[],
        outputs=[c["ac_status"]],
    )
    c["ac_btn"].click(
        fn=run_autocomplete_handler,
        inputs=[c["ac_rounds_slider"], c["ac_inpaint_check"], state],
        outputs=[c["stats_box"], c["log_box"], c["gallery"], state],
    )

    # Recreate unseen (plausible)
    c["rc_btn"].click(
        fn=run_recreate_handler,
        inputs=[c["rc_views_slider"], c["rc_extend_slider"], state],
        outputs=[c["stats_box"], c["log_box"], c["gallery"], state],
    )

    # Coverage / re-capture guidance
    c["cov_btn"].click(
        fn=run_coverage_report,
        inputs=[state],
        outputs=[c["cov_report"], c["cov_map"]],
    )
