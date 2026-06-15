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
