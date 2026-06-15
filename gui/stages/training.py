"""Stage 9: 3DGS Training.

Configures and runs 3D Gaussian Splatting training via a pluggable backend
(gsplat by default).  Training runs in a background thread with progress
streaming to the Gradio UI.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr

from core.pipeline_state import PipelineState
from core.training import (
    TrainingConfig,
    TrainingInterrupted,
    TrainingResult,
    available_backends,
    get_backend,
)


# ---------------------------------------------------------------------------
# Module-level training state (one training run at a time)
# ---------------------------------------------------------------------------

_train_thread: Optional[threading.Thread] = None
_train_result: Optional[TrainingResult] = None
_train_error: Optional[str] = None
_progress_queue: queue.Queue = queue.Queue()
_stop_requested = threading.Event()


def _is_training() -> bool:
    return _train_thread is not None and _train_thread.is_alive()


# ---------------------------------------------------------------------------
# Handler: Validate backend
# ---------------------------------------------------------------------------

def check_backend(backend_name: str) -> str:
    """Validate that the selected training backend is ready."""
    try:
        backend = get_backend(backend_name)
        ok, msg = backend.validate_environment()
        if ok:
            return f"Backend '{backend_name}' is ready.\n{msg}"
        return f"Backend '{backend_name}' not available:\n{msg}"
    except KeyError as e:
        return str(e)


# ---------------------------------------------------------------------------
# Handler: Prepare data
# ---------------------------------------------------------------------------

def prepare_training_data(
    backend_name: str,
    data_factor: int,
    coarse_to_fine: bool,
    state: PipelineState,
    progress=gr.Progress(),
) -> tuple[str, str, PipelineState]:
    """Prepare training data from the exported reconstruction."""
    if not state.export_dir:
        raise gr.Error("No export found. Complete Stage 7 (Export) first.")

    if not Path(state.export_dir).is_dir():
        raise gr.Error(f"Export directory not found: {state.export_dir}")

    try:
        backend = get_backend(backend_name)
    except KeyError as e:
        raise gr.Error(str(e))

    progress(0.1, desc=f"Preparing data for {backend_name}...")

    # Set up directories
    project_dir = Path(state.project_dir)
    training_dir = project_dir / "training" / backend_name
    training_dir.mkdir(parents=True, exist_ok=True)

    data_factor = int(data_factor)
    config = TrainingConfig(
        backend=backend_name,
        data_factor=data_factor,
        extra={
            "coarse_factor": data_factor * 2 if coarse_to_fine else 0,
        },
    )

    # Use masks if available
    mask_dir = state.mask_dir if state.masks_generated else None

    try:
        data_dir = backend.prepare_data(
            export_dir=state.export_dir,
            mask_dir=mask_dir,
            output_dir=str(training_dir),
            config=config,
            progress_cb=lambda cur, tot, *_: progress(
                0.1 + 0.8 * (cur / max(tot, 1)),
                desc=f"Processing image {cur}/{tot}...",
            ),
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise gr.Error(f"Data preparation failed: {e}\n\n{tb}")

    progress(0.95, desc="Finalizing...")

    # Count prepared images
    images_dir = Path(data_dir) / "images"
    num_images = len(list(images_dir.glob("*.*"))) if images_dir.is_dir() else 0

    factor_dir = Path(data_dir) / f"images_{data_factor}"
    if factor_dir.is_dir():
        ds_count = len(list(factor_dir.glob("*.*")))
        ds_msg = f"\nDownsampled ({data_factor}x): {ds_count} images"
        if ds_count > 0:
            # Get one image to report resolution
            sample = next(factor_dir.iterdir())
            from PIL import Image
            with Image.open(sample) as img:
                ds_msg += f" @ {img.width}x{img.height}"
    else:
        ds_msg = ""

    # Update state
    state.training_backend = backend_name
    state.training_dir = str(training_dir)
    state.training_data_dir = data_dir
    state.training_status = "prepared"
    state.save()

    stats = (
        f"Backend: {backend_name}\n"
        f"Data directory: {data_dir}\n"
        f"Full-res images: {num_images}\n"
        f"Masks applied: {'yes' if mask_dir else 'no'}"
        f"{ds_msg}"
    )
    log = f"Data preparation complete. Ready for training."

    return stats, log, state


# ---------------------------------------------------------------------------
# Handler: Start training (background thread)
# ---------------------------------------------------------------------------

def start_training(
    backend_name: str,
    max_steps: int,
    data_factor: int,
    test_every: int,
    strategy: str,
    cap_max: float,
    antialiased: bool,
    pose_opt: bool,
    coarse_to_fine: bool,
    perceptual_loss: bool,
    lpips_lambda: float,
    state: PipelineState,
    progress=gr.Progress(),
) -> tuple[str, str, PipelineState]:
    """Launch training in a background thread."""
    global _train_thread, _train_result, _train_error

    if _is_training():
        raise gr.Error("Training is already running. Stop it first.")

    if not state.training_data_dir:
        raise gr.Error("Data not prepared. Click 'Prepare Data' first.")

    if not Path(state.training_data_dir).is_dir():
        raise gr.Error(f"Training data directory not found: {state.training_data_dir}")

    try:
        backend = get_backend(backend_name)
    except KeyError as e:
        raise gr.Error(str(e))

    # Build config
    max_steps = int(max_steps)
    data_factor = int(data_factor)
    test_every = int(test_every)

    extra = {
        "random_bkgd": state.masks_generated,
        "strategy": strategy,
        "cap_max": int(cap_max),
        "antialiased": bool(antialiased),
        "pose_opt": bool(pose_opt),
        "coarse_factor": data_factor * 2 if coarse_to_fine else 0,
        "coarse_until": min(15_000, max_steps // 2),
    }
    # Perceptual (LPIPS) loss — clean win at lower res / under-resolved content;
    # at full res (factor 1) start it only in the fine phase to limit
    # over-texturing of already-well-resolved regions.
    if perceptual_loss:
        extra["lpips_lambda"] = float(lpips_lambda)
        extra["lpips_start"] = (min(15_000, max_steps // 2)
                                if (coarse_to_fine and data_factor <= 1)
                                else max(1, max_steps // 5))
    config = TrainingConfig(
        backend=backend_name,
        max_steps=max_steps,
        data_factor=data_factor,
        test_every=test_every,
        save_steps=[max_steps // 4, max_steps],
        extra=extra,
    )

    # Clear previous state
    _train_result = None
    _train_error = None
    _stop_requested.clear()
    while not _progress_queue.empty():
        _progress_queue.get_nowait()

    data_dir = state.training_data_dir
    training_dir = state.training_dir

    # Update state
    state.training_status = "running"
    state.training_iterations = 0
    state.save()

    # Launch training thread
    def _train_worker():
        global _train_result, _train_error
        try:
            def _progress_cb(step, total, loss, psnr):
                if _stop_requested.is_set():
                    raise TrainingInterrupted("Stop requested")
                _progress_queue.put({
                    "step": step,
                    "max_steps": total,
                    "loss": loss,
                    "psnr": psnr,
                })

            result = backend.train(
                data_dir=data_dir,
                output_dir=training_dir,
                config=config,
                progress_cb=_progress_cb,
            )
            _train_result = result
            _progress_queue.put({"done": True, "result": result})

        except TrainingInterrupted:
            _progress_queue.put({"stopped": True})

        except Exception as e:
            _train_error = f"{e}\n\n{traceback.format_exc()}"
            _progress_queue.put({"error": str(e)})

    _train_thread = threading.Thread(target=_train_worker, daemon=True)
    _train_thread.start()

    coarse_until = config.extra["coarse_until"]
    factor_desc = f"{data_factor}x"
    if coarse_to_fine:
        factor_desc += f" (coarse {data_factor * 2}x until step {coarse_until:,})"
    strategy_desc = strategy
    if strategy == "mcmc":
        strategy_desc += f" (budget {int(cap_max):,})"

    if perceptual_loss:
        percep_desc = (f"LPIPS λ={float(lpips_lambda):.2f} "
                       f"(from step {extra['lpips_start']:,})")
    else:
        percep_desc = "off"

    stats = (
        f"Training started!\n"
        f"Backend: {backend_name}\n"
        f"Max steps: {max_steps:,}\n"
        f"Data factor: {factor_desc}\n"
        f"Strategy: {strategy_desc}\n"
        f"Anti-aliasing: {'on' if antialiased else 'off'} | "
        f"Pose refinement: {'on' if pose_opt else 'off'}\n"
        f"Perceptual loss: {percep_desc}\n"
        f"Test every: {test_every} images\n"
        f"Masks: {'enabled (random_bkgd)' if state.masks_generated else 'none'}"
    )
    log = "Training launched in background thread. Use 'Refresh Progress' to check status."

    return stats, log, state


# ---------------------------------------------------------------------------
# Handler: Poll training progress
# ---------------------------------------------------------------------------

def poll_training_progress(
    state: PipelineState,
) -> tuple[str, str, PipelineState]:
    """Check the training progress queue and update stats."""
    if not _is_training() and _progress_queue.empty():
        if _train_error:
            state.training_status = "failed"
            state.save()
            return (
                f"Training FAILED:\n{_train_error}",
                "Training failed. Check the error above.",
                state,
            )
        if _train_result:
            return _format_completed_result(_train_result, state)
        if state.training_status == "running":
            return "Training not running (process may have exited).", "", state
        return "No training in progress.", "", state

    # Drain the queue, keep the latest update
    latest = None
    while not _progress_queue.empty():
        try:
            latest = _progress_queue.get_nowait()
        except queue.Empty:
            break

    if latest is None:
        return (
            f"Training in progress...\n"
            f"Step: {state.training_iterations:,} / ?\n"
            f"(waiting for next update)",
            "",
            state,
        )

    # Check for completion/error/stop
    if latest.get("done"):
        result = latest["result"]
        return _format_completed_result(result, state)

    if latest.get("stopped"):
        state.training_status = "stopped"
        state.save()
        return "Training stopped by user.", "Stopped.", state

    if latest.get("error"):
        state.training_status = "failed"
        state.save()
        return f"Training error: {latest['error']}", "Failed.", state

    # Regular progress update
    step = latest.get("step", 0)
    max_steps = latest.get("max_steps", 0)
    loss = latest.get("loss", 0)
    psnr = latest.get("psnr", 0)

    state.training_iterations = step
    state.training_loss = loss
    state.training_psnr = psnr

    pct = (step / max_steps * 100) if max_steps > 0 else 0
    stats = (
        f"Training in progress...\n"
        f"Step: {step:,} / {max_steps:,} ({pct:.1f}%)\n"
        f"Loss: {loss:.6f}\n"
        f"PSNR: {psnr:.2f} dB"
    )
    log = f"Step {step:,}: loss={loss:.6f}, psnr={psnr:.2f}"

    return stats, log, state


def _format_completed_result(
    result: TrainingResult, state: PipelineState,
) -> tuple[str, str, PipelineState]:
    """Format a completed training result and update state."""
    state.training_status = "completed"
    state.training_iterations = result.steps_completed
    state.training_psnr = result.psnr
    state.training_ssim = result.ssim
    state.training_loss = result.final_loss
    state.training_checkpoint = result.checkpoint_dir
    state.training_ply = result.ply_path
    state.save()

    stats = (
        f"Training COMPLETE!\n"
        f"Steps: {result.steps_completed:,}\n"
        f"Final loss: {result.final_loss:.6f}\n"
        f"PSNR: {result.psnr:.2f} dB\n"
        f"SSIM: {result.ssim:.4f}\n"
        f"Checkpoint: {result.checkpoint_dir or 'N/A'}\n"
        f"PLY: {result.ply_path or 'N/A'}"
    )
    log = "Training finished successfully."
    return stats, log, state


# ---------------------------------------------------------------------------
# Handler: Stop training
# ---------------------------------------------------------------------------

def stop_training(
    state: PipelineState,
) -> tuple[str, str, PipelineState]:
    """Request the training thread to stop."""
    if not _is_training():
        return "No training in progress.", "", state

    _stop_requested.set()

    # Also try calling backend's request_stop if it has one
    if state.training_backend:
        try:
            backend = get_backend(state.training_backend)
            if hasattr(backend, "request_stop"):
                backend.request_stop()
        except (KeyError, Exception):
            pass

    state.training_status = "stopped"
    state.save()

    return (
        "Stop signal sent. Training will stop after the current step.",
        "Stopping...",
        state,
    )


# ---------------------------------------------------------------------------
# Tab builder
# ---------------------------------------------------------------------------

def build_stage9_tab():
    """Build the Stage 9 3DGS Training tab."""
    backends = available_backends() or ["(none installed)"]

    with gr.Tab("9. 3DGS Training") as tab:
        gr.Markdown("### Stage 9: Gaussian Splatting Training")
        gr.Markdown(
            "Train a 3D Gaussian Splatting model from the exported reconstruction. "
            "Requires a completed export (Stage 7). Masks from Stage 8 are applied automatically."
        )

        # --- Configuration ---
        with gr.Row():
            with gr.Column():
                backend_dropdown = gr.Dropdown(
                    choices=backends,
                    value=backends[0],
                    label="Training Backend",
                )
                check_btn = gr.Button("Check Backend", size="sm")
                backend_status = gr.Textbox(
                    label="Backend Status", lines=3, interactive=False,
                )

            with gr.Column():
                data_factor_dropdown = gr.Dropdown(
                    choices=["1", "2", "4", "8"],
                    value="2",
                    label="Image Scale Factor",
                    info="Final training resolution: 2x → 3000x2000 from 24MP",
                )
                max_steps_slider = gr.Slider(
                    minimum=1000, maximum=100_000, value=30_000, step=1000,
                    label="Max Training Steps",
                    info="30K = good quality, 60K = high quality",
                )
                test_every_slider = gr.Slider(
                    minimum=1, maximum=32, value=8, step=1,
                    label="Test Every N Images",
                    info="Hold out every Nth image for evaluation",
                )

        with gr.Accordion("Quality Options", open=True):
            with gr.Row():
                strategy_dropdown = gr.Dropdown(
                    choices=["mcmc", "default"],
                    value="mcmc",
                    label="Densification Strategy",
                    info="MCMC: better quality + explicit Gaussian budget",
                )
                cap_max_slider = gr.Slider(
                    minimum=100_000, maximum=5_000_000, value=1_000_000, step=100_000,
                    label="Max Gaussians (MCMC budget)",
                    info="~1M for objects, ~3M for scenes (24GB VRAM)",
                )
            with gr.Row():
                antialiased_checkbox = gr.Checkbox(
                    value=True, label="Anti-aliasing (Mip-Splatting)",
                    info="Sharper results when rendering at scales other than training",
                )
                pose_opt_checkbox = gr.Checkbox(
                    value=True, label="Pose refinement",
                    info="Learn small SE(3) camera corrections during training",
                )
                coarse_to_fine_checkbox = gr.Checkbox(
                    value=True, label="Coarse-to-fine schedule",
                    info="Densify at 2x lower resolution, finish at full factor",
                )
            with gr.Row():
                perceptual_loss_checkbox = gr.Checkbox(
                    value=False, label="Perceptual (LPIPS) loss",
                    info="Recovers fine detail. Clean win at scale 2-8; at scale 1 "
                         "(full 4K) it can over-texture well-resolved areas — use a "
                         "low weight there.",
                )
                lpips_lambda_slider = gr.Slider(
                    minimum=0.02, maximum=0.4, value=0.1, step=0.02,
                    label="Perceptual Loss Weight",
                    info="~0.2 at scale 2-8; ~0.05-0.1 at scale 1 (avoids grain)",
                )

        # --- Actions ---
        with gr.Row():
            prepare_btn = gr.Button("Prepare Data", variant="secondary")
            train_btn = gr.Button("Start Training", variant="primary")
            stop_btn = gr.Button("Stop Training", variant="stop")
            refresh_btn = gr.Button("Refresh Progress")

        # --- Output ---
        with gr.Row():
            stats_box = gr.Textbox(
                label="Training Statistics", lines=9, interactive=False,
            )
            log_box = gr.Textbox(label="Log", lines=9, interactive=False)

    return tab, {
        "backend_dropdown": backend_dropdown,
        "check_btn": check_btn,
        "backend_status": backend_status,
        "data_factor_dropdown": data_factor_dropdown,
        "max_steps_slider": max_steps_slider,
        "test_every_slider": test_every_slider,
        "strategy_dropdown": strategy_dropdown,
        "cap_max_slider": cap_max_slider,
        "antialiased_checkbox": antialiased_checkbox,
        "pose_opt_checkbox": pose_opt_checkbox,
        "coarse_to_fine_checkbox": coarse_to_fine_checkbox,
        "perceptual_loss_checkbox": perceptual_loss_checkbox,
        "lpips_lambda_slider": lpips_lambda_slider,
        "prepare_btn": prepare_btn,
        "train_btn": train_btn,
        "stop_btn": stop_btn,
        "refresh_btn": refresh_btn,
        "stats_box": stats_box,
        "log_box": log_box,
    }


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def wire_stage9(components: dict, state: gr.State):
    """Connect Stage 9 UI events."""
    c = components

    # Check backend
    c["check_btn"].click(
        fn=check_backend,
        inputs=[c["backend_dropdown"]],
        outputs=[c["backend_status"]],
    )

    # Prepare data
    c["prepare_btn"].click(
        fn=prepare_training_data,
        inputs=[
            c["backend_dropdown"],
            c["data_factor_dropdown"],
            c["coarse_to_fine_checkbox"],
            state,
        ],
        outputs=[c["stats_box"], c["log_box"], state],
    )

    # Start training
    c["train_btn"].click(
        fn=start_training,
        inputs=[
            c["backend_dropdown"],
            c["max_steps_slider"],
            c["data_factor_dropdown"],
            c["test_every_slider"],
            c["strategy_dropdown"],
            c["cap_max_slider"],
            c["antialiased_checkbox"],
            c["pose_opt_checkbox"],
            c["coarse_to_fine_checkbox"],
            c["perceptual_loss_checkbox"],
            c["lpips_lambda_slider"],
            state,
        ],
        outputs=[c["stats_box"], c["log_box"], state],
    )

    # Stop training
    c["stop_btn"].click(
        fn=stop_training,
        inputs=[state],
        outputs=[c["stats_box"], c["log_box"], state],
    )

    # Refresh progress
    c["refresh_btn"].click(
        fn=poll_training_progress,
        inputs=[state],
        outputs=[c["stats_box"], c["log_box"], state],
    )
