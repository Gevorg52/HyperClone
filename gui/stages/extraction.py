"""Stage 2: Feature Extraction.

Configures and runs HLoc feature extraction with SuperPoint, DISK, or ALIKED.
"""

from pathlib import Path

import gradio as gr
import h5py

from core.feature_configs import build_extraction_config
from core.pipeline_state import PipelineState


def _get_keypoint_stats(feature_path: str) -> tuple[float, int, int]:
    """Read HDF5 features and return (avg_keypoints, min_keypoints, max_keypoints)."""
    counts = []
    with h5py.File(feature_path, "r") as f:
        for key in f.keys():
            kp = f[key]["keypoints"]
            counts.append(kp.shape[0])
    if not counts:
        return 0.0, 0, 0
    return sum(counts) / len(counts), min(counts), max(counts)


def run_extraction(
    extractor: str,
    max_keypoints: int,
    nms_radius: int,
    use_full_res: bool,
    resize_max: int,
    as_half: bool,
    state: PipelineState,
    progress=gr.Progress(),
):
    """Run HLoc feature extraction on the loaded image set."""
    if not state.image_dir or not state.image_list:
        raise gr.Error("No images loaded. Complete Stage 1 first.")

    import traceback
    from hloc import extract_features

    image_dir = Path(state.image_dir)
    project_dir = Path(state.project_dir)

    # Build config
    effective_resize = None if use_full_res else resize_max
    conf = build_extraction_config(
        extractor=extractor,
        max_keypoints=int(max_keypoints),
        nms_radius=int(nms_radius),
        resize_max=effective_resize,
        as_half=as_half,
    )

    progress(0.1, desc=f"Extracting features with {extractor}...")

    try:
        # Run HLoc extraction
        feature_path = extract_features.main(
            conf=conf,
            image_dir=image_dir,
            export_dir=project_dir,
            image_list=[Path(p).name for p in state.image_list],
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise gr.Error(f"Feature extraction failed: {e}\n\n{tb}")

    progress(0.9, desc="Computing statistics...")

    # Get stats
    avg_kp, min_kp, max_kp = _get_keypoint_stats(str(feature_path))

    # Update state
    state.feature_path = str(feature_path)
    state.extractor_name = extractor
    state.max_keypoints_used = int(max_keypoints)
    state.avg_keypoints = avg_kp

    stats = (
        f"Extractor: {extractor}\n"
        f"Feature file: {feature_path.name}\n"
        f"Images processed: {len(state.image_list)}\n"
        f"Avg keypoints/image: {avg_kp:.0f}\n"
        f"Min keypoints: {min_kp}\n"
        f"Max keypoints: {max_kp}"
    )

    log = f"Feature extraction complete. Output: {feature_path}"

    state.save()
    return stats, log, state


def build_stage2_tab():
    """Build the Stage 2 Feature Extraction tab."""
    with gr.Tab("2. Feature Extraction") as tab:
        gr.Markdown("### Stage 2: Feature Extraction")

        with gr.Row():
            with gr.Column():
                extractor_dropdown = gr.Dropdown(
                    choices=["superpoint", "disk", "aliked"],
                    value="aliked",
                    label="Feature Extractor",
                )
                license_warning = gr.Markdown(
                    "**Note:** ALIKED (default) and DISK use permissive BSD-3 licenses. "
                    "SuperPoint has a non-commercial license (MagicLeap).",
                    visible=True,
                )
                max_kp_slider = gr.Slider(
                    minimum=1024, maximum=32768, value=8192, step=512,
                    label="Max Keypoints per Image",
                )
                nms_slider = gr.Slider(
                    minimum=1, maximum=8, value=3, step=1,
                    label="NMS Radius (SuperPoint only)",
                )
            with gr.Column():
                full_res_check = gr.Checkbox(
                    value=False,
                    label="Use Full Resolution (no resize)",
                )
                resize_slider = gr.Slider(
                    minimum=800, maximum=6000, value=1600, step=100,
                    label="Max Image Dimension (if resize enabled)",
                    visible=True,
                )
                half_check = gr.Checkbox(
                    value=False,
                    label="Half Precision (float16) — uncheck for max quality",
                )

        extract_btn = gr.Button("Run Feature Extraction", variant="primary")

        with gr.Row():
            stats_box = gr.Textbox(label="Extraction Statistics", lines=7, interactive=False)
            log_box = gr.Textbox(label="Log", lines=7, interactive=False)

        # Toggle resize slider visibility
        full_res_check.change(
            fn=lambda checked: gr.update(visible=not checked),
            inputs=[full_res_check],
            outputs=[resize_slider],
        )

    return tab, {
        "extractor_dropdown": extractor_dropdown,
        "max_kp_slider": max_kp_slider,
        "nms_slider": nms_slider,
        "full_res_check": full_res_check,
        "resize_slider": resize_slider,
        "half_check": half_check,
        "extract_btn": extract_btn,
        "stats_box": stats_box,
        "log_box": log_box,
    }


def wire_stage2(components: dict, state: gr.State):
    """Connect Stage 2 UI events."""
    c = components

    c["extract_btn"].click(
        fn=run_extraction,
        inputs=[
            c["extractor_dropdown"],
            c["max_kp_slider"],
            c["nms_slider"],
            c["full_res_check"],
            c["resize_slider"],
            c["half_check"],
            state,
        ],
        outputs=[c["stats_box"], c["log_box"], state],
    )
