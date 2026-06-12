"""Stage 3: Feature Matching.

Configures pair generation (exhaustive or retrieval-based) and runs LightGlue matching.
"""

import logging
from pathlib import Path

import gradio as gr
import h5py
import numpy as np

from core.matcher_configs import build_matching_config
from core.pipeline_state import PipelineState

logger = logging.getLogger(__name__)


def _count_pairs(pairs_path: str) -> int:
    """Count lines in the pairs file."""
    with open(pairs_path) as f:
        return sum(1 for line in f if line.strip())


def _get_match_stats(match_path: str) -> tuple[float, int, int]:
    """Read matches HDF5 and return (avg_matches, min_matches, max_matches)."""
    counts = []
    with h5py.File(match_path, "r") as f:
        for key1 in f.keys():
            group = f[key1]
            for key2 in group.keys():
                matches = group[key2]["matches0"]
                # matches0 contains indices; count valid matches (>= 0)
                valid = np.sum(np.array(matches) >= 0)
                counts.append(int(valid))
    if not counts:
        return 0.0, 0, 0
    return sum(counts) / len(counts), min(counts), max(counts)


def run_matching(
    strategy: str,
    disable_adaptive: bool,
    filter_threshold: float,
    retrieval_top_k: int,
    state: PipelineState,
    progress=gr.Progress(),
):
    """Generate image pairs and run LightGlue matching."""
    state.discover_outputs()
    if not state.feature_path:
        raise gr.Error("No features extracted. Complete Stage 2 first.")

    import traceback
    from hloc import match_features, pairs_from_exhaustive

    project_dir = Path(state.project_dir)
    feature_path = Path(state.feature_path)
    image_dir = Path(state.image_dir)

    # Determine feature type from state
    features = state.extractor_name or "aliked"

    # Adaptive confidence settings
    depth_conf = -1.0 if disable_adaptive else 0.0
    width_conf = -1.0 if disable_adaptive else 0.0

    # Build matcher config
    match_conf = build_matching_config(
        matcher="lightglue",
        features=features,
        depth_confidence=depth_conf,
        width_confidence=width_conf,
        filter_threshold=filter_threshold,
    )

    progress(0.1, desc="Generating image pairs...")

    # Build image list for pair generation — only include images that have
    # features in the HDF5 file. If extraction was interrupted, some images
    # may be on disk but missing from the feature file.
    all_names = [Path(p).name for p in state.image_list]
    with h5py.File(str(feature_path), "r") as f:
        feature_keys = set(f.keys())
    image_names = [n for n in all_names if n in feature_keys]
    skipped = len(all_names) - len(image_names)
    if skipped > 0:
        logger.warning(
            f"{skipped} images have no features (extraction may have been "
            f"interrupted). Matching will use {len(image_names)}/{len(all_names)} images."
        )
    if len(image_names) < 2:
        raise gr.Error(
            f"Only {len(image_names)} images have features. "
            f"Re-run Stage 2 to extract features for all images."
        )

    try:
        # Generate pairs
        # Note: hloc pair generators do NOT return the output path, so we
        # track the expected path ourselves.
        if strategy == "exhaustive":
            pairs_path = project_dir / "pairs-exhaustive.txt"
            pairs_from_exhaustive.main(
                output=pairs_path,
                image_list=image_names,
            )
        else:
            # Retrieval-based matching
            from hloc import pairs_from_retrieval, extract_features as ef

            # First extract global features for retrieval
            retrieval_conf = {
                "output": "global-feats-netvlad",
                "model": {"name": "netvlad"},
                "preprocessing": {"resize_max": 1024},
            }
            retrieval_path = ef.main(
                conf=retrieval_conf,
                image_dir=image_dir,
                export_dir=project_dir,
                image_list=image_names,
            )
            pairs_path = project_dir / "pairs-retrieval.txt"
            pairs_from_retrieval.main(
                descriptors=retrieval_path,
                output=pairs_path,
                num_matched=retrieval_top_k,
            )

        progress(0.4, desc="Matching features...")

        # Run matching
        match_output = project_dir / f'{feature_path.stem}_{match_conf["output"]}_{pairs_path.stem}.h5'
        match_path = match_features.main(
            conf=match_conf,
            pairs=pairs_path,
            features=feature_path,
            matches=match_output,
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise gr.Error(f"Feature matching failed: {e}\n\n{tb}")

    progress(0.9, desc="Computing statistics...")

    # Stats
    num_pairs = _count_pairs(str(pairs_path))
    avg_matches, min_matches, max_matches = _get_match_stats(str(match_path))

    # Update state
    state.match_path = str(match_path)
    state.pairs_path = str(pairs_path)
    state.matcher_name = "lightglue"
    state.num_pairs = num_pairs
    state.avg_matches_per_pair = avg_matches

    adaptive_status = "DISABLED (max quality)" if disable_adaptive else "enabled (faster)"

    stats = (
        f"Strategy: {strategy}\n"
        f"Pairs generated: {num_pairs}\n"
        f"Avg matches/pair: {avg_matches:.0f}\n"
        f"Min matches: {min_matches}\n"
        f"Max matches: {max_matches}\n"
        f"Adaptive: {adaptive_status}\n"
        f"Filter threshold: {filter_threshold}"
    )

    log = f"Matching complete. Output: {match_path}"

    state.save()
    return stats, log, state


def build_stage3_tab():
    """Build the Stage 3 Feature Matching tab."""
    with gr.Tab("3. Feature Matching") as tab:
        gr.Markdown("### Stage 3: Feature Matching (LightGlue)")

        with gr.Row():
            with gr.Column():
                strategy_dropdown = gr.Dropdown(
                    choices=["exhaustive", "retrieval"],
                    value="exhaustive",
                    label="Matching Strategy",
                    info="Exhaustive for <300 images, retrieval for larger sets",
                )
                disable_adaptive_check = gr.Checkbox(
                    value=True,
                    label="Disable Adaptive (max quality, 3-5x slower)",
                )
                filter_slider = gr.Slider(
                    minimum=0.01, maximum=0.5, value=0.1, step=0.01,
                    label="Filter Threshold",
                )
            with gr.Column():
                retrieval_k_slider = gr.Slider(
                    minimum=5, maximum=50, value=20, step=5,
                    label="Retrieval Top-K (pairs per image)",
                    visible=False,
                )

        match_btn = gr.Button("Run Feature Matching", variant="primary")

        with gr.Accordion("RoMa v2 Rescue (weak pairs)", open=False):
            gr.Markdown(
                "Re-match pairs with few LightGlue matches (low texture, glossy, "
                "wide baseline) using the RoMa v2 dense matcher, snapping results "
                "to existing ALIKED keypoints. Optional dependency — see status."
            )
            with gr.Row():
                rescue_threshold_slider = gr.Slider(
                    minimum=20, maximum=500, value=100, step=10,
                    label="Weak-Pair Threshold",
                    info="Pairs with fewer matches than this are re-matched",
                )
                rescue_btn = gr.Button("Rescue Weak Pairs", variant="secondary")

        with gr.Row():
            stats_box = gr.Textbox(label="Matching Statistics", lines=8, interactive=False)
            log_box = gr.Textbox(label="Log", lines=8, interactive=False)

        # Toggle retrieval slider visibility
        strategy_dropdown.change(
            fn=lambda s: gr.update(visible=(s == "retrieval")),
            inputs=[strategy_dropdown],
            outputs=[retrieval_k_slider],
        )

    return tab, {
        "strategy_dropdown": strategy_dropdown,
        "disable_adaptive_check": disable_adaptive_check,
        "filter_slider": filter_slider,
        "retrieval_k_slider": retrieval_k_slider,
        "match_btn": match_btn,
        "rescue_threshold_slider": rescue_threshold_slider,
        "rescue_btn": rescue_btn,
        "stats_box": stats_box,
        "log_box": log_box,
    }


def run_roma_rescue(
    min_matches: int,
    state: PipelineState,
    progress=gr.Progress(),
) -> tuple[str, str, PipelineState]:
    """Re-match weak pairs with RoMa v2 (optional dependency)."""
    state.discover_outputs()
    if not state.match_path or not state.feature_path or not state.pairs_path:
        raise gr.Error("No matches found. Run feature matching first.")

    from core.roma_rescue import rescue_weak_pairs, roma_available, unload_roma

    ok, msg = roma_available()
    if not ok:
        return f"RoMa v2 unavailable.\n\n{msg}", "Rescue skipped.", state

    progress(0.05, desc="Scanning for weak pairs...")
    try:
        stats = rescue_weak_pairs(
            image_dir=state.image_dir,
            feature_path=state.feature_path,
            match_path=state.match_path,
            pairs_path=state.pairs_path,
            min_matches=int(min_matches),
            progress_cb=lambda cur, tot, desc: progress(
                0.05 + 0.9 * (cur / max(tot, 1)), desc=desc,
            ),
        )
    finally:
        unload_roma()

    state.save()
    stats_text = (
        f"RoMa v2 rescue complete.\n"
        f"Weak pairs (<{int(min_matches)} matches): {stats['weak_pairs']}\n"
        f"Pairs improved: {stats['rescued_pairs']}\n"
        f"Matches added: {stats['added_matches']:,}\n\n"
        f"Re-run Reconstruction (Stage 4-5) to use the new matches."
    )
    return stats_text, "Rescue finished.", state


def wire_stage3(components: dict, state: gr.State):
    """Connect Stage 3 UI events."""
    c = components

    c["match_btn"].click(
        fn=run_matching,
        inputs=[
            c["strategy_dropdown"],
            c["disable_adaptive_check"],
            c["filter_slider"],
            c["retrieval_k_slider"],
            state,
        ],
        outputs=[c["stats_box"], c["log_box"], state],
    )

    c["rescue_btn"].click(
        fn=run_roma_rescue,
        inputs=[c["rescue_threshold_slider"], state],
        outputs=[c["stats_box"], c["log_box"], state],
    )
