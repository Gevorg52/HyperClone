"""Main Gradio application — assembles all stage tabs and shared state."""

import gradio as gr

from core.pipeline_state import PipelineState
from gui.stages.import_qc import build_stage1_tab, wire_stage1
from gui.stages.extraction import build_stage2_tab, wire_stage2
from gui.stages.matching import build_stage3_tab, wire_stage3
from gui.stages.reconstruction import build_stage45_tab, wire_stage45
from gui.stages.export import build_stage67_tab, wire_stage67


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    with gr.Blocks(
        title="Apex Reality — SfM Pipeline",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# Apex Reality — Structure-from-Motion Pipeline")
        gr.Markdown("*Quality-first SfM pipeline for Gaussian Splatting input*")

        # Shared pipeline state across all tabs
        state = gr.State(PipelineState())

        with gr.Tabs():
            # Stage 1: Import & QC
            _, s1_components = build_stage1_tab()
            wire_stage1(s1_components, state)

            # Stage 2: Feature Extraction
            _, s2_components = build_stage2_tab()
            wire_stage2(s2_components, state)

            # Stage 3: Feature Matching
            _, s3_components = build_stage3_tab()
            wire_stage3(s3_components, state)

            # Stages 4-5: Reconstruction
            _, s45_components = build_stage45_tab()
            wire_stage45(s45_components, state)

            # Stages 6-7: Refinement & Export
            _, s67_components = build_stage67_tab()
            wire_stage67(s67_components, state)

    return app


def launch():
    """Create and launch the Gradio app."""
    app = create_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
