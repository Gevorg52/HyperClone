"""gui2 — Apex Reality GUI 2.0 wizard shell (UI-1).

A single-page, four-step wizard (Gradio 6.11, Soft base + gui2/theme.css):

  (1) New Scan     mode cards (Object / Environment), input, subject, Capture QC card,
                   Advanced drawer (validated defaults from gui2/defaults.py)
  (2) Processing   unified weighted progress + per-stage status + live log tail +
                   preview PLACEHOLDER (UI-2 fills the live turntable); Cancel / Resume
  (3) Review       "Mask review — coming in UI-3" (auto-skipped for Environment)
  (4) Deliver      list produced deliverables (name + size) + open-folder button
                   (quality card is UI-4)

The runner (gui2/runner.py) does the real work in a background thread; this file is
only the shell + wiring. One active run at a time is held in a module global (same
pattern as the legacy gui/stages/training.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from gui2 import defaults as D
from gui2 import qc as QC
from gui2 import review as RV
from gui2 import sam_service as SAM
from gui2.runner import (PipelineRunner, RunConfig, STAGE_WEIGHTS, STAGE_LABELS)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_CSS = (_HERE / "theme.css").read_text(encoding="utf-8")
_DATASET = _REPO / "_dataset"

# One active run at a time.
_RUNNER: PipelineRunner | None = None


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def _stepper_html(active: int, mode: str, done: set[int] | None = None) -> str:
    done = done or set()
    labels = ["New scan", "Processing", "Review", "Deliver"]
    cells = []
    for i, lab in enumerate(labels, start=1):
        if i == 3 and mode == "environment":
            lab = "Review (skipped)"
        cls = "apex-step"
        if i in done:
            cls += " done"
        if i == active:
            cls += " active"
        cells.append(f'<div class="{cls}"><span class="num">{i}</span>{lab}</div>')
    return f'<div class="apex-stepper">{"".join(cells)}</div>'


def _progress_html(pct: float, stage_label: str, elapsed_s: float) -> str:
    mm = int(elapsed_s // 60)
    ss = int(elapsed_s % 60)
    return (
        f'<div class="apex-progress-label"><span>{stage_label}</span>'
        f'<span><b>{pct:.0f}%</b> &middot; {mm:d}m {ss:02d}s</span></div>'
        f'<div class="apex-progress-outer"><div class="apex-progress-inner" '
        f'style="width:{max(1.0, pct):.1f}%"></div></div>'
    )


def _stages_html(snap: dict) -> str:
    rows = []
    for key in snap["stage_order"]:
        st = snap["stages"][key]
        cls = st["status"]
        label = snap["stage_labels"].get(key, key)
        frac = st.get("frac", 0.0)
        right = ""
        if st["status"] == "running":
            right = f'{frac * 100:.0f}%'
        elif st["status"] == "done":
            right = "done"
        elif st["status"] == "skipped":
            right = "skipped"
        elif st["status"] == "failed":
            right = "failed"
        desc = f' &middot; {st["desc"]}' if st.get("desc") else ""
        rows.append(f'<li class="{cls}"><span class="apex-dot"></span>{label}'
                    f'<span class="frac">{right}{desc}</span></li>')
    return f'<ul class="apex-stages">{"".join(rows)}</ul>'


def _deliverables_html(items: list[dict], run_dir: str) -> str:
    if not items:
        return ('<div class="apex-panel">No deliverables yet. They appear here when '
                'processing finishes.</div>')
    ico = {".sog": "\U0001F4E6", ".ply": "\U0001F5FF"}
    default_ico = "\U0001F4C4"
    cards = []
    for it in items:
        ext = Path(it["name"]).suffix.lower()
        glyph = ico.get(ext, default_ico)
        cards.append(
            f'<div class="card"><span class="ico">{glyph}</span>'
            f'<div><div class="name">{it["name"]}</div>'
            f'<div class="sub">{it["size_mb"]} MB &middot; {it["path"]}</div></div></div>')
    return f'<div class="apex-deliv">{"".join(cards)}</div>'


def next_step_after_processing(mode: str) -> int:
    """Which wizard step follows Processing. Environment auto-skips the mask-review
    step (3) and goes straight to Deliver (4); Object goes to Review (3).
    (design §3: Review is Object-mode only.) Module-level so it is unit-testable."""
    return 4 if mode == "environment" else 3


def _colmap_num_images(sparse_dir: Path) -> int | None:
    """Registered-image count of a COLMAP model. Reads images.bin's leading uint64
    (num_reg_images) — the cheapest exact count — falling back to counting images.txt
    entries (two lines per image). None if neither is readable."""
    binp = sparse_dir / "images.bin"
    if binp.is_file():
        try:
            import struct
            with open(binp, "rb") as f:
                return int(struct.unpack("<Q", f.read(8))[0])
        except Exception:
            return None
    txtp = sparse_dir / "images.txt"
    if txtp.is_file():
        try:
            n = sum(1 for ln in txtp.read_text().splitlines()
                    if ln.strip() and not ln.startswith("#"))
            return n // 2
        except Exception:
            return None
    return None


def _export_ready_for_review(run_dir: str) -> bool:
    """True only once the SfM export has copied ALL registered images.

    create_3dgs_directory writes sparse/0's .bin files FIRST and copies images AFTER,
    so gating on sparse/0 alone would open review on a half-copied images dir (review
    reads export/images). Gate instead on export/images holding >= the registered image
    count read from the sparse model."""
    exp = Path(run_dir) / "export"
    sparse = exp / "sparse" / "0"
    images = exp / "images"
    if not sparse.is_dir() or not images.is_dir():
        return False
    need = _colmap_num_images(sparse)
    if not need or need <= 0:
        return False
    have = sum(1 for p in images.iterdir()
               if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return have >= need


def _latest_preview(run_dir: str | None) -> str | None:
    """Newest turntable strip under <run_dir>/preview/ (UI-2). Picks by step number
    (falling back to mtime) so a re-render of the same step still refreshes."""
    if not run_dir:
        return None
    pdir = Path(run_dir) / "preview"
    if not pdir.is_dir():
        return None
    strips = list(pdir.glob("strip_*.png"))
    if not strips:
        return None

    def _s(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1
    return str(max(strips, key=lambda p: (_s(p), p.stat().st_mtime)))


# ---------------------------------------------------------------------------
# Advanced drawer <-> settings dict (single source: defaults.SETTINGS)
# ---------------------------------------------------------------------------

def _seed_values(mode: str, detail: bool) -> list:
    d = D.defaults_for(mode, detail)
    out = []
    for spec in D.SETTINGS:
        v = d[spec["key"]]
        if spec["kind"] == "factor":
            out.append(str(v))
        elif spec["kind"] == "choice":
            out.append(v)
        else:
            out.append(v)
    return out


def _collect_settings(mode: str, detail: bool, *vals) -> dict:
    """Rebuild the settings dict from the Advanced-drawer component values."""
    d = D.defaults_for(mode, detail)   # start from mode defaults, override with UI
    for spec, val in zip(D.SETTINGS, vals):
        k, kind = spec["key"], spec["kind"]
        if kind == "factor":
            d[k] = "auto" if str(val) == "auto" else int(val)
        elif kind == "bool":
            d[k] = bool(val)
        elif kind == "int":
            d[k] = int(val)
        elif kind == "float":
            d[k] = float(val)
        elif kind == "choice":
            d[k] = int(val) if k == "filter_harmonics" else val
        else:
            d[k] = val
    return d


def _build_advanced() -> tuple[gr.Accordion, list]:
    comps = []
    with gr.Accordion("Advanced settings (validated defaults pre-filled)",
                      open=False) as acc:
        gr.Markdown("Every value below is the mode's **validated default**. "
                    "Change nothing for a great result.")
        d = D.defaults_for("object", False)
        for spec in D.SETTINGS:
            k, kind = spec["key"], spec["kind"]
            info = f'{D.HELP.get(k, "")}  ·  Evidence: {D.EVIDENCE.get(k, "-")}'
            val = d[k]
            if kind == "factor":
                c = gr.Dropdown(choices=spec["choices"], value=str(val),
                                label=spec["label"], info=info)
            elif kind == "choice":
                c = gr.Dropdown(choices=spec["choices"], value=val,
                                label=spec["label"], info=info)
            elif kind == "bool":
                c = gr.Checkbox(value=bool(val), label=spec["label"], info=info)
            elif kind == "text":
                c = gr.Textbox(value=str(val), label=spec["label"], info=info)
            else:  # int | float
                c = gr.Number(value=val, label=spec["label"], info=info,
                              minimum=spec.get("min"), maximum=spec.get("max"),
                              step=spec.get("step"))
            comps.append(c)
    return acc, comps


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    with gr.Blocks(title="Apex Reality") as app:
        mode_state = gr.State("object")
        detail_state = gr.State(False)
        qc_state = gr.State({})          # {input_dir, source_mp, is_video, n_frames, scan_name}
        rv_ctx = gr.State({})            # UI-3 review context {run_dir, frames_dir, stems, indices, matte_dir, n_frames}
        rv_prompts = gr.State({})        # UI-3 per-frame prompts {frame_idx: {points, labels}}

        with gr.Row(elem_classes="apex-header"):
            gr.HTML('<div class="apex-header"><h1>Apex Reality</h1>'
                    '<p>Clean, usable Gaussian splats from ordinary captures.</p></div>')
        stepper = gr.HTML(_stepper_html(1, "object"))

        # ================= STEP 1 — NEW SCAN =================
        with gr.Column(visible=True) as step1:
            gr.Markdown("### 1 · Start a new scan")
            with gr.Row():
                with gr.Column(elem_classes="apex-modecard", scale=1) as obj_col:
                    obj_btn = gr.Button(
                        "\U0001F9F8  Object Scan\nA single product or subject on a "
                        "surface. Delivered carved out of its background.",
                        elem_id="obj_card")
                with gr.Column(elem_classes="apex-modecard", scale=1) as env_col:
                    env_btn = gr.Button(
                        "\U0001F3E0  Environment Scan\nA room or space. Delivered as the "
                        "full scene, no carving.",
                        elem_id="env_card")
            mode_badge = gr.Markdown("**Mode:** Object Scan")

            with gr.Group(elem_classes="apex-panel"):
                input_path = gr.Textbox(
                    label="Input folder (images) or video file",
                    placeholder=r"D:\path\to\images  —  or  —  D:\path\to\clip.mp4")
                with gr.Row():
                    browse_btn = gr.Button("Browse folder…", scale=1)
                    subject_box = gr.Textbox(label="Subject name (Object mode)",
                                             placeholder="e.g. blue_sneaker", scale=2)
                detail_chk = gr.Checkbox(
                    value=False, label="Detail mode (sharper microstructure, slower — "
                    "~15 MP, pose refinement off)")
                with gr.Row():
                    qc_btn = gr.Button("Run capture check", elem_classes="apex-primary")
                    extract_btn = gr.Button("Extract frames from video", visible=False)

            with gr.Group(elem_classes="apex-panel"):
                qc_md = gr.Markdown("Run the capture check to see frames, resolution, "
                                    "sharpness, and the recommended training resolution.")
                qc_hist = gr.Image(label="Sharpness histogram", visible=False,
                                   height=200)

            adv_acc, adv_comps = _build_advanced()
            with gr.Row():
                reset_btn = gr.Button("Reset to recommended")
                start_btn = gr.Button("Start  →", elem_classes="apex-primary",
                                      variant="primary")
            step1_msg = gr.Markdown("")

        # ================= STEP 2 — PROCESSING =================
        with gr.Column(visible=False) as step2:
            gr.Markdown("### 2 · Processing")
            proc_progress = gr.HTML(_progress_html(0, "Starting…", 0))
            with gr.Row():
                with gr.Column(scale=3):
                    proc_stages = gr.HTML(_stages_html({
                        "stage_order": [k for k, _ in STAGE_WEIGHTS],
                        "stages": {k: {"status": "pending", "frac": 0, "desc": ""}
                                   for k, _ in STAGE_WEIGHTS},
                        "stage_labels": STAGE_LABELS}))
                with gr.Column(scale=2):
                    gr.Markdown("**Live preview** — a 6-view turntable of the current "
                                "checkpoint, refreshed as training saves.")
                    preview_img = gr.Image(label="", height=300, visible=True)
                    with gr.Row():
                        preview_now_btn = gr.Button("Preview now", size="sm")
                        open_viewer_btn = gr.Button("Open interactive preview",
                                                    size="sm")
                    ckpt_dropdown = gr.Dropdown(
                        label="Checkpoint history (render an older step)",
                        choices=[], value=None, interactive=True)
                    preview_status = gr.Markdown("")
                    viewer_html = gr.HTML(visible=False)
            with gr.Accordion("Live log", open=False):
                log_box = gr.Textbox(label="", lines=12, max_lines=12,
                                     interactive=False, elem_classes="apex-log")
            proc_status = gr.Markdown("")
            with gr.Row():
                cancel_btn = gr.Button("Cancel", variant="stop")
                resume_btn = gr.Button("Resume from last stage", visible=False)
                review_now_btn = gr.Button("Review masks (while training)", visible=False)
                continue_btn = gr.Button("Continue →", interactive=False,
                                         elem_classes="apex-primary")
            timer = gr.Timer(1.5, active=True)

        # ================= STEP 3 — REVIEW (object only) =================
        with gr.Column(visible=False) as step3:
            gr.Markdown("### 3 · Mask review — confirm the object cut-out")
            gr.Markdown(
                "These ~12 views span the orbit with the **current object matte** tinted "
                "on top. If it is just your subject, choose **Looks right**. If it grabbed "
                "part of the stand / rig (the classic shoe-stand problem), choose **Fix "
                "it**, click to include (green) or exclude (red), and **Propagate** — the "
                "correction tracks across every view. Masks feed delivery, not training, "
                "so this costs no extra wall-clock.")
            rv_gallery = gr.Gallery(label="Representative views — current matte",
                                    columns=4, height=380, show_label=True)
            rv_status = gr.Markdown("")
            with gr.Row():
                looks_right_btn = gr.Button("Looks right ✓",
                                            elem_classes="apex-primary", variant="primary")
                fix_it_btn = gr.Button("Fix it ✎")

            with gr.Column(visible=False) as rv_edit:
                gr.Markdown("**Edit** — pick a view, choose what your click adds, then "
                            "click on the image. SAM 2.1 re-segments live. Edit as many "
                            "views as you need (the first is the primary anchor), then "
                            "propagate to the whole orbit.")
                with gr.Row():
                    rv_frame_dd = gr.Dropdown(label="View to edit", choices=[], value=None,
                                              scale=2)
                    rv_point_mode = gr.Radio(
                        ["Include (object)", "Exclude (rig)"],
                        value="Include (object)", label="Click adds", scale=2)
                    rv_clear_btn = gr.Button("Clear this view", size="sm", scale=1)
                rv_edit_img = gr.Image(label="Click to add points", height=460,
                                       interactive=True, show_label=True)
                rv_edit_note = gr.Markdown("")
                rv_propagate_btn = gr.Button("Propagate to all views →",
                                             elem_classes="apex-primary")
                rv_spotcheck = gr.Gallery(label="Spot-check after propagation",
                                          columns=4, height=300, visible=False)
                rv_accept_btn = gr.Button("Accept masks ✓",
                                          elem_classes="apex-primary", visible=False)

        # ================= STEP 4 — DELIVER =================
        with gr.Column(visible=False) as step4:
            gr.Markdown("### 4 · Deliverables")
            gr.Markdown("*Quality card + improve-further actions arrive in UI-4.*")
            deliv_html = gr.HTML(_deliverables_html([], ""))
            with gr.Row():
                open_folder_btn = gr.Button("Open deliverables folder",
                                            elem_classes="apex-primary")
                new_scan_btn = gr.Button("New scan")
            deliv_msg = gr.Markdown("")

        # -------------------------------------------------------------------
        # Wiring
        # -------------------------------------------------------------------

        adv_by_key = {spec["key"]: c for spec, c in zip(D.SETTINGS, adv_comps)}

        # Mode/detail changes update ONLY the settings intrinsic to that choice, so a
        # user's other manual Advanced edits are NOT silently clobbered. Mode owns the
        # two delivery fields (extract_object, filter_harmonics); Detail owns target_mp
        # + pose_opt. "Reset to recommended" is the explicit full reload.
        def _pick_mode(mode):
            d = D.defaults_for(mode, False)
            badge = f"**Mode:** {'Object Scan' if mode == 'object' else 'Environment Scan'}"
            note = ("Switched mode — the two delivery settings updated to this mode's "
                    "recommended; your other Advanced edits are kept (use **Reset to "
                    "recommended** to reload everything).")
            return (mode, badge, gr.update(visible=(mode == "object")),
                    _stepper_html(1, mode), bool(d["extract_object"]),
                    int(d["filter_harmonics"]), note)

        _mode_outputs = [mode_state, mode_badge, subject_box, stepper,
                         adv_by_key["extract_object"], adv_by_key["filter_harmonics"],
                         step1_msg]
        obj_btn.click(lambda: _pick_mode("object"), outputs=_mode_outputs)
        env_btn.click(lambda: _pick_mode("environment"), outputs=_mode_outputs)

        def _on_detail(detail, mode):
            d = D.defaults_for(mode, detail)
            note = ("Detail mode ON — ~15 MP training and pose refinement OFF (sharper "
                    "microstructure, slower)." if detail
                    else "Detail mode off — ~6 MP recommended.")
            return float(d["target_mp"]), bool(d["pose_opt"]), note
        detail_chk.change(lambda d: d, inputs=detail_chk, outputs=detail_state).then(
            _on_detail, inputs=[detail_state, mode_state],
            outputs=[adv_by_key["target_mp"], adv_by_key["pose_opt"], step1_msg])

        reset_btn.click(_seed_values, inputs=[mode_state, detail_state], outputs=adv_comps)

        browse_btn.click(_browse_folder, outputs=input_path)

        def _run_qc(path, mode, detail, subject):
            rep = QC.run_qc(path, mode, subject=subject or "", detail=detail)
            hist = (gr.update(value=rep.hist_png, visible=True) if rep.hist_png
                    else gr.update(visible=False))
            qs = {}
            if rep.ok and not rep.is_video:
                qs = {"input_dir": path, "source_mp": rep.megapixels,
                      "is_video": False, "n_frames": rep.n_frames}
            elif rep.ok and rep.is_video:
                qs = {"video_path": path, "source_mp": rep.megapixels,
                      "is_video": True, "n_frames": rep.n_frames}
            md = rep.summary_md if rep.ok else f"**{rep.message}**"
            return (md, hist, qs, gr.update(visible=rep.is_video))

        qc_btn.click(_run_qc,
                     inputs=[input_path, mode_state, detail_chk, subject_box],
                     outputs=[qc_md, qc_hist, qc_state, extract_btn])

        def _extract(path, subject, mode, detail):
            scan = _scan_name(subject, path)
            out = _DATASET / scan / "input_images"
            # clears `out` first, extracts all frames, keeps the sharpest-per-window
            kept, _report = QC.extract_and_select_frames(path, str(out))
            rep = QC.run_qc(str(out), mode, subject=subject or "", detail=detail)
            qs = {"input_dir": str(out), "source_mp": rep.megapixels,
                  "is_video": False, "n_frames": rep.n_frames, "scan_name": scan}
            hist = (gr.update(value=rep.hist_png, visible=True) if rep.hist_png
                    else gr.update(visible=False))
            return rep.summary_md, hist, qs

        extract_btn.click(_extract,
                          inputs=[input_path, subject_box, mode_state, detail_chk],
                          outputs=[qc_md, qc_hist, qc_state])

        # ---- Start ----
        def _start(mode, detail, subject, qs, *adv_vals):
            global _RUNNER
            if _RUNNER is not None and _RUNNER.is_alive():
                return (gr.update(), gr.update(), gr.update(),
                        "**A scan is already running — cancel it (Step 2) before "
                        "starting another.** (One run at a time; two-tabs-two-scans is "
                        "out of scope for UI-1.)", _stepper_html(1, mode))
            if not qs or not qs.get("input_dir"):
                return (gr.update(), gr.update(), gr.update(),
                        "**Run the capture check on a folder (or extract video "
                        "frames) first.**", _stepper_html(1, mode))
            settings = _collect_settings(mode, detail, *adv_vals)
            scan = qs.get("scan_name") or _scan_name(subject, qs["input_dir"])
            run_dir = _DATASET / scan
            run_dir.mkdir(parents=True, exist_ok=True)
            cfg = RunConfig(
                scan_name=scan, mode=("object" if mode == "object" else "environment"),
                input_dir=qs["input_dir"], run_dir=str(run_dir), settings=settings,
                source_mp=float(qs.get("source_mp", 0.0)), subject=subject or "",
                detail=detail)
            _RUNNER = PipelineRunner(cfg)
            _RUNNER.start()
            return (gr.update(visible=False), gr.update(visible=True),
                    gr.update(), f"Started **{scan}** ({mode}).",
                    _stepper_html(2, mode, done={1}))

        start_btn.click(
            _start,
            inputs=[mode_state, detail_state, subject_box, qc_state, *adv_comps],
            outputs=[step1, step2, step3, step1_msg, stepper])

        # ---- Timer tick ----
        def _tick(mode):
            if _RUNNER is None:
                return (gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update())
            snap = _RUNNER.snapshot()
            prog = _progress_html(snap["global_pct"], snap["current_stage_label"],
                                  snap["elapsed_s"])
            stages = _stages_html(snap)
            log = snap["log_tail"]
            prev = _latest_preview(_RUNNER.cfg.run_dir)
            prev_up = gr.update(value=prev) if prev else gr.update()
            # Refresh the checkpoint-history dropdown; gr.update(choices=…) keeps the
            # user's current selection if that path is still present.
            cks = _RUNNER.list_checkpoints()
            ck_up = gr.update(choices=[(c["label"], c["path"]) for c in cks])
            done = snap["status"] == "done"
            if done:
                nxt = ("Continue to review →" if mode == "object"
                       else "See results →")
                cont = gr.update(interactive=True, value=nxt, variant="primary")
                status = f"**{snap['message']}**"
            elif snap["status"] in ("failed", "cancelled"):
                cont = gr.update(interactive=False)
                status = f"**{snap['message']}**"
            else:
                cont = gr.update(interactive=False)
                status = snap["message"]
            # Only push preview_status on a systematic-failure note; otherwise leave it
            # (no-op) so a button's transient message isn't clobbered every tick.
            note = snap.get("preview_note", "")
            pv_note = gr.update(value=f"⚠ {note}") if note else gr.update()
            # UI-3 parallelism: enable "Review masks" for Object scans once the SfM
            # export has copied ALL registered images (review reads export/images) and
            # while the run isn't finished. Gating on export/images completeness (not on
            # sparse/0, which is written first) avoids opening review on a half-copied set.
            rv_now = gr.update()
            if mode == "object":
                rv_now = gr.update(
                    visible=_export_ready_for_review(_RUNNER.cfg.run_dir) and not done)
            return prog, stages, log, prev_up, cont, status, ck_up, pv_note, rv_now

        timer.tick(_tick, inputs=[mode_state],
                   outputs=[proc_progress, proc_stages, log_box, preview_img,
                            continue_btn, proc_status, ckpt_dropdown, preview_status,
                            review_now_btn])

        # ---- Live-preview controls (UI-2) ----
        def _preview_now():
            if _RUNNER is None:
                return gr.update(), "No run yet."
            path, msg = _RUNNER.preview_now(wait=True, timeout=45)
            return (gr.update(value=path) if path else gr.update()), msg
        preview_now_btn.click(_preview_now,
                              outputs=[preview_img, preview_status])

        def _preview_ckpt(ckpt_path):
            if _RUNNER is None or not ckpt_path:
                return gr.update(), "Pick a checkpoint from the list."
            path, msg = _RUNNER.preview_now(ckpt=ckpt_path, wait=True, timeout=45)
            return (gr.update(value=path) if path else gr.update()), msg
        ckpt_dropdown.change(_preview_ckpt, inputs=[ckpt_dropdown],
                             outputs=[preview_img, preview_status])

        def _open_viewer():
            if _RUNNER is None:
                return gr.update(visible=False), "No run yet."
            sog, msg = _RUNNER.export_viewer()
            if not sog:
                return gr.update(visible=False), msg
            return gr.update(value=_viewer_iframe_html(sog), visible=True), msg
        open_viewer_btn.click(_open_viewer,
                              outputs=[viewer_html, preview_status])

        cancel_btn.click(_cancel, outputs=[proc_status, resume_btn])

        def _resume(mode, detail, subject, qs, *adv_vals):
            global _RUNNER
            settings = _collect_settings(mode, detail, *adv_vals)
            scan = qs.get("scan_name") or _scan_name(subject, qs.get("input_dir", ""))
            run_dir = _DATASET / scan
            cfg = RunConfig(
                scan_name=scan, mode=("object" if mode == "object" else "environment"),
                input_dir=qs["input_dir"], run_dir=str(run_dir), settings=settings,
                source_mp=float(qs.get("source_mp", 0.0)), subject=subject or "",
                detail=detail, resume=True)
            _RUNNER = PipelineRunner(cfg)
            _RUNNER.start()
            return gr.update(visible=False), "Resuming…"
        resume_btn.click(
            _resume,
            inputs=[mode_state, detail_state, subject_box, qc_state, *adv_comps],
            outputs=[resume_btn, proc_status])

        # ---- navigation from processing ----
        def _after_processing(mode):
            if next_step_after_processing(mode) == 3:
                return (gr.update(visible=False), gr.update(visible=True),
                        gr.update(visible=False), _stepper_html(3, mode, done={1, 2}),
                        gr.update())
            # environment: skip step 3 -> deliver
            items = _RUNNER.snapshot()["deliverables"] if _RUNNER else []
            rd = _RUNNER.cfg.run_dir if _RUNNER else ""
            return (gr.update(visible=False), gr.update(visible=False),
                    gr.update(visible=True), _stepper_html(4, mode, done={1, 2, 3}),
                    _deliverables_html(items, rd))
        continue_btn.click(
            _after_processing, inputs=[mode_state],
            outputs=[step2, step3, step4, stepper, deliv_html])
        # (object) populate the review gallery once Step-③ is shown — registered in the
        # UI-3 wiring block below, after _init_review is defined.

        # ---- UI-3 Mask Review wiring ----
        def _init_review(mode):
            """Prepare working frames + representative-view mattes + the gallery."""
            if mode != "object" or _RUNNER is None:
                return gr.update(), {}, gr.update(choices=[]), gr.update()
            run_dir = _RUNNER.cfg.run_dir
            try:
                fdir, stems, _images = RV.prepare_working_frames(run_dir)
            except Exception as e:
                return (gr.update(value=[]), {}, gr.update(choices=[]),
                        f"**Could not load frames for review: {e}**")
            n = len(stems)
            idxs = RV.spread_indices(n, 12)
            matte_dir = str(Path(run_dir) / "sam_work" / "mattes")
            existing = None
            dd = getattr(_RUNNER, "data_dir", None)
            df = getattr(_RUNNER, "data_factor", None)
            if dd and df:
                existing = str(Path(dd) / f"masks_{df}")
            RV.ensure_mattes(fdir, stems, idxs, matte_dir, existing_mask_dir=existing)
            items = RV.gallery_items(fdir, stems, idxs, matte_dir, tint=RV.BIREFNET_TINT)
            ctx = {"run_dir": run_dir, "frames_dir": fdir, "stems": stems,
                   "indices": idxs, "matte_dir": matte_dir, "n_frames": n}
            dd_up = gr.update(choices=[(f"view {i} · {stems[i]}", i) for i in idxs],
                              value=idxs[len(idxs) // 2])
            return (items, ctx, dd_up,
                    f"Orbit has {n} views; showing {len(idxs)} spread across it. "
                    "Green tint would be your object; red tint = the current matte "
                    "(which may still include the stand).")

        def _pick_edit_frame(idx, ctx, prompts):
            if not ctx or idx is None:
                return gr.update(), "Pick a view to edit."
            i = int(idx)
            frame = RV.load_frame(ctx["frames_dir"], i)
            entry = prompts.get(i)
            pts = entry["points"] if entry else []
            lbl = entry["labels"] if entry else []
            mask = (SAM.prompt_frame(frame, pts, lbl, image_key=f"{ctx['run_dir']}::{i}")
                    if pts else None)
            ov = RV.overlay_rgb(frame, mask, RV.INCLUDE_TINT, pts, lbl)
            return ov, f"Editing view {i} ({ctx['stems'][i]}) — {len(pts)} point(s)."

        def _edit_click(evt: gr.SelectData, mode, idx, ctx, prompts):
            if not ctx or idx is None:
                return gr.update(), prompts, "Pick a view first."
            i = int(idx)
            x, y = int(evt.index[0]), int(evt.index[1])
            label = 1 if str(mode).startswith("Include") else 0
            prompts = dict(prompts)
            e = prompts.get(i, {"points": [], "labels": []})
            entry = {"points": list(e["points"]) + [[x, y]],
                     "labels": list(e["labels"]) + [label]}
            prompts[i] = entry
            frame = RV.load_frame(ctx["frames_dir"], i)
            mask = SAM.prompt_frame(frame, entry["points"], entry["labels"],
                                    image_key=f"{ctx['run_dir']}::{i}")
            ov = RV.overlay_rgb(frame, mask, RV.INCLUDE_TINT, entry["points"], entry["labels"])
            ninc = sum(l == 1 for l in entry["labels"])
            nexc = sum(l == 0 for l in entry["labels"])
            return ov, prompts, f"View {i}: {ninc} include, {nexc} exclude."

        def _clear_view(idx, ctx, prompts):
            if not ctx or idx is None:
                return gr.update(), prompts, ""
            i = int(idx)
            prompts = dict(prompts)
            prompts.pop(i, None)
            frame = RV.load_frame(ctx["frames_dir"], i)
            return RV.overlay_rgb(frame, None, RV.INCLUDE_TINT), prompts, f"Cleared view {i}."

        def _propagate(ctx, prompts):
            if not ctx or not prompts:
                return (gr.update(visible=False), gr.update(visible=False),
                        "Add at least one include point on an object view first.", ctx)
            # First-edited frame = primary anchor (unlimited); the rest are LOCAL
            # corrections (they only override a window around their seed).
            plist = []
            for k, (fi, entry) in enumerate(prompts.items()):
                p = {"frame_idx": int(fi), "points": entry["points"],
                     "labels": entry["labels"]}
                if k > 0:
                    p["radius"] = SAM.DEFAULT_CORRECTION_RADIUS
                plist.append(p)
            work = Path(ctx["run_dir"]) / "sam_work"
            out_dir = str(work / "masks_sam")
            stats_out = str(work / "propagate_stats.json")
            log_path = work / "propagate.log"
            # Drop any stale stats file so a prior run's success can't masquerade as
            # this run's (the stats file is written LAST, only on a clean propagate).
            if os.path.isfile(stats_out):
                os.remove(stats_out)
            with open(str(log_path), "w") as logf:
                proc = RV.propagate_subprocess(ctx["frames_dir"], plist, out_dir,
                                               stats_out, log_file=logf)
                proc.wait()
            # Surface a failed propagation and BLOCK Accept: nonzero exit or a missing
            # stats file means the masks are unusable — keep the accept button hidden
            # and clear any prop_out_dir so a stale prior result can't be accepted.
            if proc.returncode != 0 or not os.path.isfile(stats_out):
                tail = ""
                try:
                    tail = log_path.read_text(errors="ignore")[-800:]
                except Exception:
                    pass
                ctx = dict(ctx)
                ctx.pop("prop_out_dir", None)
                msg = (f"**Propagation failed (exit {proc.returncode}).** No masks were "
                       f"produced — nothing to accept. Adjust the prompts and try again, "
                       f"or see `sam_work/propagate.log`.")
                if tail:
                    msg += f"\n\n```\n{tail}\n```"
                return (gr.update(visible=False), gr.update(visible=False), msg, ctx)
            ctx = dict(ctx)
            ctx["prop_out_dir"] = out_dir
            spot = RV.spread_indices(ctx["n_frames"], 4)
            items = RV.gallery_items(ctx["frames_dir"], ctx["stems"], spot, out_dir,
                                     tint=RV.REVIEWED_TINT)
            import json
            stats = json.load(open(stats_out))
            msg = (f"Propagated shoe-only masks to {stats.get('n_masks', '?')} views in "
                   f"{stats.get('seconds', '?')}s (seeds {stats.get('seed_idxs', [])}). "
                   "Check the spot-check strip, then Accept or edit another view.")
            return (gr.update(value=items, visible=True), gr.update(visible=True), msg, ctx)

        def _leave_review(mode, message):
            done = _RUNNER is not None and _RUNNER.snapshot()["status"] == "done"
            if done:
                items = _RUNNER.snapshot()["deliverables"] if _RUNNER else []
                rd = _RUNNER.cfg.run_dir if _RUNNER else ""
                return (gr.update(visible=False), gr.update(visible=False),
                        gr.update(visible=True), _stepper_html(4, mode, done={1, 2, 3}),
                        _deliverables_html(items, rd), message)
            # training still running -> back to processing; masks applied at delivery
            return (gr.update(visible=False), gr.update(visible=True),
                    gr.update(visible=False), _stepper_html(2, mode, done={1}),
                    gr.update(), message + " Training continues.")

        _leave_outputs = [step3, step2, step4, stepper, deliv_html, rv_status]

        def _accept(mode, ctx):
            if ctx and ctx.get("prop_out_dir"):
                d, n = RV.accept_reviewed(ctx["run_dir"], ctx["prop_out_dir"], ctx["stems"])
                msg = f"Accepted {n} reviewed masks — delivery will carve from these."
            else:
                msg = "No propagated masks to accept."
            return _leave_review(mode, msg)

        def _looks_right(mode, ctx):
            if ctx and ctx.get("run_dir"):
                RV.clear_reviewed(ctx["run_dir"])
            return _leave_review(mode, "Current matte accepted — delivery will use it.")

        # Populate the review gallery when the sequential (post-processing) path reaches
        # Step-③ (runs after _after_processing's handler; no-op for Environment).
        continue_btn.click(_init_review, inputs=[mode_state],
                           outputs=[rv_gallery, rv_ctx, rv_frame_dd, rv_status])

        looks_right_btn.click(_looks_right, inputs=[mode_state, rv_ctx],
                              outputs=_leave_outputs)
        fix_it_btn.click(lambda: gr.update(visible=True), outputs=rv_edit).then(
            _pick_edit_frame, inputs=[rv_frame_dd, rv_ctx, rv_prompts],
            outputs=[rv_edit_img, rv_edit_note])
        rv_frame_dd.change(_pick_edit_frame, inputs=[rv_frame_dd, rv_ctx, rv_prompts],
                           outputs=[rv_edit_img, rv_edit_note])
        rv_edit_img.select(_edit_click,
                           inputs=[rv_point_mode, rv_frame_dd, rv_ctx, rv_prompts],
                           outputs=[rv_edit_img, rv_prompts, rv_edit_note])
        rv_clear_btn.click(_clear_view, inputs=[rv_frame_dd, rv_ctx, rv_prompts],
                           outputs=[rv_edit_img, rv_prompts, rv_edit_note])
        rv_propagate_btn.click(_propagate, inputs=[rv_ctx, rv_prompts],
                               outputs=[rv_spotcheck, rv_accept_btn, rv_edit_note, rv_ctx])
        rv_accept_btn.click(_accept, inputs=[mode_state, rv_ctx], outputs=_leave_outputs)

        # Enter Review from Step-② parallelism button (object, once SfM export exists).
        def _enter_review_parallel(mode):
            return gr.update(visible=False), gr.update(visible=True), _stepper_html(3, mode, done={1})
        review_now_btn.click(_enter_review_parallel, inputs=[mode_state],
                             outputs=[step2, step3, stepper]).then(
            _init_review, inputs=[mode_state],
            outputs=[rv_gallery, rv_ctx, rv_frame_dd, rv_status])

        open_folder_btn.click(_open_deliverables, outputs=deliv_msg)

        def _new_scan(mode):
            return (gr.update(visible=True), gr.update(visible=False),
                    _stepper_html(1, mode))
        new_scan_btn.click(_new_scan, inputs=[mode_state],
                           outputs=[step1, step4, stepper])

    return app


# ---------------------------------------------------------------------------
# small handlers (module-level so they're importable/testable)
# ---------------------------------------------------------------------------

def _scan_name(subject: str, path: str) -> str:
    raw = subject.strip() if subject and subject.strip() else Path(path).stem
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in raw) or "scan"


def _browse_folder() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            folder = filedialog.askdirectory(title="Select image folder")
        finally:
            root.destroy()
        return folder or ""
    except Exception:
        return ""


def _cancel():
    if _RUNNER is not None:
        _RUNNER.cancel()
        return "Cancelling…", gr.update(visible=True)
    return "Nothing to cancel.", gr.update(visible=False)


def _open_deliverables() -> str:
    if _RUNNER is None:
        return "No run yet."
    folder = str(Path(_RUNNER.cfg.run_dir) / "deliverables")
    try:
        os.startfile(folder)  # noqa: E1101 (Windows)
        return f"Opened {folder}"
    except Exception as e:
        return f"Folder: {folder} ({e})"


def _viewer_iframe_html(sog_path: str) -> str:
    """Build the same-origin SuperSplat iframe for an exported .sog. The scene file is
    served from the `/scenes` static mount (= _dataset), the viewer bundle from
    `/viewer`; both share the Gradio origin, so `?content=` resolves with no CORS or
    mixed-content problem (the hosted superspl.at viewer is CORS-blocked for localhost —
    see _infusion/ui2_RECORD.md)."""
    from urllib.parse import quote
    try:
        rel = Path(sog_path).resolve().relative_to(_DATASET.resolve()).as_posix()
        content = "/scenes/" + quote(rel)
    except ValueError:  # outside _dataset — best-effort via Gradio's own file route
        content = "/gradio_api/file=" + quote(str(Path(sog_path).resolve()))
    src = f"/viewer/index.html?content={content}"
    return (
        f'<iframe src="{src}" title="Interactive splat preview" '
        f'style="width:100%;height:70vh;min-height:440px;border:0;border-radius:12px;'
        f'background:#111;" allow="xr-spatial-tracking; fullscreen"></iframe>'
        f'<div style="font-size:12px;opacity:0.65;margin-top:6px;">Drag to orbit · '
        f'scroll to zoom · right-drag to pan.</div>')


def launch(server_port: int = 7861):
    """Serve the wizard + the vendored SuperSplat viewer + scene files on ONE origin
    (127.0.0.1:port). Same-origin serving is what makes the interactive viewer work:
    the hosted superspl.at viewer is CORS-blocked from fetching localhost files, so we
    self-host PlayCanvas's open-source `@playcanvas/supersplat-viewer` bundle under
    /viewer and the exported .sog under /scenes (= _dataset)."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    _DATASET.mkdir(parents=True, exist_ok=True)
    fastapi_app = FastAPI()
    fastapi_app.mount("/viewer",
                      StaticFiles(directory=str(_HERE / "viewer")), name="viewer")
    fastapi_app.mount("/scenes",
                      StaticFiles(directory=str(_DATASET)), name="scenes")

    demo = create_app()
    demo.queue(default_concurrency_limit=4)
    app = gr.mount_gradio_app(fastapi_app, demo, path="/",
                              theme=gr.themes.Soft(), css=_CSS)
    uvicorn.run(app, host="127.0.0.1", port=server_port)
