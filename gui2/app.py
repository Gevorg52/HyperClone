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
                continue_btn = gr.Button("Continue →", interactive=False,
                                         elem_classes="apex-primary")
            timer = gr.Timer(1.5, active=True)

        # ================= STEP 3 — REVIEW (object only) =================
        with gr.Column(visible=False) as step3:
            gr.Markdown("### 3 · Mask review")
            gr.Markdown("**Mask review — coming in UI-3.** Here you will confirm the "
                        "object cut-out (BiRefNet overlay) and fix it with SAM 2.1 "
                        "point-prompts. Masks feed delivery, not training, so this runs "
                        "in parallel and costs no extra time.")
            to_deliver_btn = gr.Button("Continue to results →",
                                       elem_classes="apex-primary")

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
            return prog, stages, log, prev_up, cont, status, ck_up, pv_note

        timer.tick(_tick, inputs=[mode_state],
                   outputs=[proc_progress, proc_stages, log_box, preview_img,
                            continue_btn, proc_status, ckpt_dropdown, preview_status])

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

        def _to_deliver(mode):
            items = _RUNNER.snapshot()["deliverables"] if _RUNNER else []
            rd = _RUNNER.cfg.run_dir if _RUNNER else ""
            return (gr.update(visible=False), gr.update(visible=True),
                    _stepper_html(4, mode, done={1, 2, 3}),
                    _deliverables_html(items, rd))
        to_deliver_btn.click(_to_deliver, inputs=[mode_state],
                             outputs=[step3, step4, stepper, deliv_html])

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
