# HYPERclone GUI 2.0 — Design Plan (July 2026, for validation)

**Goal:** replace the 10-tab expert GUI with a modern, simple, wizard-style app where the
validated-optimal settings are the defaults, an untrained user gets a great splat with zero
knob-turning, and an expert can still reach every lever. Two product modes: **Object Scan** and
**Environment Scan**. Status: VALIDATED by user 2026-07-07 — Gradio-6 rebuild / SAM point-prompts / auto-strip preview all approved as recommended. **UI-1 BUILT + smoke-gated 2026-07-07** (`gui2/`, `run2.py`; evidence `_infusion/ui1_RECORD.md`). **UI-2 BUILT + gated 2026-07-08** (live preview strip + interactive SuperSplat viewer; evidence `_infusion/ui2_RECORD.md`). **UI-3 BUILT + gated 2026-07-09** (SAM 2.1 mask review: point-prompts + video propagation → shoe-only masks; evidence `_infusion/ui3_RECORD.md`). UI-4 pending.

---

## 1. Architecture decision (recommendation)

**Rebuild in Gradio 6 as a fresh app (`gui2/`), keep the old GUI untouched until parity.**
- WHY Gradio (POC phase): the pipeline is Python + subprocesses; Gradio speaks to it natively with
  zero API layer; progress/galleries/file-drops are built-in; we already ship Gradio 6.11; a
  React+FastAPI app is the *funded-phase* rebuild (this design's information architecture carries
  over 1:1).
- The one thing Gradio lacks — an interactive 3D splat viewer — we embed: **PlayCanvas SuperSplat
  viewer in an iframe** (`gr.HTML`), fed by a tiny static file route serving the latest exported
  `.sog`. Interactivity where it matters, Python everywhere else.
- Visual identity: Gradio Soft theme + custom CSS (dark, generous whitespace, one accent color),
  wizard layout (single page, stepped), no tab farm.

## 2. The two modes (defaults = the validated stack)

| Setting | **Object Scan** | **Environment Scan** | Evidence |
|---|---|---|---|
| Training data | **FULL scene** (never masked) | full scene | WP-S2: masking costs +24% obj-LPIPS via pose drift |
| data_factor | auto → nearest **~6 MP** | auto → ~6 MP | validated regime (WP-S1/T1) |
| Iterations / cap | 30k / MCMC 3M | 30k / 3M | convergence + capacity evidence (X1, N1) |
| Levers ON | dense_init · mono_depth 0.05 · erank 0.01@7k · gated-LPIPS 0.2 · pose_opt · antialiased · c2f | same | WP-S1 stack (all compose, no interactions) |
| Planar losses | off | off (exposed in Advanced: "flatten walls/floors" — geometry use only) | WP-F4: intrinsic perceptual cost |
| Delivery | **object SOG (SH3, extracted)** + full SOG (SH2, decimated) | full SOG + PLY master | WP-D1 gates |
| Extra step | **Mask Review** (see §4) | — | shoe-stand problem |
| "Detail mode" toggle | factor→~15 MP + **pose_opt OFF** (slower, sharper microstructure) | same | WP-N1 native recipe |

## 3. The wizard — four steps, one screen each

**① NEW SCAN** — mode picker (two big cards: Object / Environment) · drag-drop a folder or video ·
instant **Capture QC card** (runs in seconds, BEFORE the user commits hours): frame count,
resolution, sharpness histogram (masked VoL), duplicate/blur cull preview, estimated processing
time, and for Object mode a coarse coverage estimate ("mostly single ring — expect a weaker
underside"). One primary button: **Start**. Advanced drawer (collapsed): every trainer flag,
pre-filled with the mode defaults; a "reset to recommended" button.

**② PROCESSING** — one unified progress bar with weighted stages
(QC 2% · features 8% · matching 25% · reconstruction 20% · export 5% · priors+dense-init 10% ·
training 28% · delivery 2%), per-stage status line + ETA (learned from run history), live log tail
(collapsed by default). **Live training preview panel:** every N minutes (default 5) the trainer's
latest checkpoint renders a 6-frame mini-turntable strip that refreshes in-place — the user *sees*
the splat improving; a "Preview now" button forces a fresh strip; from the first PLY export onward
an **"Open interactive preview"** button loads the current model in the embedded viewer.
Pause / Resume / Cancel (training checkpoints make pause safe). Progress plumbing: stages already
emit `progress` callbacks; the trainer subprocess's step lines are parsed (`step=N` → fraction);
checkpoint-save events trigger the preview refresh.

**③ REVIEW (Object mode only; auto-skipped for Environment)** — the shoe-stand fix:
- Gallery of ~12 representative views (coverage-spread) with the BiRefNet matte overlaid
  (tinted). Verdict buttons: **Looks right** → continue · **Fix it** → edit mode.
- **Edit mode = SAM 2.1 point prompts:** click on the model = *include* (green point), right/alt
  click = *exclude* (red point) — e.g. one red click on the stand removes it. Prompts run SAM 2.1
  on that frame instantly; **propagation** to all 588 views via SAM 2's native video tracking
  (frames ordered by orbit sequence = a video). Re-render 4 spot-check overlays → accept.
- Accepted masks feed the delivery-time extraction (vote-carve + masked refit). NOTE: masks are
  a DELIVERY input, not a training input (WP-S2) — so mask review runs in parallel with training,
  costing zero wall-clock.
- Advanced: per-frame brush touch-up (deferred to funded phase; point-prompts cover the real cases).

**④ DELIVER** — quality card (4 friendly numbers: object sharpness [region-LPIPS→star scale],
geometric consistency [TSED], floaters [SOR %], views used) + honest flags (e.g. "underside was
weakly observed") · deliverable cards: **Object .sog** / **Full scene .sog** / **Master .ply**,
each with size, an embedded-viewer button, a file-location button · "Improve further" actions:
Detail-mode re-run · re-capture report (coverage gaps → "shoot these angles") · floater cleanup.

## 4. Settings inventory (what's exposed where)

- **Step 1 visible:** mode, input, (Object) subject name. That's it.
- **Advanced drawer (per mode, pre-filled):** data_factor, iterations, cap_max, every lever flag
  with one-line plain-language help + the evidence citation, seed, detail-mode toggle,
  eval split, delivery options (SH level, decimation %, extraction on/off).
- **Utilities menu (top-right):** Capture QC (standalone) · Coverage/re-capture report · Floater
  cleanup (existing prune) · QA scorecard (full 6-scorer, expert) · Compare two models (A/B strips)
  · Open deliverables folder.
- **Hidden entirely:** anything failing its gate (WD-R, SR injection, ungated LPIPS…), debug flags.

## 5. Implementation plan (each WP gated + reviewed, per house rules)

| WP | Scope | Est |
|---|---|---|
| **UI-1** ✅ | `gui2/` skeleton: wizard shell, mode cards, QC card, Advanced drawer w/ validated defaults, pipeline runner (reuses stage handlers headless) + unified progress plumbing | 2–3 d |
| **UI-2** ✅ | Training preview system: checkpoint→turntable strip renderer (subprocess, cheap), auto-refresh, Preview-now, ckpt-history dropdown; same-origin static route + self-hosted SuperSplat `.sog` iframe embed. Gated 2026-07-08 (`_infusion/ui2_RECORD.md`). | 2 d |
| **UI-3** ✅ | Mask Review: representative-view picker, matte overlay gallery, SAM 2.1 point-prompt service, video-propagation, delivery re-carve hook. Gated 2026-07-09 (`_infusion/ui3_RECORD.md`). | 3–4 d |
| **UI-4** | Deliver screen: quality card mapping, deliverable cards + viewer, "improve further" actions; Utilities menu | 2 d |

Total ≈ 2 weeks of agent time; UI-1 alone is already usable end-to-end. Old GUI stays until
parity, then `run.py` flips to `gui2`.

### 5.1 UI-1 as built (2026-07-07) + hook points for UI-2/3/4

Files (`gui2/`): `defaults.py` (mode defaults = single source of truth), `qc.py` (Capture
QC), `runner.py` (`PipelineRunner`, full headless chain + unified weighted progress +
cancel/resume), `app.py` (wizard shell), `theme.css`; launcher `run2.py` (port 7861, legacy
`run.py` stays on 7860). Smoke-gated end-to-end (Env mode, 39-frame subset, factor 4, 500
steps): monotonic progress through all 8 weighted stages, 2 deliverables, defaults byte-for-byte,
Env skips Review. Evidence: `_infusion/ui1_RECORD.md`.

**Hook points for later WPs (stable APIs the skeleton already exposes):**
- **UI-2 (live preview + interactive viewer):** `runner.PipelineRunner` — add a
  checkpoint-save callback in the training stage (a strip renderer subprocess) that writes to a
  `<run_dir>/preview/` dir; `app._latest_preview(run_dir)` already surfaces the newest PNG into
  the Step-② `preview_img` on the `gr.Timer` tick. A "Preview now" button wires to a new runner
  method; the SuperSplat iframe mounts in Step② where `preview_img` sits (add a static file route
  for the latest `.ply`/`.sog` — `runner.ckpt_path` / delivery output give the source).
  Pause is documented as unavailable on Windows — the skeleton ships **Cancel + resume-from-stage**
  (`runner.cancel()` + `RunConfig.resume=True`, which skips stages whose on-disk artifacts exist).
- **UI-3 (mask review, Object only):** Step③ container already exists and is auto-skipped for
  Environment via `app.next_step_after_processing(mode)` (returns 3 for object, 4 for env) — the
  single place that policy lives. Masks are a DELIVERY input: the runner passes
  `assemble_delivery_flags()` → `--extract_object --gen_masks` for Object mode; UI-3 replaces
  `--gen_masks` with reviewed masks written under the run dir, and delivery's `--mask_dir` picks
  them up. Representative-view picking can reuse the QC frame sample.
- **UI-4 (quality card + utilities):** Step④ (`deliv_html`) already lists deliverables (name/
  size/path) from `runner.snapshot()["deliverables"]`. UI-4 adds the quality card by reading the
  delivery QA JSON (`<run_dir>/deliverables/<name>_<mode>_delivery.json`, produced when
  `delivery_no_qa=False`) + `qa_report` scorecards; the Utilities menu wraps existing CLIs.
- **Settings:** any new lever is one row in `defaults.SETTINGS` (+ `HELP`/`EVIDENCE`) and, if it
  maps to the trainer, one line in `defaults.assemble_training_extra()`. The Advanced drawer,
  reset button, and mode/detail reseeding all derive from `SETTINGS` automatically.

### 5.2 UI-2 as built (2026-07-08) — live preview + interactive viewer

New: `gui2/preview.py` (subprocess strip renderer + ckpt→SOG export) and
`gui2/viewer/` (vendored PlayCanvas `@playcanvas/supersplat-viewer` v1.27.1 bundle, MIT).
Runner + app wired per §5.1. Gate PASSED (5/5): evidence `_infusion/ui2_RECORD.md`
(+ `ui2_gate_trace.json`, `ui2_burst.json`, `ui2_viewer_http_probe.json`,
`ui2_strip_step{124,499}.png`).

- **Strip renderer** (`gui2/preview.py::render_strip`, subprocess `python -m gui2.preview strip`),
  mode-specific: **OBJECT** = synthesized 6-view turntable (spread azimuths at the median captured
  elevation around the object center — `coverage.object_center_from_cameras` + orbit basis,
  point-median fallback); **ENVIRONMENT** = 6 REAL training-camera poses spread across the capture
  (`select_real_poses`) — a synthesized orbit around a divergent scene can point into
  unreconstructed void, real poses can't. Composited 3×2 (~720px tall, `--cols`/`--height`),
  reusing the exact `gsplat_trainer.render_splats` path. GPU burst **~0.24 s even for a 3M-Gaussian
  model** — the rest of the subprocess is CPU-side import/COLMAP-load, so the trainer never stalls
  (measured <1 s vs the 5 s bar; re-runnable via `_infusion/ui2_burst_measure.py`).
- **Runner hooks** (`gui2/runner.py`): `_train_cb` → `_maybe_render_preview()` (debounced 1.5 s,
  skips if a render is in flight; rides UI-1's `save_all` quarterly ckpts) writes
  `<run_dir>/preview/strip_<step>.png`. On CLEAN completion, `_render_final_strip()` waits (bounded
  15 s) for any in-flight render, then renders the true final strip unconditionally (a CANCELLED
  run has no "final" and correctly skips it). 3 consecutive render failures surface a visible note
  (`preview_note` → Step②). Public: `preview_now(ckpt=None)`, `list_checkpoints()`,
  `export_viewer()`. `cancel()` also tree-kills a live preview.
- **Interactive viewer**: served **same-origin** — `launch()` mounts `/viewer` (bundle) + `/scenes`
  (= `_dataset`) as FastAPI StaticFiles via `gr.mount_gradio_app`; the iframe is
  `/viewer/index.html?content=/scenes/<scan>/preview/viewer_<step>.sog`. The HOSTED superspl.at
  viewer is a dead end for localhost (CORS — Gradio withholds `Access-Control-Allow-Origin` from
  external origins by design; mixed content is exempt for loopback), hence self-hosting.
- **Step② panel**: preview image (auto-refresh on the Timer tick via `app._latest_preview` reading
  `<run_dir>/preview/`), **Preview now**, **Open interactive preview**, **checkpoint-history
  dropdown** (render a strip from an older ckpt — cheap time-travel).
- **Hook for UI-3/4:** the same `/scenes` static mount + `_viewer_iframe_html()` embed serve the
  Step④ deliverable viewers and any Step③ overlay files with zero extra plumbing.
- **Answers §6-Q3:** auto-refreshing turntable strip + on-demand interactive viewer is what shipped
  (not a permanently-docked heavier viewer) — the user-validated balance.

### 5.3 UI-3 as built (2026-07-09) — Mask Review (SAM 2.1 point-prompts + video propagation)

New: `gui2/sam_service.py` (SAM 2.1 service) and `gui2/review.py` (review logic); Step-③
in `gui2/app.py` is now the real screen (was a stub). Delivery wired via `gui2/defaults.py`
+ `gui2/runner.py` + a minimal `core/delivery.py::ensure_masks` change. Gate PASSED on the
SONY_TEST shoe-stand scan; evidence `_infusion/ui3_RECORD.md` (+ `ui3_propagate_stats.json`,
`ui3_delivery_check.json`, `ui3_ui_smoke.json`, `ui3_strips/*.png`, and the re-runnable
`ui3_delivery_check.py` / `ui3_ui_smoke.py`).

- **SAM 2.1 install (first task):** was NOT installed; installed editable with
  `SAM2_BUILD_CUDA=0` + constraints (torch untouched). sm_120 OK (SDPA; `_C` skipped and
  `fill_hole_area=0` so it is never called). Checkpoint `sam2.1_hiera_large.pt` (898 MB)
  was already present. Logged in THIRD_PARTY.md.
- **`gui2/sam_service.py`:** (a) `prompt_frame(image, points, labels)` — interactive
  single-frame segmentation via the SAM 2.1 image predictor (0.33 s cold / 5 ms warm, in
  the app process); (b) `propagate(frames_dir, prompts, out_masks_dir)` — subprocess
  (`python -m gui2.sam_service propagate`, mirrors `preview.py`) that tracks the object
  across the orbit with the SAM 2.1 **video predictor**. Frames downsampled to 1024 long
  edge (draft-decode of the 60 MP Sony **MPO** frames: 588-frame prep **91.7 s**);
  propagation **8.6 GB VRAM**, ~16 fps single-seed (36 s / 588 frames), ~93 s for the
  4-anchor gate run. Multi-anchor merge = **nearest seed wins, per-seed claim radius**
  (a corrective click fixes its local window only) — OURS; naive multi-conditioning
  regresses because SAM 2 propagates sequentially from the min conditioning frame.
- **Step-③ UI:** ~12 representative views (uniform orbit stride) with the current BiRefNet
  matte tinted (red) → **Looks right** (keep matte; delivery uses `--gen_masks`) / **Fix
  it** → click include (green) / exclude (red) with a `gr.Radio` mode toggle (Gradio click
  events carry no mouse button), live SAM re-seg on `gr.Image.select`, edit multiple views,
  **Propagate**, 4 spot-check overlays, **Accept**. Accepted masks → `<run_dir>/masks_reviewed/`.
- **Delivery wiring:** `runner._run_delivery` passes `--mask_dir masks_reviewed` (dropping
  `--gen_masks`) when reviewed masks exist; `assemble_delivery_flags(reviewed_mask_dir=…)`
  and `ensure_masks` (now treats an explicit `--mask_dir` as authoritative, re-staging over
  stale `masks_{factor}`) — both **default-compatible** (unchanged when no reviewed masks).
- **Parallelism:** a "Review masks (while training)" button in Step-② is enabled as soon as
  the SfM export exists (masks only need the frames); accepting during a live run returns to
  Processing and the masks are applied at delivery — zero added wall-clock (masks are a
  delivery input, WP-S2).
- **Gate results (SONY_TEST shoe on a stand):** single include/exclude seed → 91.7% shoe-only
  spot-checks (only the sole-on view collapsed); the design's iterate loop (3 sole-arc
  corrective clicks) → **40/40 (100%)** on an every-15-frame sweep. Reviewed-mask delivery
  extract = 884k Gaussians with **mean out-of-shoe-mask alpha 0.013** vs the BiRefNet baseline
  (with stand) 0.104 (8× less) — the stand is removed from the object SOG (visual +
  render-alpha evidence). UI flow smoke: 12/12 checks.
- **Limitations:** interactive exclude uses a Radio mode toggle, not a real right-click
  (Gradio limitation); per-frame brush touch-up deferred to the funded phase; the sole/underside
  arc of a 360° turntable is the hard region and typically needs 1–3 corrective clicks; the
  parallel-review→delivery hand-off is best-effort (delivery reads `masks_reviewed` at delivery
  time — if the user is still reviewing when training ends, delivery falls back to BiRefNet).

## 6. Open questions for the user (validate before build)

1. **Architecture:** Gradio-6 rebuild now + web-app at funded phase — agreed, or jump straight
   to web now?
2. **Mask editing depth:** are SAM point-prompts (click include/exclude + auto-propagate) enough
   for v1, or is a paint-brush per-frame editor required from day one?
3. **Preview:** is the auto-refreshing turntable strip + on-demand interactive viewer the right
   balance, or do you want the interactive viewer permanently docked (heavier)?
4. Anything to add to the Utilities list?
