# HYPERclone GUI 2.0 — Design Plan (July 2026, for validation)

**Goal:** replace the 10-tab expert GUI with a modern, simple, wizard-style app where the
validated-optimal settings are the defaults, an untrained user gets a great splat with zero
knob-turning, and an expert can still reach every lever. Two product modes: **Object Scan** and
**Environment Scan**. Status: VALIDATED by user 2026-07-07 — Gradio-6 rebuild / SAM point-prompts / auto-strip preview all approved as recommended. BUILD IN PROGRESS (UI-1).

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
| **UI-1** | `gui2/` skeleton: wizard shell, mode cards, QC card, Advanced drawer w/ validated defaults, pipeline runner (reuses stage handlers headless) + unified progress plumbing | 2–3 d |
| **UI-2** | Training preview system: checkpoint→turntable strip renderer (subprocess, cheap), auto-refresh, pause/resume/cancel; static route + SuperSplat iframe embed | 2 d |
| **UI-3** | Mask Review: representative-view picker, matte overlay gallery, SAM 2.1 point-prompt service (needs SAM2 install verify), video-propagation, re-carve hook | 3–4 d |
| **UI-4** | Deliver screen: quality card mapping, deliverable cards + viewer, "improve further" actions; Utilities menu | 2 d |

Total ≈ 2 weeks of agent time; UI-1 alone is already usable end-to-end. Old GUI stays until
parity, then `run.py` flips to `gui2`.

## 6. Open questions for the user (validate before build)

1. **Architecture:** Gradio-6 rebuild now + web-app at funded phase — agreed, or jump straight
   to web now?
2. **Mask editing depth:** are SAM point-prompts (click include/exclude + auto-propagate) enough
   for v1, or is a paint-brush per-frame editor required from day one?
3. **Preview:** is the auto-refreshing turntable strip + on-demand interactive viewer the right
   balance, or do you want the interactive viewer permanently docked (heavier)?
4. Anything to add to the Utilities list?
