# UI-2 — Live training preview system — build + gate record (2026-07-08)

**Status: DONE. Gate PASSED (all 5 criteria, `PASS: true`).** Do NOT commit (per WP).
Design followed: `docs/GUI_REDESIGN.md` §3② (auto-refreshing 6-frame turntable strip +
on-demand interactive viewer — the user-validated design, NOT a docked viewer) + §5.1
(UI-1 hook points). Legacy `gui/` + `run.py` untouched.

Machine-readable evidence + re-runnable scripts (non-gitignored, under `_infusion/`):
- `ui2_gate_driver.py` — the gate driver (salvaged from scratchpad; re-runnable).
- `ui2_gate_trace.json` — full gate trace (config, strips, preview_now, export, stall, gate booleans).
- `ui2_burst_measure.py` + `ui2_burst_3m.log` — re-runnable GPU-contention measurement @ 3M gaussians.
- `ui2_burst.json` — direct GPU-contention (trainer-stall) summary.
- `ui2_viewer_http_probe.json` — HTTP verification of the interactive viewer against the REAL gate model.
- `ui2_strip_step124.png`, `ui2_strip_step499.png` — early vs final gate strips (splat improving).
- `ui2_env_realposes_step499.png` — env-mode strip using REAL capture poses (review fix #2 proof).

---

## 1. What was built — file map

| File | Role |
|---|---|
| `gui2/preview.py` (NEW, ~360 lines) | Subprocess CLI `python -m gui2.preview strip|export`. **strip**: mode-specific poses — OBJECT `build_turntable_poses` (turntable) / ENVIRONMENT `select_real_poses` (real capture poses); reuses `gsplat_trainer.{load_colmap_data, render_splats}` + `coverage.{object_center_from_cameras, estimate_up, _basis}`, composites a 3×2 grid (~720px) with a step label, atomic-writes `strip_<step>.png`. `--mode`/`--test_every` args. **export**: checkpoint→SOG for the viewer (reuses the trainer's `point_cloud_<step>.ply` when present, else `delivery.export_ply`; then `delivery.run_splat_transform` PLY→SOG, SH2 quick profile, reduced iterations). |
| `gui2/runner.py` (EDIT) | Preview state in `__init__`; sets `ckpt_dir`/`ply_dir` at training start; `_train_cb` fires `_maybe_render_preview()` (debounced 1.5s, skips if a render is in flight; passes `--mode`/`--test_every`) after each checkpoint save; `_render_final_strip()` (waits bounded for in-flight, then renders the true final strip unconditionally, clean-completion-scoped); `_eval_last_preview()` sets `preview_note` after 3 consecutive failures; `cancel()` tree-kills any in-flight preview; new public `preview_now()`, `list_checkpoints()`, `latest_strip()`, `export_viewer()`. |
| `gui2/app.py` (EDIT) | Step② panel: preview image (auto-refresh via Timer tick → `_latest_preview` now reads `<run_dir>/preview/strip_*.png`), **Preview now** button, **Open interactive preview** button (→ `_viewer_iframe_html` gr.HTML iframe), **checkpoint-history dropdown** (renders a strip from an older ckpt). `launch()` rewritten to mount the vendored viewer (`/viewer`) + scene files (`/scenes` = `_dataset`) as FastAPI StaticFiles via `gr.mount_gradio_app` + uvicorn (same-origin serving). `run2.py` unchanged (still calls `launch()`). |
| `gui2/viewer/{index.html,index.css,index.js}` (NEW, vendored) | Prebuilt PlayCanvas `@playcanvas/supersplat-viewer` v1.27.1 bundle (MIT). Self-contained, offline, reads `.sog` via `?content=`. Logged in `THIRD_PARTY.md`. |

## 2. How the interactive viewer embed works (and its limits)

**Chosen approach: self-hosted PlayCanvas SuperSplat viewer, same-origin.**
- The wizard, the viewer bundle, and the exported `.sog` are all served from ONE origin
  (`127.0.0.1:<port>`): `gui2/app.py::launch()` builds a FastAPI app, mounts
  `StaticFiles("/viewer" → gui2/viewer)` and `StaticFiles("/scenes" → _dataset)`, then
  `gr.mount_gradio_app(fastapi, demo, path="/", theme=…, css=…)` and `uvicorn.run`.
- The embed is a `gr.HTML` iframe:
  `<iframe src="/viewer/index.html?content=/scenes/<scan>/preview/viewer_<step>.sog">`.
  The viewer's own `?content=` param is fetched **same-origin** → no CORS, no mixed content.

**Why NOT the hosted superspl.at viewer (verified 2026-07-08, researcher primary sources):**
the hosted page at `https://superspl.at` **cannot** load a `http://127.0.0.1` file. Mixed
content is NOT the blocker (browsers exempt loopback `127.0.0.1`/`localhost`). **CORS IS**: a
fetch from the `superspl.at` origin to localhost is cross-origin, and Gradio's
`CustomCORSMiddleware` deliberately withholds `Access-Control-Allow-Origin` for external
origins hitting a local server (by design, to block remote-page-reads-local-file). So the
hosted iframe is a dead end for localhost — self-hosting same-origin is the fix. Options
considered: (1) self-hosted supersplat-viewer — **chosen** (reads `.sog`, PlayCanvas-official,
the design's named viewer, single prebuilt bundle, offline); (2) antimatter15/splat — simpler
(2 files) but `.ply`-only, no `.sog`; (3) `gr.Model3D` native gsplat — no iframe but `.ply`-only
and drops the SuperSplat design intent. Ranked #1 chosen; #2/#3 documented as fallbacks.

**Limits (honest):** (a) The `.sog` MIME served by StaticFiles is `text/plain` — harmless (the
viewer fetches an ArrayBuffer), and range requests work (206 verified). (b) On-demand SOG export
of a **full 30k / 3M-gaussian** model takes ~30–65s (splat-transform SOG compression); we use a
reduced iteration count (`iterations=8`) for the preview to keep it quick — the gate's tiny model
exports in a couple seconds. (c) The actual WebGL render (does the splat visibly appear + orbit)
cannot be automated headless; verified up to the HTTP boundary + iframe HTML (below).

**HOW TO SEE IT (human click-through):** launch `python run2.py`, run a scan; in Step②, once a
checkpoint exists, click **Open interactive preview** → the iframe loads
`/viewer/index.html?content=/scenes/<scan>/preview/viewer_<step>.sog`. Drag to orbit, scroll to
zoom, right-drag to pan. Verified reachable URL for the gate model (server on :7899):
`http://127.0.0.1:7899/viewer/index.html?content=/scenes/UI2_SMOKE/preview/viewer_499.sog`.

## 3. Gate — mini smoke (UI-1 config + save_all quarterly ckpts)

Driver: `scratchpad/ui2_gate.py` (guarded under `__main__` — avoids the UI-1 hloc
DataLoader-worker fork-bomb). Config: **environment** mode, input = the existing 39-frame
every-8th Laptop subset (`_dataset/UI1_SMOKE/input_images`), run_dir `_dataset/UI2_SMOKE`,
`max_steps=500`, `data_factor=4`, `cap_max=100000`, `seed=0`, `delivery_no_qa=True`,
`save_all=True` → ckpts at 125/250/375/500 (files `ckpt_{124,249,374,499}`). Ran on the free
GPU (0% util at start; no WP-T2 job active). Full chain **done in 211.5s**.

Values below are from the **re-run after the review fixes** (`ui2_gate_trace.json`, elapsed 223.4s).

| Gate criterion | Result | Evidence |
|---|---|---|
| **Strips auto-appear at each checkpoint (≥3)** | **PASS** — 3 strips at steps **124, 249, 499** (`spawn_events` = 124 auto, 249 Preview-now, 499 via `_render_final_strip` blocking) | `ui2_gate_trace.json.{strip_steps,spawn_events}`; `ui2_strip_step{124,499}.png` |
| **"Preview now" works mid-run** | **PASS** — rendered from `ckpt_249`, file written (`"Preview rendered from step 249."`, `exists: true`) | `ui2_gate_trace.json.preview_now` |
| **Interactive viewer loads final model at a browser-reachable URL** | **PASS** — the real `viewer_499.sog` (1.46 MB) exported and served: wizard/viewer-html/viewer-js = 200, SOG full = 200, SOG range = **206 Partial Content**, viewer+`?content=` = 200 | `ui2_viewer_http_probe.json` |
| **No GPU contention / trainer stall <5s** | **PASS** — see §4 | `ui2_burst_3m.log`, `ui2_burst.json`, `ui2_gate_trace.json.trainer_stall` |
| **Run completes** | **PASS** — `status: done`, 2 deliverables | `ui2_gate_trace.json` |

`gate: {run_done, strips_ge_3, preview_now_ok, viewer_export_ok, trainer_stall_under_5s, PASS} = all true`.
(First-run trace pre-fix: strips [124, 374, 499] — same PASS; the review fixes did not regress it.)

## 4. Trainer-stall measurement (honest)

**Direct GPU-contention measurement — REPRODUCIBLE (review fix #3):** the preview subprocess only
shares the GPU for the render burst; the rest of its wall time is CPU-side (torch import + gsplat
import + COLMAP parse). `gpu_render_burst` is measured with `torch.cuda.synchronize` around the
render loop only. Persisted script + log so the headline is re-runnable from committed artifacts:
- Script: `_infusion/ui2_burst_measure.py` (defaults to the WP-X1 cap3m 3M ckpt).
- Log: `_infusion/ui2_burst_3m.log` — **3,000,000 gaussians → gpu_render_burst = 0.24 s (cold),
  0.06 s (warm)**.
- Gate model (54,435 gaussians): 0.18 s GPU burst (`ui2_burst.json`).

So even a maximal 3M-Gaussian render occupies the GPU for **~0.24 s** — the trainer's worst-case
contention window. **Far under the 5 s bar.**

**Corroborating (gate run, `trainer_stall`):** the trainer's progress callback logs every 100
steps (`n_steps_logged: 5`), so the recorded gaps are 100-step *windows*, not per-step. Every
window was **~1.8–2.7 s** with NO anomaly during the concurrent preview renders → the max window
(2.685 s re-run / 2.607 s first run) bounds the total stall from above (a multi-second freeze would
inflate its window). Combined with the 0.24 s direct burst, the trainer stall from preview
contention is **sub-second**. Design choice: the strip renderer "accepts brief overlap" (per the WP
option) — no GPU lock needed because the burst is negligible and VRAM is ample (96 GB).

## 5. Flags / levers

No new TRAINER flag. UI-2 adds gui2-only preview plumbing; every non-gui2 caller is byte-identical
(the shared `save_all` / quarterly-ckpt behavior was already added by UI-1's closeout — UI-2's
hooks ride those existing saves). The strip renderer and SOG export are OURS; the only new
third-party component is the vendored viewer bundle (`THIRD_PARTY.md` → UI-2 section). Default app
behavior unchanged until a scan runs.

## 6. Deviations / notes (honest)

- **"~720px height" → 3×2 grid.** A single 6-wide row at 720px would display as an unreadable
  sliver in the side panel, so frames (~360px each) are laid in a 3-col × 2-row grid → strip ≈720px
  tall AND legible. `--cols`/`--height` are CLI flags. The turntable still reads left→right,
  top→bottom as the object rotates.
- **Strip poses are mode-specific (review fix #2).** OBJECT mode = synthesized turntable
  (`build_turntable_poses`: `object_center_from_cameras` + orbit basis, SfM-point-median fallback)
  — verified on the RTX card object scan (`scratchpad/test_strip.png`, subject centered while
  azimuth sweeps). ENVIRONMENT mode = **6 REAL training-camera poses** spread across the capture
  (`select_real_poses`) — a synthesized orbit around a divergent/inside-out scene can point into
  unreconstructed void, whereas real capture poses always look at reconstructed geometry. Proven on
  the env-mode gate scene: `_infusion/ui2_env_realposes_step499.png` (labeled "6 capture views",
  all frames on real geometry). `mode`/`test_every` thread from `RunConfig` → `_spawn_preview` →
  `python -m gui2.preview strip --mode … --test_every …`.
- **`/scenes` static mount exposes all of `_dataset` (review fix #5b, ACCEPTED as scope).** The
  server binds `127.0.0.1` only (localhost POC), so whole-`_dataset` read exposure is acceptable;
  a funded/multi-user rebuild would scope the mount to the active run dir.
- **`launch()` rewritten** to `gr.mount_gradio_app` + uvicorn (needed for the same-origin
  `/viewer` + `/scenes` static mounts; theme/css moved to `mount_gradio_app`, per Gradio 6). The
  runner-driven path (used by UI-1 smoke + this gate) is untouched. 146 pytest tests still collect.
- **WebGL render not headless-verifiable** — verified to the HTTP + iframe-HTML boundary; the
  actual orbit is a human click (URL in §2).
- **Preview subprocess robustness:** tolerates a checkpoint still being written (retry-on-load),
  writes strips atomically, and never breaks training on a preview error (wrapped in try/except).

---

## 7. Review closeout (2026-07-09) — all 5 items addressed; gate re-run PASS

Reviewer verdict: gate stood; provenance/security/conventions verified clean; 5 nits. All fixed;
**gate re-run with the fixed code → PASS (5/5)** (`ui2_gate_trace.json`, strips [124, 249, 499],
final strip via `_render_final_strip` blocking, done in 223.4s). 146 pytest tests still collect.

- **MED #1 — "guaranteed final strip" made real.** (a) Cancel scoping DOCUMENTED (record §6,
  GUI_REDESIGN §5.2): the final strip sits after the training `_check_cancel`, so a cancelled run
  correctly has no "final" strip — by design. (b) FIXED: replaced the guarded `_maybe_render_preview`
  final trigger with `runner._render_final_strip()` — it **waits (bounded 15s) for any in-flight
  per-step render, then renders the true final strip UNCONDITIONALLY** (bypasses the step +
  in-flight guards). Re-run proof: `spawn_events` shows step 499 spawned `blocking: true` and
  `strip_499.png` written.
- **MED #2 — environment-mode turntable safety.** FIXED: env mode no longer synthesizes an orbit
  (which can stare into void for divergent captures) — `select_real_poses` picks 6 REAL training
  poses spread across the capture; object mode keeps the synthesized turntable. `mode` threads
  `RunConfig → _spawn_preview → preview.py`. Proof: `ui2_env_realposes_step499.png` ("6 capture
  views", all real geometry).
- **MED #3 — 0.22s@3M made reproducible.** FIXED: `_infusion/ui2_burst_measure.py` (committed) +
  re-run against the WP-X1 cap3m 3M ckpt → `_infusion/ui2_burst_3m.log`: **gpu_render_burst = 0.24s
  cold / 0.06s warm @ 3,000,000 gaussians**.
- **#4 — gate driver salvaged.** DONE FIRST: `scratchpad/ui2_gate.py` → `_infusion/ui2_gate_driver.py`
  (re-runnable; used for the re-run above).
- **LOW #5.** (a) Systematic-failure surfacing: `runner._eval_last_preview()` bumps a streak on a
  failed/absent-output preview subprocess and sets `preview_note` after 3 consecutive failures; the
  app tick pushes "⚠ Live preview unavailable — see preview/render.log" into Step②'s status (only
  when a note exists, so button messages aren't clobbered). (b) `/scenes` whole-`_dataset` exposure
  accepted as localhost-POC scope (record §6). (c) `preview.py` hardcoded `test_every=8` → now reads
  the run's actual value via `--test_every` (runner passes `cfg.settings["test_every"]`).
