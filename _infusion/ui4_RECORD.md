# UI-4 — GUI 2.0 Deliver screen + Utilities menu — build + gate record (2026-07-09)

**Status: DONE. Gate PASSED 16/16 (`PASS=True`) — re-run after the review closeout (§8).**
Do NOT commit (per WP instructions). Design followed: `docs/GUI_REDESIGN.md` §3④ + §4 + §5
(new §5.4). GUI 2.0 wizard is now feature-complete (UI-1..4). Legacy `gui/` + `run.py`
untouched. **The persisted `ui4_gate_trace.json` is the source of truth for numbers** —
the mini run is mildly nondeterministic (registration/training), so headline figures below
are from the latest (post-closeout) re-run.

Machine-readable evidence (non-gitignored `_infusion/`):
- `ui4_gate_trace.json` — full gate trace (16 checks + every step's final-state numbers).
- `ui4_gate_driver.py` — re-runnable driver (`python _infusion/ui4_gate_driver.py`).
- `ui4_gate_run.log` — full stdout of the object-mode mini run + gate.
- `ui4_quality_card.html` / `ui4_deliverable_cards.html` — the actual rendered Step-④ HTML.
- `ui4_coverage_map.png` — the re-capture coverage map produced by the improve-further action.
- `ui4_util_qc_hist.png` — the Utilities → Capture-QC standalone sharpness histogram.
Run artifacts (gitignored `_dataset/UI4_SMOKE/`): `deliverables/UI4_SMOKE_object.{sog,ply,
_delivery.json}`, `improve/coverage_map.png`, `run_state.json`.

---

## 1. File map

| File | Role |
|---|---|
| `gui2/deliver.py` (NEW, ~430 lines) | All Deliver-screen logic, **Gradio-free** (unit-testable / gateable without a browser): star-scale mapping (`lpips_to_stars`/`star_string`/`star_label`), `quality_card()` + `quality_card_html()`, `deliverable_cards()` + `deliverable_cards_html()` + `scene_url_for()`, `coverage_report_for()` (re-capture), `cleanup_master()` (floater prune), `resolve_data_dir()`. |
| `gui2/app.py` (edited) | Step-④ rewritten (quality card + deliverable cards + preview/locate dropdown + View-in-3D iframe + Improve-further accordion + Refresh); a top **Utilities** accordion (5 tabs); `_render_deliver` chained after every Step-4 transition; handlers for view/show-folder/detail-rerun/recapture/floater-cleanup + the 5 utilities; module-level `_spawn_detached`. `_deliverables_html` now delegates to `deliver.deliverable_cards_html`. |
| `gui2/theme.css` (edited) | Quality-card grid (responsive 4→2→1), star/flags styling, utilities spacing. |
| `_infusion/ui4_gate_driver.py` (NEW) | The validation gate. |

## 2. Star-scale mapping (chosen — single source `deliver.lpips_to_stars`)

Object-region (or scene-masked) LPIPS → **5-star**, from the banked range in the WP:

    stars = clamp( 5 − 4·(LPIPS − 0.18)/(0.35 − 0.18), 1, 5 )

so **LPIPS ≤ 0.18 → 5.0★ "Excellent"**, **0.235 → 3.7★ "Very good"** (Sony shoe reference),
**0.28 → 2.65★ "Good"**, **≥ 0.35 → 1.0★ "Poor"**. Labels bucket the star value
(≥4.5 Excellent / ≥3.5 Very good / ≥2.5 Good / ≥1.5 Fair / else Poor). The **raw LPIPS is
always shown underneath** the stars for experts. Env mode relabels to "Scene sharpness"
and reads `qa.delivered.masked_lpips`.

The other three friendly numbers:
- **Geometric consistency** = TSED consistent-pair ratio **if** a QA scorecard has the
  phase-2 TSED scorer, **else the record's spatial-compactness %**
  (`qa.spatial_floaters.within_0.5_scale_pct`). Default delivery does not run TSED, so the
  documented **fallback fires** (gate: 100.0% compactness).
- **Floaters** = object bg-bleed alpha (`mean_bg_bleed_alpha`) → Clean (≤3%) / Minor (≤8%) /
  Noticeable; env = SOR-removed %.
- **Views used** = registered / total (COLMAP `num_reg_images` vs input frame count).
- **Honest flags** derive ONLY from fields present in the record (bleed >5%, >50 beyond-core
  Gaussians, unregistered frames). "underside weakly observed" needs per-view PSNR spread
  that is **not banked in the record → omitted, not faked** (per the WP's "else omit").

## 3. Gate — full object-mode wizard on the mini config (PASS 16/16)

**Config:** `_dataset/UI1_SMOKE/input_images` (39-frame Laptop subset, 3840×2160, 8.29 MP),
**mode=object**, `data_factor=4`, `max_steps=500`, `cap_max=100k`, seed 0,
**`extract_object` ON** (`--extract_object --gen_masks --decimate 60% --filter_harmonics 3`),
`delivery_refit=False` (skips the 3000-step refit → **GPU stays bursty**), QA **ON**.
Full chain done in **241 s**; drives `runner.PipelineRunner` directly (same API the wizard
wires), then exercises every UI-4 surface via the pure `gui2.deliver` / `gui2.qc` functions
+ an HTTP probe of the real `/scenes` + `/viewer` static mounts.

| # | Gate criterion | Result |
|---|---|---|
| 1 | run completes | `status=done`, 241 s |
| 2 | object flow routes through Step③ | `next_step_after_processing('object')==3` |
| 3 | Step③ became reachable during run | export-ready at global-frac 0.60 |
| 4 | quality card **has record** | true |
| 5 | quality card sharpness **real** | **obj-LPIPS 0.231 → 3.8★ "Very good"** |
| 6 | quality card geometry **real** | **compactness 100.0%** (TSED-fallback) |
| 7 | quality card views **real** (STRONG: total==39) | **38/39 registered** (fix HIGH-1) |
| 8 | quality card floaters **real** | **10.2% bleed → "Noticeable bleed"** (flag trips) |
| 9 | deliverable object card present | `UI4_SMOKE_object` |
| 10 | deliverable card meta | **21,980 Gaussians · SH 3 · .sog 1.01 MB · .ply 5.35 MB** |
| 11 | viewer bundle HTTP 200 | `/viewer/index.html` → 200 (49,250 B) |
| 12 | **object SOG HTTP 200** | `/scenes/UI4_SMOKE/deliverables/UI4_SMOKE_object.sog` → 200 (1,014,881 B) |
| 13 | iframe URL well-formed | `/viewer/index.html?content=/scenes/…UI4_SMOKE_object.sog` |
| 14 | **re-capture coverage map produced** | **82.5%** hemisphere coverage, PNG written |
| 15 | Utilities Capture-QC standalone | **39 frames, 3840×2160**, histogram PNG written |
| 16 | object extraction actually ran | vote kept **37,561 of 54,374** master Gaussians |

`16/16 checks passed; PASS=True` (persisted `ui4_gate_trace.json`); full run 229 s.

## 4. Each step's final state (precise descriptions; no headless browser — same as UI-1..3)

- **Step① New scan** — unchanged from UI-1; mode cards, input, subject, Capture-QC card,
  Advanced drawer (validated defaults), one primary CTA (**Start**). Utilities accordion
  now sits under the header, collapsed.
- **Step② Processing** — unchanged from UI-2/3 (unified progress, live turntable strip,
  interactive preview, checkpoint history, Review-masks parallel button).
- **Step③ Review** — reached in the object flow (gate #2/#3); UI-3 screen (matte gallery →
  Looks-right / Fix-it SAM point-prompts → propagate → accept). Auto-skipped for Environment.
- **Step④ Deliver** — **quality card** (4 metric tiles: Object sharpness ★★★★☆ "Very good" /
  region-LPIPS 0.221; Geometric consistency 100.0% compactness; Floaters "Noticeable bleed"
  10.0%; Views used 38/38) + a **Notes** flag ("Some background bleed remains at the object
  edges (10.0% out-of-mask alpha)."). Below: a **deliverable card** (🧸 Object · 22,660
  Gaussians · SH 3 · .sog 1.02 MB · .ply 5.35 MB), a **Preview/locate** dropdown +
  **View in 3D** (loads the SuperSplat iframe on the object SOG) + **Show in folder**, an
  **Improve further** accordion (Detail re-run · Re-capture report → coverage map ·
  Floater cleanup), and Refresh / Open-folder / New-scan. Rendered HTML persisted at
  `_infusion/ui4_quality_card.html` + `ui4_deliverable_cards.html`.

## 5. Flags / levers documented

UI-4 adds **no new trainer flag** and **no new external component** (THIRD_PARTY.md UI-4
entry: reuses `core/delivery.py`, `core/enhancement/coverage.py`, `core/training/prune.py`,
`qa_report.py`, `gui2/preview.py`, the vendored SuperSplat viewer, BiRefNet — all logged).
The improve-further actions are user-initiated and land in **variant run dirs**
(`<scan>_detail`, `<scan>_cleaned`) so they never clobber the base run. Production behavior
of every non-gui2 module is byte-unchanged.

## 6. Tests

`python -m pytest tests/ -q` → **151 passed** (UI-4 touches only `gui2/` + `docs` +
`THIRD_PARTY.md`; no `tests/` imports `gui2`, so the suite is unaffected).

## 7. GUI 2.0 status vs the design doc (UI-1..4 checklist)

**Shipped (UI-1..4):**
- ✅ Wizard shell, two modes, Capture-QC card, Advanced drawer w/ validated defaults,
  headless runner + unified weighted progress, cancel + resume-from-stage (UI-1).
- ✅ Live turntable-strip preview + on-demand interactive SuperSplat viewer, checkpoint
  history, same-origin `/scenes` + `/viewer` mounts (UI-2).
- ✅ Mask Review: representative views, SAM 2.1 point-prompts + video propagation,
  delivery re-carve hook, parallel-with-training (UI-3).
- ✅ Deliver screen: quality card (4 friendly numbers + honest flags), deliverable cards +
  View-in-3D + Show-in-folder, Improve-further (detail re-run / re-capture / floater
  cleanup), Utilities menu (Capture QC / Coverage / QA scorecard / Compare / Open folder),
  polish pass (UI-4).

**Deferred to the funded rebuild (honest):**
- Dual **object + full SOG in ONE run** (design §2 delivery): the runner still produces one
  deliverable set per run; UI-4 renders whatever files exist and does not add the 2nd delivery.
- **SfM reuse** across detail/cleanup variants (they re-run SfM into the variant dir).
- **Pause** during training (Windows has no reliable trainer SIGSTOP — Cancel+resume ships).
- Real **HTML embed** for the QA `report.html` (shown as a path) + browser screenshots
  (no headless browser installed; rendered HTML + PNGs are the persisted visual evidence).
- Per-frame **brush** mask touch-up (design §3 note; SAM point-prompts cover the real cases).

## 8. Review closeout (2026-07-09) — 2 HIGH + 2 MED + minors, all fixed; gate re-run PASS

Reviewer flagged 2 ship-blocking HIGHs + coordination gaps. All addressed; gate re-run
**16/16 PASS** (validates the refactored runner delivery path end-to-end); `tests/` **151
passed**; argv dry-run **PASS** (`ui4_floater_cleanup_argv_proof.{py,json}`).

- **HIGH-1 — "Views used" total was confidently-wrong (100%) on folder-drop.** `input_images`
  lives under the run dir only for the video-extract flow; the external-folder-drop flow
  (the gate itself: input = `UI1_SMOKE/input_images`, run dir = `UI4_SMOKE`) had none, so
  the old fallback set `total = registered` → **38/38**. FIXED (`deliver._registered_total`):
  primary total = `export/images/` count (present in BOTH flows, holds every input not just
  the registered subset), then `input_images/`, then `pipeline_state.json` image_list;
  **`total` is NEVER defaulted to registered** (None if truly unreadable). Gate strengthened
  to assert **`total == 39`** (real-number, not not-None). Now honestly **38/39**.
- **HIGH-2 — floater cleanup bypassed the reviewed-mask plumbing.** It emitted
  `--gen_masks` unconditionally (post-UI-3 scans silently re-extracted from the UNREVIEWED
  BiRefNet matte — the stand returns) and let refit default back ON. FIXED by extracting
  the runner's reviewed-mask+marker resolution and refit/no_qa assembly into ONE shared
  helper — `deliver.build_delivery_flags()` (+ `deliver.resolve_reviewed_mask_dir()`) —
  that BOTH `runner._run_delivery` and `app._floater_cleanup` now call (DRY, rules.txt).
  Proof (`ui4_floater_cleanup_argv_proof.json`, dry — no execution): against SONY_TEST's
  accepted `masks_reviewed` (588 PNGs + `.complete=588`), cleanup argv carries
  `--mask_dir …/masks_reviewed` (NOT `--gen_masks`) + `--extract_object`; refit flag matches
  the base cfg (**None → neither flag · False → `--no-refit` · True → `--refit`**), `--no_qa`
  passes through; a no-reviewed control correctly falls back to `--gen_masks`.
- **MED-3 — detached spawns were invisible / no collision guard.** Added a module job
  registry `_JOBS{name: Popen}`: `_spawn_detached` registers by name; `_floater_cleanup`
  refuses a duplicate same-name spawn while alive; **Refresh** surfaces a
  running/done/failed line (`_jobs_status_line` via `poll()`); the Start-scan guard warns
  (does not hard-block) when a background re-delivery is live.
- **MED-4 — expert utilities could silently contend for the GPU.** `_util_qa` /
  `_util_compare` now prefix a visible "⚠ A scan is training — this shares the GPU" line
  when `_RUNNER.is_alive()`, then proceed (simplest robust; Gradio confirm dialogs are awkward).
- **MINORS — (5a)** env-mode floaters label is **"Not scored"** when the record has no
  `removed_outliers` (was "Cleaned"); **(5b)** `deliver._pretty_variant` dead double-loop
  simplified to a single tokenized variant lookup.

### 8.1 Gate coverage — automated vs wired-but-manually-verified (honest)

**Automated in the gate driver (executed + asserted):** the full **object-mode** wizard
flow (SfM → training → object-extraction delivery → QA record), the quality card's four
numbers on that real object record, deliverable cards + scene URL, View-in-3D over the live
`/scenes` + `/viewer` HTTP mounts, the **re-capture / coverage** improve-further action, and
**1 of 5** Utilities (**Capture QC** standalone). The runner's refactored `build_delivery_flags`
path is exercised end-to-end by the run itself.

**Wired + verified by unit/dry-run, NOT by a full gate execution:** the **env-mode** quality
card (verified against banked WP-D1 full-scene records + the Sony object record, not a fresh
env run); **Detail-mode re-run** (variant-run-dir construction reviewed, not launched);
**Floater cleanup** (prune helper + the shared re-delivery argv proven by
`ui4_floater_cleanup_argv_proof.py`, delivery not executed to conserve GPU); the remaining
**4 Utilities** (Coverage/QA-scorecard/Compare/Open-folder — each wraps an existing,
separately-gated CLI). These are documented as manual-verify so a reviewer knows exactly
what the 16/16 does and does not cover.

**Known metric gap (honest):** **Geometric consistency is always "not scored" for
environment / full-scene deliveries** — the fallback source (`qa.spatial_floaters`) is
written only in object mode, and TSED is a phase-2 scorer not run in the default delivery.
Env-mode geometry therefore shows "not scored" until a TSED-enabled scorecard exists or a
full-scene compactness proxy is added (deferred to the funded phase).
