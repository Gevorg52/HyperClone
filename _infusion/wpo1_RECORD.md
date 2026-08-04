# WP-O1 — 360° ingest module — GATE RECORD (2026-08-04)

Plan of record: `docs/PLAN_360.md` §5 (WP-O1). Stub: `docs/EXECUTION_PLAN.md` §WP-O.
Deliverable: `core/ingest_360.py` (+ CLI `python -m core.ingest_360`, tests
`tests/test_ingest_360.py`, evidence collector `_infusion/wpo1_evidence.py`).

**GATE: PASSED** on all four checks. Evidence in `_infusion/wpo1/` (committed);
heavy outputs live in the gitignored `_dataset/360/*` and are reproducible from the
argv recorded in every `report_*.json`.

---

## 1. What was built

`core/ingest_360.py` — one 360° video file → the plain image folder the pipeline already
understands. **Default-OFF**: `main()` returns 2 and does nothing unless
`--capture-type insta360` is passed. Nothing in the production path imports the module,
so production behaviour is byte-for-byte unchanged with the flag off.

```
<out_dir>/images/pano_00000.jpg    q95 equirectangular
<out_dir>/masks/pano_00000.png     uint8, 255 = VALID, 0 = EXCLUDED  (repo convention,
                                   gsplat_trainer.py:421-428 / gsplat_backend.py:71)
<out_dir>/ingest_report.json       probe + argv + stitch params + per-frame sharpness/selection
<out_dir>/qa/{contact_sheet.jpg,sharpness.png}
```

| stage | how |
|---|---|
| probe | `ffprobe -show_streams`: 2 equal SQUARE video streams → `dual_fisheye`; 1 stream at 2:1 → `erp` (stitch skipped); anything else → hard `ValueError` naming the sizes. `--force_layout` overrides. |
| sample | **window-relative max sharpness only.** window = `round(fps / --target_fps)` (default 2.0 pano-fps → 15 frames at 29.97 fps); sharpest frame per window by var-of-Laplacian on the central crop, decoded through ONE reduced-size raw-gray ffmpeg pipe. Window selection reuses `core.frame_select.select_indices`. **No absolute threshold anywhere** (DEVELOPER_NOTES §5); `ingest_report.json` carries `"absolute_threshold": null` and the full per-source-frame sharpness array. |
| stitch | one ffmpeg pass, filter script file (no argv-length limit): `[0:0][0:1]hstack → select(n∈selected) → v360=dfisheye:e:ih_fov=200:iv_fov=200:w:h`. **`select` sits BEFORE `v360`** so the expensive remap only runs on the ~6% of frames we keep. Output defaults to 2×input-height × input-height ⇒ resolution-agnostic. Frames come back over a rawvideo pipe and are JPEG-encoded by cv2 at q95 (exact quality control, and the SAM downscale is produced in the same pass). |
| mask | one uint8 PNG per frame = nadir band ∪ seam bands ∪ operator. Nadir: bottom `--nadir_deg` (12°). Seams: ±`--seam_deg` (4°) around yaw ±90° (x = W/4, 3W/4) — auto-0 for pre-stitched ERP input. Operator: SAM 2.1 **video propagation** over the selected frames, clamped to the rear hemisphere, dilated `--operator_dilate_px` (12), with area/lost guards falling back to a fixed core box. |

### Verified geometry (measured on this footage, 2026-08-04)
- ffprobe stream **0 = scene-facing lens**, stream 1 = operator-facing lens
  (`_infusion/wpo1` note; checked by extracting both lenses at t=60 s). The sharpness scan
  therefore scores stream 0 by default (`--sharpness_stream`).
- With `[0:0][0:1]hstack → v360=dfisheye:e`, ffmpeg puts the **operator-facing lens at ERP
  CENTRE** (yaw 0) and the scene lens at the ±180° edges. Confirmed at t = 2/20/60 s.
  Seams therefore land exactly at x = W/4 and 3W/4 — visible in every contact sheet.
- ERP conventions used throughout: `yaw(x) = (x/W − 0.5)·360`, `pitch(y) = (0.5 − y/H)·180`.

---

## 2. Gate results

### 2a. Primary asset — `VID_20260721_133413_00_077.insv` (2×1920², 75.2 s, 46 Mbps)

```
python -m core.ingest_360 --capture-type insta360 \
  --input D:/HYPERCLONE/FOOTAGE/FlowerShop/VID_20260721_133413_00_077.insv \
  --out_dir _dataset/360/flowershop_077 --overwrite --contact_sheet 20 --seed 0 --progress 5000
```
(all other values = defaults; full argv also stored in `report_077.json`)

| gate item | required | measured |
|---|---|---|
| selected ERP frames | ≥ 150 | **151** at 3840×1920, effective 1.998 pano-fps |
| sharpness histogram sane | report median/p10 | all frames: min 244 · **p10 454** · **median 1691** · p90 2339 · max 3160 · selected: p10 548 · median 1886 (**+11.6%** vs all-frame median) |
| operator covered on 20 RANDOM frames | yes | **20/20**, all from SAM (`operator_sources: {"sam": 20}`), 0 area-guard, 0 lost-guard. Visual: `_infusion/wpo1/contact_077.jpg` (seed 0 → picks 2,5,10,24,36,41,68,72,81,83,84,87,92,93,107,112,114,130,141,150) and the rear-hemisphere zoom `_infusion/wpo1/zoom_operator_077.jpg` |
| seam/nadir bands present | yes | geometric excluded fraction **0.10847** (nadir 12° = 6.67% + two 8°-wide seam bands = 4.44%, minus overlap); mean total excluded **0.15461**; operator-only fraction median 0.055 |

Wall clock **46.8 s** total (scan 13.7 · stitch 14.7 · mask 16.7) on the RTX PRO 6000.
Output 373 MB / 151 frames.

### 2b. Resolution-agnostic — `VID_20260721_123244_00_027.insv` (2×3840² 8K, 46.8 s, 128 Mbps)

Same code, **zero changes, zero flags changed** except the paths and `--contact_sheet 6`.

- layout auto-detected `dual_fisheye` 3840×3840 → **7680×3840 ERP**, **94 frames** @ 1.998 pano-fps
- sharpness: min 185 · p10 310 · median 447 · p90 729 · max 984; selected median 536 (+19.9%)
- masks: geometric excluded 0.10799, mean excluded 0.15612, operator = SAM on all frames,
  0 guards fired — the flowershop-tuned prompt points generalised to a different room
- wall clock 105 s (scan 24.7 · stitch 58.0 · mask 20.6); 634 MB / 94 frames
- evidence: 3 sample frames as `sample8k_{0,1,2}_overview_7680x3840.jpg` **plus
  `sample8k_{0,1,2}_crop1to1.jpg`** (1024×768 crops at NATIVE 8K resolution — the overview
  downscale would otherwise hide the point of the test), `contact_027_8k.jpg`,
  `report_027_8k.json`

### 2c. Compatibility — `FLOWERSHOP.mov` (in-camera .mov variant)

Auto-detected `dual_fisheye` 1920×1920 from the .mov container without any hint →
**184 frames** @ 1.998 pano-fps, 3840×1920. Sharpness min 24 · p10 82 · median 377 ·
p90 854 (selected median 431, +14.2%). Operator SAM-masked on all 6 contact-sheet frames,
0 guards. 48.5 s. Evidence: `contact_mov.jpg`, `report_mov.json`.

> Cross-check: PLAN_360 §1 measured this same file at 0.5 s cadence as min 36 / p10 77 /
> median 357 / p90 831. Our numbers above (min 24 / p10 82 / median 377 / p90 854) are the
> var-of-Laplacian over **every frame of the full native-fps decode**, not over a 2 pano-fps
> subsample — the 2 pano-fps figure is the *selection* rate, not the scan rate — so the two
> populations differ slightly by construction; the metric still reproduces independently.

### 2d. Pre-stitched-ERP branch (not in the brief's gate list, checked anyway)

No pre-stitched asset exists on this machine, so one was synthesised:
```
ffmpeg -y -t 6 -i VID_..._077.insv -filter_complex \
  "[0:0][0:1]hstack=inputs=2[h];[h]v360=dfisheye:e:ih_fov=200:iv_fov=200:w=1920:h=960[o]" \
  -map "[o]" -c:v libx264 -crf 20 -pix_fmt yuv420p erp_clip.mp4
```
→ auto-detected `erp`, stitch skipped, **seam bands auto-disabled** (`seam_deg 0.0`) and
operator masking auto-off (`operator_mode "none"`) because neither the seam yaw nor the
operator position is knowable for a foreign ERP source. 12 frames, mean excluded 0.0667
(= nadir only). Evidence: `report_erp_path.json`, `contact_erp_path.jpg`.

### 2e. Output integrity (programmatic, `_infusion/wpo1/integrity.json`)

Across all four runs: `n_images == n_masks`, stems pair 1:1 (`pano_NNNNN.jpg` ↔
`pano_NNNNN.png`), every mask `uint8` with pixel value set exactly `{0, 255}`, 0 mask/image
shape mismatches.

### 2f. Tests

`tests/test_ingest_360.py` — 22 new tests (default-OFF gate, clean error on a bad input with
no out_dir debris, window sizing, an explicit "a globally dark stretch still contributes
frames" regression for the DEVELOPER_NOTES §5 trap, band geometry in degrees, operator clamp,
filter-graph ordering, ERP-size derivation, and a real ffmpeg round-trip incl. full ERP
ingest). **173 passed** for `pytest tests/` (151 pre-existing + 22 new); nothing broken.

---

## 3. Negative findings / deviations — recorded honestly

1. **PLAN_360 §7 mislabels the pilot-primary asset.** `VID_20260721_133413_00_077.insv`
   (creation 2026-07-21T20:34:13Z) is **NOT the flower shop** — it is the corridor +
   charcoal-drawing rooms. `VID_..._125954_00_049.insv` is the post-it-note room.
   The flower-shop scene appears only in `FLOWERSHOP.mov` (creation 20:39:00Z ⇒ the `_082`
   slot, matching `LRV_..._133900_01_082.lrv`) and `FLOWERSHOP2.mov`. Consequence:
   **the flower-shop room exists ONLY in the low-res in-camera .mov proxy** — the highest-
   quality asset of that specific room is the 377-median-sharpness file, not the
   1691-median `_077`. WP-O2/O3 should pick the asset by CONTENT, not by the §7 label.
2. **Prompt points are pose-tuned, not learned.** The default `--operator_pos` (7 points:
   head, torso, both shirt corners, both forearms, shorts) was tuned on frame 0 of `_077`.
   The FIRST attempt (4 points, torso only) left the extended forearm unmasked on frames
   like `pano_00081`; adding both-forearm positives fixed it, and the same points then
   generalised unchanged to the 8K theatre room and the .mov. *(That first-pass output was
   overwritten during tuning and is NOT persisted — only the final-defaults run is. To
   reproduce the failure: rerun with `--operator_pos
   "0.475,0.90;0.44,0.84;0.51,0.84;0.475,0.97" --operator_box_union 1`.)*
   A different rig/pose (overhead stick, shoulder mount) will need retuning — the points,
   the yaw/pitch clamp, the guards and the fallback box are all CLI-tunable for that reason.
3. **The core-box union is OFF by default** (`--operator_box_union 0`). Running it ON costs
   ~3% of every frame in real scene content (mean excluded 0.1778 vs 0.1546) for no measured
   coverage gain once the prompts were fixed — SAM never lost the operator across
   151 + 94 + 184 = 429 frames (0 lost-guard, 0 area-guard). The box remains the automatic
   fallback when SAM 2.1 is unavailable or a guard fires.
4. **Other people are NOT masked.** A second person walks through several `_077` and
   `FLOWERSHOP.mov` frames; only the operator is tracked. This is per spec, but it is a real
   dynamic distractor that WP-O3 must be checked against (expect ghosts). Adding a second
   tracked object is a small extension of `sam2_operator_masks`, deliberately not built here.
5. **The nadir band does not always fully cover the stick/hand.** At `--nadir_deg 12` a
   forearm/stick occasionally reaches a few degrees above the band on frames where the
   operator holds the rig low (e.g. `pano_00092`). SAM covers most of it; raising
   `--nadir_deg` to ~15 would close the rest at the cost of more floor. Left at 12 for v1.
6. **SAM 2 post-processing is disabled in this environment** — `sam2` is used from the
   source clone without its optional `_C` CUDA extension, so `fill_holes_in_mask_scores`
   is skipped (upstream's own warning: "doesn't affect the results in most cases"). No hole
   artefacts were observed in the overlays. Not fixed: building `_C` is a CUDA-extension
   build we do not need.
7. **ih_fov was NOT calibrated.** PLAN_360 §5 suggests a 192–200° checker seam-continuity
   sweep "once". Not done — 200/200 produced coherent panoramas at both 1920² and 3840²,
   and the seam bands are masked out anyway. The flags exist (`--ih_fov/--iv_fov`); the
   sweep belongs with WP-O2, where reprojection error can actually score it.
8. **PLAN_360 §5 says "yaw-normalize"; nothing is yaw-NORMALIZED and nothing needs to be.**
   Both fisheyes are rigidly mounted, so the ERP yaw origin is fixed relative to the camera
   BODY — the stitch path applies ONE constant `v360 …:yaw=` to every frame (checked in
   `build_filter_graph`; there is no per-frame term anywhere), so there is no per-frame yaw
   drift to remove. The only yaw choice is that constant rig→ERP offset, exposed as `--yaw`
   (default 0 = operator lens at ERP centre) and shifted 180° by `--operator_stream 0`.
   Orientation relative to the WORLD (the operator turns while walking; no IMU
   horizon-levelling is applied here — Insta360's own stabilisation is not replicated) is
   camera POSE and belongs to WP-O2, not to ingest. Documented in the module docstring.
9. **Stream order is an ASSUMPTION, not a detection — spot-check it per camera/firmware.**
   "stream 0 = scene, stream 1 = operator" was verified BY EYE on the 2026-07-21 clips only.
   For any new camera or firmware, extract one frame per stream and LOOK
   (`ffmpeg -i CLIP.insv -map 0:0 -frames:v 1 s0.jpg`, same for `0:1`) before trusting the
   default; pass `--operator_stream 0` if the operator is in stream 0. Getting it wrong
   silently scores the wrong lens for sharpness AND masks real scene content while the
   operator goes unmasked at the ±180° edges — the area/lost guards would NOT reliably catch
   that. Deliberately NOT auto-detected (added 2026-08-04, see §6).
10. **Sharpness values are not comparable across clips/resolutions** (the metric is computed
   on a central crop whose size depends on the source). Only within-clip window-relative
   comparison is used, which is all the selection needs — but do not read `_077` median 1691
   vs `_027` median 447 as "the 8K clip is blurrier"; the 8K crop is downscaled 2× first.

## 4. Not in scope / not touched

No GUI changes. No changes to any existing module (only additions:
`core/ingest_360.py`, `tests/test_ingest_360.py`, `_infusion/wpo1_evidence.py`,
`_infusion/wpo1/*`, plus doc/ledger edits). torch untouched, no pip installs, no CUDA builds.

## 5. Reproduce

```
python -m core.ingest_360 --capture-type insta360 --input <file> --out_dir <dir> --overwrite
python _infusion/wpo1_evidence.py          # re-derives _infusion/wpo1/* from the run dirs
python -m pytest tests/test_ingest_360.py -q
```

---

## 6. Review fixes (2026-08-04) — follow-up to the PASS-WITH-NOTES review

Hardening pass on the gate-passed module before WP-O2 starts running it repeatedly. **The
gate was NOT re-run and no gate-relevant behaviour changed**: with default flags the resolved
scan stream is still 0, the stitch yaw is still 0.0, and a smoke re-run of `_077.insv`
reproduces the recorded histogram exactly (min 244 · p10 454 · median 1691 · p90 2339 ·
max 3160, 151 selected @ 1.998 pano-fps, operator = SAM, 0 guards).

| # | finding | fix |
|---|---|---|
| 1 | **Stale outputs on re-run without `--overwrite`** — a shorter second run left the first run's orphaned image/mask pairs beside the new report | `prior_output()` + a re-run guard at the TOP of `ingest()` (before probe/scan, so a refused run does no work and writes nothing): if `out_dir` holds `images/`, `masks/`, `ingest_report.json` or `INCOMPLETE` and `--overwrite` is absent → `FileExistsError` → `ERROR: out_dir contains prior output (…); pass --overwrite`, **exit 2** |
| 2 | **Mid-pipeline failures surfaced raw tracebacks and left a partial `images/` with no masks/report** | (a) `main()` now also catches `RuntimeError` (ffmpeg `_run_pipe`, the frame-count sanity check, SAM) → same one-line `ERROR:` + exit 1; unexpected exception types still raise so real bugs keep their traceback. (b) any failure during a run drops an `INCOMPLETE` marker naming the exception in `out_dir` (chosen over deleting it: the partial frames stay debuggable, and the marker feeds the re-run guard). (c) `ingest_report.json` is still written only on success — verified by test. (d) SAM 2.1 predictor construction + propagation are now inside a `try` that re-raises as a clean `RuntimeError`; **no silent box fallback** for a SAM that is present but failing (the pre-existing "SAM not importable / checkpoint missing → box" path is unchanged and still prints why) |
| 3 | **Hardcoded stream order** (stream 0 = scene, stream 1 = operator at ERP centre) | new `--operator_stream {0,1}` (default 1 = the verified order). Wired through both places it matters: `scan_stream()` (sharpness scores the OTHER lens; `--sharpness_stream` default changed `0 → -1` = auto, an explicit value still wins) and `stitch_yaw()` (`--operator_stream 0` adds 180° so the operator always lands at ERP centre, where the prompt points, the operator clamp and the seam bands expect it). Recorded in `ingest_report.json` under `stream_order` (`operator_stream`, `scene_stream`, `auto_detected: false`, full assumption text) and `stitch.stitch_yaw`. NO auto-detection — see §3 item 9 for the spot-check procedure |
| 4 | **`yaw-normalize` in the PLAN_360 §5 spec vs a manual `--yaw` in the code** | checked the stitch path; there is no per-frame yaw term to normalize on a rigid dual-lens rig. Stated honestly in the module docstring + §3 item 8 above |
| 5 | RECORD §2c called the `FLOWERSHOP.mov` histogram a "2 pano-fps scan" | reworded: the histogram is over the full native-fps decode; 2 pano-fps is the selection rate |

### Evidence for the `--operator_stream` wiring (real footage, this machine)

```
# A: python -m core.ingest_360 --capture-type insta360 --input .../VID_20260721_133413_00_077.insv \
#      --out_dir <tmp>/o1smoke   --overwrite --max_frames 3 --contact_sheet 3 --contact_cols 3 --progress 5000
# B: python -m core.ingest_360 --capture-type insta360 --input .../VID_20260721_133413_00_077.insv \
#      --out_dir <tmp>/o1smoke_s0b --overwrite --operator_stream 0 --sharpness_stream 0 \
#      --operator_mask none --max_frames 1 --contact_sheet 0 --progress 5000
```
- A (defaults): `[lens] --operator_stream 1 … scene lens = stream 0, stitch yaw 0 deg`;
  same sharpness histogram as the gate run ⇒ **default behaviour byte-identical**.
- B (`--operator_stream 0`, same source frame 11 forced via the explicit `--sharpness_stream 0`):
  **MAE vs the 180°-rolled A frame = 0.049** (vs 50.60 unrolled) ⇒ the flip is exactly the
  180° ERP rotation it claims to be, and ffmpeg accepts `yaw=180` in the graph.
- B without the explicit override picked source frame **8** instead of 11 — i.e. the auto
  scan-stream really does follow `--operator_stream`.
- Re-run guard on a real dir: second invocation returned **exit 2** in <1 s with nothing
  written (no probe, no scan).
- Smoke outputs were written to the session scratchpad (not `_dataset/`) and are not
  committed; both argv lines above reproduce them.

### Tests

`tests/test_ingest_360.py` **22 → 32** (+10): `TestRerunGuard` (3 — `prior_output`
enumeration, guard fires before the probe and touches nothing, `INCOMPLETE` also blocks),
`TestStreamOrder` (4 — defaults, the 0-stream flip of scan+yaw, explicit override wins, ERP
input makes no stream choice), and 3 ffmpeg round-trips (report records the stream-order
assumption; run-5-frames → run-2-frames is refused and `--overwrite` rebuilds cleanly; a
forced-`dual_fisheye`-on-one-stream file fails mid-stitch → exit 1, no report, `INCOMPLETE`
written, re-run refused). Full suite **173 → 183 passing**.
