# UI-3 — Mask Review (SAM 2.1 point-prompts + video propagation) — GATE RECORD

**Built + gated 2026-07-09.** Object-mode Step-③ of the GUI 2.0 wizard: the user-facing fix
for "the shoe's stand got exported with the object" (GUI_REDESIGN.md §3③).

Files added/changed:
- `gui2/sam_service.py` (NEW) — SAM 2.1 image-predictor prompting (in-process) + video-predictor
  propagation (subprocess).
- `gui2/review.py` (NEW) — review logic (view picking, mattes, overlays, accept staging).
- `gui2/app.py` — Step-③ is now the real screen (gallery + Looks-right / Fix-it → SAM edit →
  propagate → spot-check → accept), plus a Step-② "Review masks (while training)" button.
- `gui2/defaults.py::assemble_delivery_flags(reviewed_mask_dir=…)` — reviewed masks → `--mask_dir`.
- `gui2/runner.py::_run_delivery` — passes reviewed masks when present (drops `--gen_masks`).
- `core/delivery.py::ensure_masks` — an explicit `--mask_dir` is now AUTHORITATIVE (re-stages over
  stale `masks_{factor}`). Both delivery/defaults changes are **default-compatible** (unchanged
  when no reviewed masks / no `--mask_dir`).

Evidence (all in `_infusion/`): `ui3_propagate_stats.json`, `ui3_spotcheck_final24.json`
(the truthful re-run; see Closeout §C4), `ui3_delivery_check.json`, `ui3_ui_smoke.json`,
`ui3_strips/{beforeafter_grid,spotcheck_4,delivery_render_check}.png`, re-runnable
`ui3_delivery_check.py` + `ui3_spotcheck_final24.py` + `ui3_ui_smoke.py`.
(`ui3_spotcheck_REJECTED_reseed.json` is the discarded global-reseed attempt, kept for
the trail — do NOT cite it.)

---

## 1. SAM 2.1 install outcome (+ sm_120)

- **Was NOT installed** (only a stale Feb egg-info). Repo `D:\REPOS\HYPERclone\sam2`; checkpoint
  `sam2.1_hiera_large.pt` (898 MB) **already present** (no download).
- Installed editable: `SAM2_BUILD_CUDA=0 pip install -e . --no-build-isolation -c
  tools/blackwell_session_2026-06/constraints.txt`. **torch 2.11.0+cu128 / torchvision 0.26.0+cu128
  UNTOUCHED** (already satisfy `torch>=2.5.1`); pip only added pure-python deps (hydra-core,
  omegaconf, iopath, portalocker).
- **sm_120 (RTX PRO 6000, cc 12.0): OK.** SAM 2 is plain attention / SDPA. The `_C`
  connected-components CUDA ext is **not built** (only used for hole/sprinkle fill); we set
  `predictor.fill_hole_area = 0` so the video predictor never calls it, and the image predictor's
  max_hole/sprinkle default to 0. Verified real forward passes:
  - image predictor: load 3.4 s, first predict 0.33 s, warm predict 5 ms, **1.42 GB VRAM**.
  - video predictor: full 588-frame track, **8.63 GB VRAM**, no `_C` error.

## 2. Propagation runtime @ 588 frames

- Frame prep (60 MP Sony **MPO** → 1024-long-edge JPEG video dir, PIL `draft` decode):
  **91.7 s** for 588 frames (3.2× faster than the naive 292 s; MPO needed adding to the draft
  format allow-list). One-time per scan on entering review.
- Single-seed forward+reverse propagation: **36.4 s** (16.2 fps), 8.63 GB VRAM, all 588 masks.
- 4-anchor gate run (1 primary + 3 sole-arc correctives): **93.5 s** (each seed tracked
  independently + merged), 8.63 GB VRAM.
- Resolution choice: propagate at the model's internal 1024 image-size (masks come back at that
  res; delivery + trainer `cv2.resize` masks to the image, so resolution is irrelevant downstream).
  Kept frames on GPU (`offload_video_to_cpu=False`); 8.6 GB leaves ample room to run alongside a
  3M-Gaussian training on the 96 GB card (an `--offload_video` flag exists if ever needed).

## 3. Stand-removal gate (SONY_TEST — the real shoe-on-stand case)

Seed = mid-orbit side view DSC01497 (idx 294): include the shoe (upper + toe + heel + midsole +
outsole), exclude the stand (hub + 2 prongs). Spot-check criterion = SAM mask not collapsed vs the
BiRefNet matte (fg>2% and SAM/BiRefNet area ratio ≥ 0.4).

| Anchors | Spot-check | Note |
|---|---|---|
| 1 (seed only) | **11/12 = 91.7%** | only the sole-on view (DSC01310) collapsed |
| + sole corrective (107, radius 45) | **23/24 = 95.8%** | sole-on fixed; sole-arc edges still weak |
| + 2 more sole-arc correctives (75, 150) | **40/40 = 100.0%** (every-15-frame sweep) | full sole/underside arc covered |

Before/after strips: `ui3_strips/beforeafter_grid.png` (12 views, BiRefNet-with-stand vs SAM
shoe-only), `ui3_strips/spotcheck_4.png`. Every SAM panel is shoe-only with the stand prongs/hub/
platform excluded across the full orbit. **The iterate loop is the mechanism** — a naive attempt to
re-seed a hard view as a global anchor REGRESSED (down to 75–79%) because SAM 2 propagates
sequentially from the min conditioning frame; the fix (per-seed independent tracks + nearest-seed
merge with a per-seed claim radius) keeps each correction local.

## 4. Delivery gate — the object SOG contains no stand

Ran `core.delivery` with `--extract_object --mask_dir masks_reviewed` on the 3M-Gaussian SONY
master (SH3, factor 2, `output_t2` ckpt), `--vote_stride 2 --no_qa`. Result vs the pre-existing
BiRefNet-matte baseline (`sony_shoe_detail_object.ply`, stand INCLUDED):

| Extract | # Gaussians | mean out-of-shoe-mask alpha (4 spread views) |
|---|---|---|
| REVIEWED (shoe-only masks) | 884,140 | **0.013** (clean cut) |
| BASELINE (BiRefNet, +stand) | 942,289 | 0.104 (stand renders outside the shoe) |

`stand_removed = true` (reviewed 0.013 < 0.02 and < ½ of baseline). Render strip
`ui3_strips/delivery_render_check.png`: the baseline splat shows the shoe **plus** the red-tinted
stand (prongs/hub/platform); the reviewed splat shows **shoe only** (only faint mask-edge residue).
Reviewed delivery output: `_dataset/SONY_TEST/deliverables_reviewed/sony_shoe_object.{ply,sog}`.

## 5. Delivery wiring choice

`masks_reviewed/` (COLMAP stems) is the canonical accepted-mask store. `accept_reviewed` stages
it ATOMICALLY (copy into `masks_reviewed.tmp/`, write a `.complete` marker holding the mask count
LAST, then `os.replace`-swap into place) so a cross-process delivery read never sees a half-copied
dir (Closeout §C3). The runner passes `--mask_dir masks_reviewed` (dropping `--gen_masks`) **only
when the `.complete` marker is present AND its count matches the on-disk `.png` count**; otherwise
it falls back to `--gen_masks` (BiRefNet). `ensure_masks` treats an explicit `--mask_dir` as
authoritative AND judges completeness by masks actually COPIED FROM `--mask_dir` (not by counting
whatever sits in `masks_{factor}`), so a stale dst file can never silently fill a gap — a partial
`--mask_dir` RAISES (Closeout §C1). No mask-resolution handling needed: both trainer
(`gsplat_trainer.py:426`) and delivery (`mask_in_render_frame`) `cv2.resize` masks to the image.
"Looks right" writes NO `masks_reviewed`, so delivery reproduces the approved matte via its own
`--gen_masks` (BiRefNet) path.

## 6. UI flow smoke (programmatic) — 12/12

`ui3_ui_smoke.py`: app builds; view picking; frame prep (reuse); representative mattes; gallery
items; live `prompt_frame` (fg 0.176); overlay; `accept_reviewed` (588 staged); delivery flags
(reviewed → `--mask_dir`, default → `--gen_masks`); `clear_reviewed`. Full pytest suite: **146
passed**.

## 7. UI limitations (honest)

- **Exclude interaction:** Gradio `gr.Image.select` carries no mouse button, so include vs exclude
  is a `gr.Radio` mode toggle ("Click adds: Include / Exclude"), not a right/alt-click. Faithful to
  the design intent; per-frame brush touch-up is deferred to the funded phase.
- **Hard sole/underside arc:** a 360° turntable of a shoe-on-a-stand needs 1–3 corrective clicks in
  the sole arc to clear ≥95% — single-seed alone is ~92%. This is the point of the review step.
- **Parallel review → delivery hand-off is safe, not racy:** delivery reads `masks_reviewed` at
  delivery time, gated on the atomic `.complete` marker + count-match (Closeout §C3). If the user
  is still reviewing (or has not yet accepted) when training finishes, there is no valid marker and
  delivery falls back to BiRefNet; re-delivering after acceptance uses the reviewed masks.
- **Propagation blocks the Gradio handler** (~1.5 min for a 4-anchor 588-frame run); it runs as a
  subprocess (cancellable pattern available) but the current Step-③ handler `.wait()`s on it.

---

## Closeout — code-review fixes (2026-07-09)

Reviewer findings addressed (finding #1 was REPRODUCED). Full suite after: **151 passed**
(was 146; +5 new in `tests/test_delivery_masks.py`).

- **C1 (HIGH, reproduced) — `ensure_masks` no longer mixes fresh+stale masks.** With an explicit
  `--mask_dir`, completeness is now judged by masks actually COPIED FROM `--mask_dir` vs the required
  stems (not by counting any file already in `masks_{factor}`). A partial `--mask_dir` RAISES with
  the missing count + example stems (or, with `--gen_masks`, BiRefNet fills only the still-missing
  stems). The None-`--mask_dir` path is unchanged. New `tests/test_delivery_masks.py` reproduces the
  reviewer's scenario (mask_dir 1/2 stems + a stale dst file for the other → RAISES post-fix) plus
  the None-path-unchanged cases. `core/delivery.py::ensure_masks`.
- **C2 (HIGH) — `_propagate` now honors the subprocess result.** A nonzero exit or a missing stats
  file surfaces a failure message in the Step-③ status, hides the spot-check, and BLOCKS Accept
  (accept button stays hidden and `prop_out_dir` is cleared so a stale prior result can't be
  accepted). The stale stats file is deleted before launch so a prior run's success can't
  masquerade. `gui2/app.py::_propagate`.
- **C3 (HIGH) — accept/delivery race removed.** `accept_reviewed` now stages into
  `masks_reviewed.tmp/`, writes a `.complete` marker (staged count) LAST, then `os.replace`-swaps
  the dir into place (Windows-safe: move-aside → rename → delete-old; the only non-atomic window
  leaves `masks_reviewed` briefly ABSENT, which makes a concurrent reader fall back to `--gen_masks`,
  never read a partial dir). The runner requires the marker AND a matching on-disk `.png` count,
  else `--gen_masks`. The existing SONY_TEST `masks_reviewed/` got a `.complete` marker (588).
  `gui2/review.py::accept_reviewed` + `MASKS_COMPLETE_MARKER`, `gui2/runner.py::_run_delivery`.
- **C4 (MODERATE, evidence) — the "23/24 = 95.8%" number is RE-CONFIRMED with the final code.**
  Re-ran the 24-view spot-check via the shipped per-seed-merge `gui2.sam_service.propagate` at the
  crediting iteration (anchors `[294 primary, 107 radius=45 corrective]`, from `sam_work/prompt.json`),
  scored with the same collapse detector (`sam_frac>0.02 and sam/bref ratio≥0.4`; BiRefNet
  denominator = `masks_2_birefnet_bak`, per-stem fractions match the original sweep to 3 dp):
  **23/24 = 95.8%** (propagate 56.9 s, 8.63 GB VRAM; single fail DSC01280 ratio 0.321). Banked at
  `_infusion/ui3_spotcheck_final24.json` (driver `ui3_spotcheck_final24.py`). The stale
  `ui3_spotcheck_metrics.json` (19/24 = 79.2%, from the REJECTED global-reseed attempt) is renamed
  `ui3_spotcheck_REJECTED_reseed.json`. So the RECORD's original 95.8% was correct; the leftover
  79.2% JSON was the misattribution.
- **C5 (MODERATE) — "Review masks (while training)" button gates on images-done, not sparse/0.**
  `create_3dgs_directory` copies `sparse/0` FIRST and images AFTER, so the button now enables only
  once `export/images` holds ≥ the registered image count (read from `sparse/0/images.bin`'s leading
  `num_reg_images` uint64, falling back to `images.txt`). `gui2/app.py::_export_ready_for_review` +
  `_colmap_num_images`, used in `_tick`.
- **C6 (nits).** (a) GPU selection unified: `gui2/sam_service.py` propagate subprocess now selects
  the device the same way as `gui2/preview.py` — explicit `torch.cuda.set_device` + `cuda:<idx>`,
  no `CUDA_VISIBLE_DEVICES` mutation. (b) Intentional SONY_TEST dirs, noted for reviewers:
  `_dataset/SONY_TEST/sam_work/` (SAM 2 working frames + `index_map.json` + `prompt.json` +
  propagated masks — the re-runnable review scratch) and
  `_dataset/SONY_TEST/training/gsplat/data/masks_2_birefnet_bak/` (byte-identical BiRefNet-matte
  backup; `masks_2`'s documented resting content, and the spot-check's BiRefNet denominator).

**Delivery gate re-confirmed with the hardened `ensure_masks`:** `ui3_delivery_check.py` (now
self-staging: it stages `masks_2 ← masks_reviewed` via `ensure_masks` at the top, so it is
re-runnable regardless of `masks_2`'s resting content and exercises the delivery mask-staging path)
→ mean out-of-shoe-mask alpha **reviewed 0.013 vs baseline 0.104, `stand_removed = true`** (identical
to the original run; `_infusion/ui3_delivery_check.json`). `masks_2` was left in its BiRefNet
resting state after validation.
