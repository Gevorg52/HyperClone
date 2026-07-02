# WP-F1 mono-depth A/B record (run 2026-07-02, evidence persisted 2026-07-02)

Two 15k-step runs on Laptop (`_dataset/Laptop/apex_images/training/gsplat_prod/data`),
factor 2, MCMC cap 1M, seed 0, run SEQUENTIALLY on an otherwise-idle GPU,
identical except the mono-depth flag. Checkpoints:
`_dataset/Laptop/apex_images/training/wpf1_ab/{off,on}/output/ckpts/ckpt_14999_rank0.pt`
(`_dataset/` is gitignored — the ckpts are recomputable from the argv below; all
gate NUMBERS are persisted in the committable artifacts listed at the bottom).

## Exact argv

Both arms (shared):
```
python -m core.training.gsplat_trainer \
  --data_dir _dataset/Laptop/apex_images/training/gsplat_prod/data \
  --data_factor 2 --max_steps 15000 --test_every 8 \
  --strategy mcmc --cap_max 1000000 --opacity_reg 0.01 --scale_reg 0.01 \
  --antialiased --seed 0 --save_steps 15000 \
  --result_dir _dataset/Laptop/apex_images/training/wpf1_ab/<ARM>/output
```
- `off/` — nothing else (mono_depth_lambda defaults to 0.0).
- `on/`  — adds `--mono_depth_lambda 0.05 --mono_depth_start 2000`
  (per-view alignment mode, 303/304 views reliable; `--seed 0` controls
  Prefetcher view order per the WP-Q2 fix, so the arms are view-order matched).

**Recipe caveat:** `pose_opt=false`, no coarse-to-fine — consistent across BOTH
arms (matches the WP-Q2 erank A/B recipe, `wpq2_ab/AB_RECORD.md`) but differs
from the production recipe (`pose_opt=True`, `coarse_factor 4`). Flag this when
comparing these absolute numbers against other gates' tables.

## Gate table (all values recomputed from the checkpoints and persisted)

| metric | OFF | ON (mono 0.05) | delta | verdict |
|---|---|---|---|---|
| masked PSNR (qa_report, 38 views) | 23.278 | 23.167 | −0.11 dB | ~equal |
| masked SSIM | 0.8417 | 0.8409 | −0.0008 | ~equal |
| **masked LPIPS** | 0.3546 | **0.3540** | −0.0006 (−0.2%) | **not worse ✓** |
| full-frame LPIPS (VGG, unmasked) | 0.3546 | 0.3540 | −0.0006 | not worse ✓ |
| deck-region LPIPS (crop 0.30–0.72 x, 0.55–0.95 y) | 0.3013 | **0.3002** | −0.0011 (−0.4%) | not worse ✓ |
| **depth residual, held-out median \|Δ\|/z** | 0.0652 | **0.0182** | **−72.0%** | **improved ✓** |
| depth residual mean | 0.2064 | 0.0344 | −83.3% | improved ✓ |
| depth residual p90 | 0.5991 | 0.0805 | −86.6% | improved ✓ |
| needle (erank<2) | 91.16% | 90.48% | −0.68 pp | cleaner |
| SOR-outlier | 0.02% | 0.01% | −0.01 pp | cleaner |
| giant-blob | 10.98% | 10.42% | −0.56 pp | cleaner |
| low-opacity | 10.5% | 10.0% | −0.5 pp | cleaner |

**GATE: PASSED.** Held-out masked-LPIPS not worse (marginally better), geometry
improves sharply, floater indicators all marginally cleaner. The ~null
appearance deltas on this texture-rich, well-captured scene match the
sparse-depth precedent (the scene doesn't exercise the appearance lever; the
target regime is starved/holey captures) — the geometry win is the payload.

## Honest characterization of the depth-residual metric

The mono-depth loss consumed **MoGe** depth on **train** views. The eval
measures **rendered** depth against **COLMAP sparse points** on **held-out**
camera poses (filters: reproj ≤2.5 px, track ≥5, depth∈[0.001,1000]; same
machinery as the sparse `--depth_lambda` loss). Two independence properties
hold: the eval target is a *different modality* from the supervision signal
(triangulated SfM vs monocular network prior), and the evaluated renders are
from poses the loss never supervised. One does NOT: the sparse 3D points are
triangulated from tracks that *include train views*, so the point cloud itself
is not fully independent of train-view geometry. The −72% median / −87% p90
should be read as "the supervised model's geometry agrees far better with the
SfM reconstruction on unseen views", not as agreement with ground-truth depth.

## Full-frame eval PSNR/SSIM (trainer's built-in eval, `results.json`)

off 23.278 / 0.8417 · on 23.166 / 0.8409 — the small negative PSNR trend is the
expected depth↔appearance trade (DEVELOPER_NOTES §6: depth priors fix geometry,
not appearance); judged by the LPIPS + geometry gate above, per plan §0.

## Persisted artifacts (committable)

- `qa/wpf1_ab_off/scorecard.json`, `qa/wpf1_ab_on/scorecard.json` — masked
  PSNR/SSIM/LPIPS per view + aggregate (deterministic; regenerate with
  `python qa_report.py <ckpt> <data> --factor 2 --out qa/wpf1_ab_{off,on}`).
- `_infusion/wpf1_depth_eval.py` → `_infusion/wpf1_ab_depth.json` — held-out
  depth-residual eval (script + output; includes per-view medians).
- `_infusion/wpf1_deck_eval.py` → `_infusion/wpf1_ab_deck.json` — deck-region +
  full-frame VGG-LPIPS (exact crop box in the JSON).
- `_infusion/wpf1_ab_floaters.txt` — floater_report output for both ckpts.
- This record.

Recomputation check (2026-07-02): every number above reproduced EXACTLY from
the checkpoints on re-run (qa_report is deterministic; depth/deck evals are
no_grad renders at fixed poses).

## Post-review fixes applied to the shipped code (same day)

1. `core/priors.py` global-mode `reliable` now requires the view to AGREE with
   the global scale (median residual under global scale ≤0.30 AND own scale in
   [0.2,5]×global), not merely "had ≥10 correspondences". Forced-global test on
   Laptop: reliable 303/304 (old logic) → **170/304** (new logic) — badly
   disagreeing views are now excluded rather than floor-weighted at 0.2.
   Production caches are unaffected (both scenes select per-view mode; Laptop
   303/304, GPU_V4 569/569 reliable — unchanged after rebuild).
2. `gsplat_trainer.py` hard-errors when the prior cache factor ≠ `--data_factor`
   (was: silent per-batch shape-mismatch no-op) and when 0 train views have a
   usable prior (was: printed "Mono-depth loss ON" while supervising nothing).
   Startup line now reports `N/M train views covered`.
3. Dead `mono_start` variable removed — `args.mono_depth_start` is now actually
   raised to `max(mono_depth_start, coarse_until+1)` and used as the loss gate.
