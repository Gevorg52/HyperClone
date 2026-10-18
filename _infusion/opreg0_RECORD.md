# STEP 0 — `opacity_reg` default defect (0.01 -> 0) — RECORD (2026-10-17)

**Verdict: GATE PASSED, and the change is a large production IMPROVEMENT, not just "no
regression".** On the gui2 production stack (WP-S1 Laptop STACK recipe, identical argv and seed,
only `--opacity_reg` varied) setting `opacity_reg` 0 instead of the old default 0.01 gives held-out
masked LPIPS **0.3098 -> 0.2794 (-9.8 %)**, masked PSNR **+1.08 dB**, deck-region LPIPS **-12.3 %**,
and removes most of the low-opacity fog: Gaussians below opacity 0.05 **70.9 % -> 26.7 %**, fog-layer
alpha at off-path viewpoints **0.576 -> 0.049**. The seed-0 repeat of the old config reproduces
the banked July WP-S1 run to within 0.0002 LPIPS, so the effect is ~150x the run-to-run noise.

The defect was real: `opacity_reg` had defaulted to 0.01 since commit `a4adc60` (2026-06-11,
"Modernize 3DGS trainer: MCMC ..."), and neither GUI, gui2 nor any `run_*` script overrode it.
WP-O3/O5 had shown 0.01 and 0.001 destroy many-view walkthroughs; this record shows it was ALSO
costing ~10 % LPIPS on every object-orbit production run.

---

## 1. Code change (default 0; still available as an opt-in flag)

| file:line | before | after |
|---|---|---|
| `core/training/gsplat_trainer.py:1945` | `--opacity_reg` default `0.01` | default `0.0` + help text (opt-in, history) |
| `core/training/gsplat_backend.py:250` | `"opacity_reg": 0.01` | `0.0` (GUI v1 + gui2 + delivery all inherit this) |
| `run_train_laptop.py:60` | `0.01` | `0.0` |
| `run_train_ab.py:35` | `0.01` | `0.0` |
| `run_distill_laptop.py:63` | `0.01` | `0.0` |
| `configs/defaults.yaml:76` | `0.01` ("required for MCMC") | `0.0` (documentation file; not loaded by code) |

Who sets it / inherits it:
- `gui2/defaults.py` does **not** set `opacity_reg` -> inherits the backend default (was 0.01, now 0).
- `gui/stages/training.py` (GUI v1) does not set it -> inherits the backend default.
- `core/delivery.py::self_refit` runs the trainer CLI without the flag -> inherits the trainer
  default (was 0.01; refit has densification off, `refine_stop_iter=1`).
- **360 path:** no production 360 training driver sets it; the WP-O3/O5 drivers
  (`_infusion/wpo3_train*.cmd`, `_infusion/wpo5_train.py`) always passed `--opacity_reg 0`
  explicitly (except the deliberate 0.01/0.001 test arms). `core/ingest_360.py` and
  `core/train_360_prep.py` do not train.
- `scale_reg` is unchanged (0.01) — WP-O3 isolated it as harmless.

Tests: `python -m pytest tests -q` -> **331 passed** after the change.

## 2. Which runs / deliverables were trained with 0.01

`results.json` does not record `opacity_reg`, so this is reconstructed from driver scripts,
`argv.txt` files and the default history (0.01 from 2026-06-11 until today): **every trainer run
under `_dataset/` that did not pass `--opacity_reg 0` explicitly used 0.01.**

- **Used 0.01 (affected):** all GPU / GPU_V3 / GPU_V4 runs (`gsplat_prod`, `gsplat_native`,
  `wpq2_*`, `wps1/{ctrl,stack,n1/na,n1/nb}`, `wps2/*`, `wpx1/cap{3,4,6,8}m`), all Laptop runs
  (`gsplat_prod/output_4k*`, `ab_*`, `wpf1_ab`, `wpf2_ab`, `wpf4_*`, `wps1/{ctrl,stack}`,
  `train*`, distill v1/v2/v3 rounds via `run_distill_laptop.py`), SONY_TEST
  (`training/gsplat`, `output_t2` — `train_sony*.sh` / `priors_then_train.bat` pass 0.01
  explicitly), the UI1/UI2/UI4 smoke runs (gui2 -> backend default), and 360
  `wpo3_run_baseline` + `wpo3_diag/{A2,B,H}` (the collapsed arms).
- **Delivered artifacts built on 0.01 models:** `_dataset/SONY_TEST/deliverables/sony_shoe_detail_object.{ply,sog}`
  and `_dataset/SONY_TEST/deliverables_reviewed/sony_shoe_object.{ply,sog}` (masters from
  `output_t2`; delivery refit also at 0.01), `_dataset/UI{1_CANCEL,1_SMOKE,2_SMOKE,4_SMOKE}/deliverables/*.sog`.
- **Used 0 (unaffected):** 360 `wpo3_run_arm2`, `wpo3_diag/{D,E,G}`, `wpo5_run_{A,B,D}`
  (incl. the delivered `wpo5_run_D/ply/point_cloud_29999_aligned_pruned_op05.ply`).
  `wpo5_run_C` used 0.001 (deliberate negative arm).
- Consequence: every banked object-orbit number (WP-Q1 baselines, WP-S1, WP-F1..F4, WP-X1,
  WP-S2, T1/T2) was measured with 0.01 in BOTH arms, so those A/B *deltas* stand, but the
  absolute numbers are ~10 % LPIPS pessimistic. Re-banking baselines at 0 is the natural
  follow-up (not done here — one regression run was the brief).

## 3. Regression A/B (Laptop, WP-S1 STACK = gui2 production recipe)

Driver `_infusion/opreg0_run.sh <arm>`; eval `_infusion/opreg0_eval.sh or0 or01 wps1`;
table `_infusion/gate_summarize.py` -> `_infusion/opreg0_laptop_gate.json`.

**Common argv (all arms, seed 0):**
```
python -m core.training.gsplat_trainer --data_dir _dataset/Laptop/apex_images/training/wps1/stack/data
  --result_dir <arm>/output --data_factor 2 --max_steps 30000 --test_every 8 --seed 0
  --sh_degree 3 --ssim_lambda 0.2 --strategy mcmc --cap_max 3000000 --opacity_reg <0|0.01>
  --scale_reg 0.01 --antialiased --normalize_world_space --pose_opt --pose_opt_lr 1e-5
  --coarse_factor 4 --coarse_until 15000 --lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000
  --lpips_gated --mono_depth_lambda 0.05 --mono_depth_start 2000 --mono_depth_use_weight
  --erank_lambda 0.01 --erank_start 7000 --save_steps 7000 30000
```
| arm | opacity_reg | env | run |
|---|---|---|---|
| `or0` | **0** (new default) | system python, gsplat 1.5.3 | 2026-10-17, 8631 s (concurrent with or01) |
| `or01` | 0.01 (old default) | same | 2026-10-17, 8630 s |
| `wps1` | 0.01 | same recipe, banked 2026-07-04 | WP-S1 STACK ckpt (reference / noise floor) |

### Gate table — held-out (Laptop every-8, 38 views, factor 2, WP-Q1 masked metrics + 6 scorers)
| arm | mLPIPS | mPSNR | mSSIM | deck LPIPS | crossscore | nova | tsed | raft-warp | jod |
|---|---|---|---|---|---|---|---|---|---|
| **or0 (0)** | **0.2794** | **23.037** | **0.8311** | **0.2400** | 0.8651 | **0.1180** | 0.970 | 0.0418 | -2.859 |
| or01 (0.01) | 0.3098 | 21.956 | 0.7997 | 0.2737 | 0.8644 | 0.1203 | 0.970 | **0.0397** | -2.859 |
| wps1 (0.01, July) | 0.3096 | 21.957 | 0.8001 | 0.2739 | 0.8636 | 0.1221 | 0.939 | 0.0393 | -2.868 |

### Fog / free-viewpoint (`_infusion/offpath_orbit.py`, 21 off-path viewpoints per arm)
Viewpoints: per 4 anchors dolly-in 0.6x, pull-out 1.5x, +25 deg / -20 deg elevation re-aimed,
lateral 0.25x translation; plus the largest-azimuth-gap midpoint. `foglayerA` = mean alpha the
opacity<0.05 Gaussians ALONE deposit at those viewpoints; `prune_dMAE` = mean |full - pruned|
render difference (0-255) when they are deleted.
| arm | ckpt op<0.05 | mean opacity | foglayerA (off-path) | prune_dMAE | needle% | giant% | SOR% |
|---|---|---|---|---|---|---|---|
| **or0** | **26.7 %** | **0.538** | **0.049** | **0.23** | **8.87** | **11.16** | 0.03 |
| or01 | 70.9 % | 0.073 | 0.576 | 1.18 | 15.81 | 12.43 | 0.03 |
| wps1 | 71.2 % | 0.073 | 0.569 | 1.21 | 15.32 | 12.72 | 0.03 |

Pruned PLYs (opacity < 0.05 removed, `tools/prune_ply.py`, stats in `qa/opreg0_laptop_<arm>/prune_stats.json`):
`_dataset/Laptop/apex_images/training/opreg0/{or0,or01}/output/ply/point_cloud_29999_pruned_op05.ply`,
`.../wps1/stack/output/ply/point_cloud_29999_pruned_op05.ply`. With 0, pruning removes 26.7 %
of Gaussians; with 0.01 it removes 71 % (the model was mostly fog).

**Visual (matched off-path sheets `qa/opreg0_laptop_offpath_or0_or01_wps1/sheets/`):** the arms
look alike at moderate displacement; or0 is slightly crisper (table grain, lid edge) and the
lid's top edge is less veiled at the -20 deg elevation view (`cmp_13`). No new artifacts in or0.

### Noise floor
or01 (rerun today) vs wps1 (July): mLPIPS 0.3098 vs 0.3096, PSNR 21.956 vs 21.957 — the
seeded trainer reproduces to ~0.0002 LPIPS. The or0 effect (-0.0304) is ~150x that.

### Second scene (GPU_V4) — added 2026-10-18 from the WP-F5 baseline arm
The WP-F5 `gpu base` arm is exactly the WP-S1 GPU_V4 STACK argv with `--opacity_reg 0`, seed 0, but
trained in the gsplat-main venv (WP-B2 showed gsplat main reproduces 1.5.3 to within noise on this
recipe). Compared against the banked WP-S1 GPU_V4 STACK (0.01, gsplat 1.5.3; on Laptop the same
bank reproduced to 0.0002 LPIPS). 72 held-out views, same evaluator (`qa/wpf5_gpu_base/`,
`qa/wps1_gpu_stack/`, shiny box (0.28,0.72,0.30,0.55)):
| arm | mLPIPS | mPSNR | mSSIM | shiny LPIPS | needle% | giant% |
|---|---|---|---|---|---|---|
| **opacity_reg 0** (`wpf5/base`) | **0.3698** | **24.744** | **0.8437** | **0.2997** | **5.91** | **6.82** |
| 0.01 (WP-S1 stack, July) | 0.4071 | 22.933 | 0.8249 | 0.3473 | 10.58 | 12.09 |
-> **-9.2 % masked LPIPS, +1.81 dB, -13.7 % shiny-region LPIPS** — the Laptop result replicates.

### Honest caveats
- Two object-orbit scenes (Laptop A/B under identical conditions; GPU_V4 vs the July bank, with
  a gsplat-version difference shown to be noise-level by WP-B2).
- raft-warp (temporal consistency on a rendered trajectory) is 5 % worse with 0 (0.0397 ->
  0.0418); tsed/flicker/jod are equal. Small, but it is the one axis that moved the wrong way.
- Mechanism (consistent with WP-O3): with 0.01 the opacity pressure acts on every Gaussian
  every step, so MCMC ends with 71 % near-dead Gaussians (mean opacity 0.07) that are cheap
  fog; at 0 the same budget is spent on visible surface (mean opacity 0.54) -> sharper.

## 4. Persisted artifacts
- `_infusion/opreg0_RECORD.md` (this), `_infusion/opreg0_run.sh`, `_infusion/opreg0_eval.sh`,
  `_infusion/gate_summarize.py`, `_infusion/offpath_orbit.py`, `_infusion/opreg0_laptop_gate.json`,
  `_infusion/opreg0_sheets/` (committable copies of the key off-path + fog-pair sheets; `qa/**/*.jpg`
  is gitignored)
- `qa/opreg0_laptop_{or0,or01,wps1}/` (scorecard.json, report.html, region_deck.json,
  floaters.txt, prune_stats.json), `qa/opreg0_laptop_offpath_or0_or01_wps1/` (offpath.json + sheets)
- ckpts/train logs: `_dataset/Laptop/apex_images/training/opreg0/{or0,or01}/` (gitignored;
  recomputable from the argv above)
