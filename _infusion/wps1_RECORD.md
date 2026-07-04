# WP-S1 — Stacked-levers production run — RECORD (run 2026-07-03/04)

**Verdict: GATE PASSED (primary criteria) — with an honest floater caveat.** Stacking all
four validated faithful levers (dense-init + mono-depth 0.05 + erank 0.01@7k + the
existing gated-LPIPS 0.2 production recipe) beats the production control on held-out
**masked-LPIPS AND the flat/shiny region crops on BOTH real captures**, and collapses
degenerate needle Gaussians ~6-9× — with NO negative lever interaction (stack ≥ control on
every primary metric, so the pre-approved drop-one-lever probe was not needed). The one
regression is a modest giant-blob rise (the known erank isotropization side-effect). The
improvements are **modest** (masked-LPIPS −0.85 to −1.5%, region-LPIPS −1.2 to −1.5%): on
these well-captured, texture-adequate 360° orbits the levers' payoff is mostly geometric
cleanup + slight sharpening, NOT a visual leap. **The view-dependent specular reflections on
the flat/shiny surfaces remain unmodeled in both arms** — the clear signal for a dedicated
specular lever (see "What still looks wrong").

---

## Runs (4 × 30k, factor 2, seed 0, sequential on the RTX PRO 6000)

Trainer invoked directly (`python -m core.training.gsplat_trainer`) via
`_infusion/wps1_run.sh` (driver) + `_infusion/wps1_chain.sh` (sequential chain).
Production recipe reconstructed from `run_train_laptop.py` (the validated recipe) + the
banked GPU_V4 `results.json` (pose_opt, coarse_factor 4, factor 2, mcmc, antialiased, 30k).

**Common argv (BOTH arms, BOTH scenes):**
```
--data_factor 2 --max_steps 30000 --test_every 8 --seed 0 --sh_degree 3 --ssim_lambda 0.2
--strategy mcmc --cap_max 3000000 --opacity_reg 0.01 --scale_reg 0.01
--antialiased --normalize_world_space --pose_opt --pose_opt_lr 1e-5
--coarse_factor 4 --coarse_until 15000
--lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000 --lpips_gated
--save_steps 7000 30000
```
**STACK arm adds the three new levers:**
```
--mono_depth_lambda 0.05 --mono_depth_start 2000 --mono_depth_use_weight
--erank_lambda 0.01 --erank_start 7000
```
plus dense-init: the STACK data dir's `sparse/0` IS the RoMa-augmented COLMAP model
(WP-F2; no trainer flag — the trainer loads whatever `sparse/0` it is given). CTRL uses
the original ALIKED sparse and none of the three levers = reproduces production @ seed 0.

**mono_depth_start note:** the trainer raises it to `coarse_until+1 = 15001` (priors are
fine-res; coarse phase renders at factor 4). So mono-depth supervises only the fine phase
(steps 15001–30000). Startup line confirmed `303/304 reliable, 266/266 train views covered`
(Laptop) — prior cache factor-check passed.

**Data dirs** (images_2/images_4/priors are directory JUNCTIONS to each scene's
`gsplat_prod/data`; only `sparse/0` differs between arms — WP-F2 methodology):
| arm | data_dir | sparse/0 points |
|---|---|---|
| Laptop ctrl | `_dataset/Laptop/.../wps1/ctrl/data`  | 93,126 (orig ALIKED) |
| Laptop stack| `_dataset/Laptop/.../wps1/stack/data` | **707,268** (dense, reused WP-F2 `wpf2_ab/on`) |
| GPU_V4 ctrl | `_dataset/GPU_V4/.../wps1/ctrl/data`  | 123,289 (orig ALIKED) |
| GPU_V4 stack| `_dataset/GPU_V4/.../wps1/stack/data` | **882,902** (dense, built this WP) |

Checkpoints: each `<arm>/output/ckpts/ckpt_29999_rank0.pt` (708 MB, 3,000,000 Gaussians).

**GPU_V4 dense-init (built this WP, seed 0)** — `core.dense_init`, RoMa outdoor, 560×560
fp16, num_matched 5, 15000 samples/pair, cert≥0.05, voxel_scale 1.0:
123,289 → **882,902 points (7.16×)**, +759,613 (−281,770 voxel-dup), new-point mean reproj
**1.39 px**, poses preserved (max|Δt|=0.0, asserted), 2,845 pairs, 41.7 min
(34 match + 7.3 triangulate). Stats: `_dataset/GPU_V4/.../wps1/dense_work/dense_init_stats.json`.
(Laptop dense model reused from WP-F2: 93,126→707,268, 7.59×, reproj 1.42 px.)

---

## GATE TABLE 1 — held-out masked scorecards (qa_report, factor 2, every-8, 6 scorers)
Laptop = 38 views, GPU_V4 = 72 views. Masks are each model's own rendered alpha (OGGSplat;
`--vis_view_count` OFF — identical protocol to the banked baselines). obs% = 100 for all.
Scorecards: `qa/wps1_{laptop,gpu}_{ctrl,stack}/scorecard.json`.

### Laptop (semi-gloss black plastic deck/lid)
| metric | ctrl | **stack** | Δ (stack−ctrl) | verdict |
|---|---|---|---|---|
| **masked LPIPS** | 0.3143 | **0.3096** | **−0.0047 (−1.50%)** | **stack better ✓ (PRIMARY)** |
| masked PSNR | 21.982 | 21.957 | −0.025 dB | ~equal (LPIPS↔PSNR trade) |
| masked SSIM | 0.8022 | 0.8001 | −0.0021 | ~equal |
| crossscore ↑ | 0.8620 | 0.8636 | +0.0016 | slightly better |
| nova ↓ | 0.1187 | 0.1221 | +0.0034 | slightly worse |
| tsed ratio ↑ | 0.939 | 0.939 | 0.000 | equal |
| raft-warp ↓ | 0.0396 | 0.0393 | −0.0003 | slightly better |
| nvs-sqa (jod) ↑ | −2.856 | −2.868 | −0.012 | ~equal |
| flicker ↑ | 0.9942 | 0.9939 | −0.0003 | equal |

### GPU_V4 (semi-gloss black metal backplate / metal shroud+fins)
| metric | ctrl | **stack** | Δ (stack−ctrl) | verdict |
|---|---|---|---|---|
| **masked LPIPS** | 0.4106 | **0.4071** | **−0.0035 (−0.85%)** | **stack better ✓ (PRIMARY)** |
| masked PSNR | 23.043 | 22.933 | −0.110 dB | slightly worse (trade) |
| masked SSIM | 0.8260 | 0.8249 | −0.0011 | ~equal |
| crossscore ↑ | 0.9075 | 0.9007 | −0.0068 | worse |
| nova ↓ | 0.0230 | 0.0241 | +0.0011 | slightly worse |
| tsed ratio ↑ | 1.000 | 1.000 | 0.000 | equal (perfect both) |
| raft-warp ↓ | 0.0142 | 0.0142 | 0.000 | equal |
| nvs-sqa (jod) ↑ | −2.387 | −2.362 | +0.025 | slightly better |
| flicker ↑ | 0.9884 | 0.9885 | +0.0001 | equal |

## GATE TABLE 2 — flat/shiny REGION crops (VGG-LPIPS normalize=True, `wps1_region_eval.py`)
Laptop **deck** box = (0.30,0.72,0.55,0.95) [WP-F1/F2 lineage].
GPU_V4 **shiny** box = **(0.28,0.72,0.30,0.55)** — sits on the semi-gloss black metal
backplate (etched pattern + chrome ZOTAC logo) in backplate views and the metal shroud-top
+ fans in fan views. Reference image: `_infusion/wps1_gpu_shinybox_ref.jpg` (view48, idx384).
JSON: `_infusion/wps1_{laptop,gpu}_region.json`.

| scene / region | metric | ctrl | **stack** | Δ | verdict |
|---|---|---|---|---|---|
| Laptop **deck** | region LPIPS | 0.2773 | **0.2739** | **−0.0034 (−1.23%)** | **stack better ✓** |
| Laptop deck | region PSNR | 22.547 | 22.448 | −0.099 dB | ~equal |
| GPU_V4 **shiny** | region LPIPS | 0.3527 | **0.3473** | **−0.0054 (−1.53%)** | **stack better ✓** |
| GPU_V4 shiny | region PSNR | 22.418 | 22.480 | +0.061 dB | slightly better |

## GATE TABLE 3 — floaters (`floater_report.py`; lower cleaner except mean-erank)
Files: `_infusion/wps1_{laptop,gpu}_floaters.txt`.

| scene | arm | needle% | SOR% | giant% | low-opa% | mean-erank |
|---|---|---|---|---|---|---|
| Laptop | ctrl | 89.03 | 0.04 | 8.34 | 47.1 | 1.501 |
| Laptop | **stack** | **15.32** | 0.03 | **12.72** | 43.1 | **2.154** |
| GPU_V4 | ctrl | 92.15 | 0.17 | 8.90 | 58.2 | 1.532 |
| GPU_V4 | **stack** | **10.58** | 0.06 | **12.09** | 51.3 | **2.175** |

**Needles collapse 89.0→15.3% (Laptop) and 92.2→10.6% (GPU) — a 5.8–8.7× reduction of the
dominant speck-floater / zoom-aliasing artifact** (the erank hinge target, cf. WP-Q2). SOR
outliers cleaner-or-equal, low-opacity cleaner, mean effective-rank healthier (1.5→2.2).
**Honest regression:** giant-blob % rises +4.4pp (Laptop) / +3.2pp (GPU). This is the erank
isotropization side-effect — former thin needles become rounder and cross the fixed-size
"giant" bar; with mean-erank rising to 2.2 these are isotropic blobs, not pathological thin
discs, so the distribution is net-healthier. Recorded as a real but benign sub-regression.

## GATE VERDICT
1. masked-LPIPS: stack beats ctrl on **both** scenes ✓ (−1.50% Laptop, −0.85% GPU).
2. flat/shiny region crops: stack beats ctrl on **both** scenes ✓ (−1.23% deck, −1.53% shiny).
3. floaters "not worse": needles collapse 6–9×, SOR/low-opa/erank all cleaner-or-healthier;
   giant-blob +3–4pp (benign isotropization side-effect). **Net substantially cleaner.**
→ **GATE PASSED.** No negative lever interaction (mono-depth × dense-init × erank co-ran for
the first time and compose positively — stack ≥ ctrl on every primary metric), so the
pre-approved "drop-one-lever 15k probe" was unnecessary.

---

## Zoom strips — what STILL looks wrong on the shiny surfaces (feeds the specular decision)
Strips (GT | ctrl | stack, full frame + 2× region crop): `qa/wps1_{laptop,gpu}_strips/`.

Visual read (region2x crops):
- **GPU_V4 backplate (view48/view24):** both arms render the semi-gloss backplate cleanly
  (no floaters/holes) and capture the high-contrast chrome ZOTAC logo. But the **fine metal
  microstructure is softened in both** — the heatsink fin ridges (view24, fine parallel
  metal lines) smear together, and the etched circuit traces + "LIVE TO GAME" microtext
  (view48) blur out. Stack is marginally crisper (consistent with −1.5% region LPIPS) but
  does not resolve it.
- **Laptop lid/deck (view8/view0):** the **glossy black lid reflecting the outdoor
  environment renders as a mushy brown/dark smear in BOTH arms** — the classic
  view-dependent specular failure (a moving mirror reflection the SH-degree-3 splat cannot
  track). Keyboard key edges, palmrest micro-texture, and the AORUS sticker microtext are
  softened in both; stack marginally sharper.

**Diagnosis for the specular lever:** the four faithful levers give clean geometry (dense
init + mono-depth), kill needles (erank), and slightly sharpen (gated LPIPS) — but they add
**no view-dependent specular modeling**. The two residual failures on flat/shiny surfaces
are (1) **view-dependent specular reflections** (glossy lid mirror; metal sheen) rendered as
smears, and (2) **fine high-frequency microstructure** on dark low-texture metal/plastic,
partly a factor-2 resolution limit. (1) is the strongest, most visible failure and the clear
case for a dedicated specular/reflection lever; the faithful stack alone will not fix it.

---

## "Did the user's captures get better?" (vs OLD banked baselines) — plain language
Banked scorecards: `qa/GPU_V4` (2M cap, factor 2, ckpt_29999) and `qa/Laptop`
(4K/factor-1 training `output_4k`, evaluated factor 2).

| | Laptop bank | **Laptop stack** | GPU_V4 bank | **GPU_V4 stack** |
|---|---|---|---|---|
| masked LPIPS ↓ | 0.3540 | **0.3096 (−12.6%)** | 0.4125 | **0.4071 (−1.3%)** |
| masked PSNR ↑ | 22.319 | 21.957 (−0.36) | 25.278 | 22.933 (−2.35) |
| masked SSIM ↑ | 0.8120 | 0.8001 | 0.8514 | 0.8249 |
| crossscore ↑ | 0.825 | 0.864 | 0.9116 | 0.9007 |
| tsed ↑ | 0.909 | 0.939 | 1.000 | 1.000 |

**Plain answer:** Judged by the metric we trust for perceptual quality (LPIPS), **yes on
Laptop** — the stacked model is clearly better than the bank (−12.6% LPIPS, +crossscore,
+tsed) despite being trained at LOWER resolution (factor 2 vs the bank's 4K). On **GPU_V4**
the stacked model is marginally better on LPIPS (−1.3%) but ~2.3 dB LOWER on PSNR/SSIM.
**That PSNR gap is a recipe confound, NOT a regression from the levers:** the WP-S1 *control*
(same production recipe, no levers) already sits 2.2 dB below the GPU bank (23.04 vs 25.28),
so the gap is caused by recipe differences (the Jun-14 GPU bank was cap-2M and almost
certainly used little/no gated-LPIPS → higher PSNR, worse LPIPS; the current production
recipe trades PSNR for perceptual sharpness via gated-LPIPS 0.2). The clean, apples-to-apples
result is the internal stack-vs-control A/B above: **adding the four levers to the current
production recipe makes both of the user's real captures perceptually better and
geometrically much cleaner** — modestly, because these captures are already well-sampled.
The flat/shiny surfaces improve slightly but their specular reflections stay broken.

---

## Persisted artifacts (committable)
- `_infusion/wps1_RECORD.md` (this file)
- `qa/wps1_laptop_ctrl/`, `qa/wps1_laptop_stack/`, `qa/wps1_gpu_ctrl/`, `qa/wps1_gpu_stack/`
  (full 6-scorer scorecards + report.html)
- `_infusion/wps1_laptop_region.json`, `_infusion/wps1_gpu_region.json` (region + full LPIPS/PSNR, exact boxes)
- `_infusion/wps1_gpu_shinybox_ref.jpg` (GPU_V4 shiny-box reference, view48)
- `_infusion/wps1_laptop_floaters.txt`, `_infusion/wps1_gpu_floaters.txt`
- `qa/wps1_laptop_strips/`, `qa/wps1_gpu_strips/` (GT|ctrl|stack full + 2× region strips)
- `_infusion/wps1_region_eval.py` (region metrics + strips), `_infusion/wps1_run.sh`,
  `_infusion/wps1_chain.sh` (exact recipe/argv), `_infusion/wps1_eval.sh` (eval driver)
- `_dataset/GPU_V4/.../wps1/dense_work/dense_init_stats.json` (GPU_V4 dense-init, seed 0;
  `_dataset/` gitignored — recomputable from the argv above)
- ckpts: `_dataset/{Laptop,GPU_V4}/.../wps1/{ctrl,stack}/output/ckpts/ckpt_29999_rank0.pt`
  (gitignored; recomputable from the argv above)

## Deviations / operational notes
- **Recipe cap:** used cap_max **3M** (the validated production recipe: `run_train_laptop.py`,
  DEVELOPER_NOTES §"3M", quality-roadmap "keep cap_max 3M"). The OLD banked GPU_V4 ckpt is
  2M — a documented confound for the bank comparison only; the stack-vs-ctrl A/B is internally
  consistent at 3M.
- **factor 2** for both scenes (task directive; mandatory for the stack since the MoGe prior
  cache is factor-2 and the trainer hard-errors on a factor mismatch). The Laptop bank was
  factor-1 (4K) — confound noted above.
- **Orphan-survival operational finding:** the harness periodically severs background-job
  tracking (emits "killed" ~hourly) but the underlying process tree keeps running as an
  orphan and completes. Early relaunching of what were actually still-running orphans created
  transient duplicate trainers (cleaned up); the final sequence ran to `ALL_DONE` on a single
  surviving orphaned chain (all four runs exit=0). Trainer determinism caveat (WP-Q2/F2): same
  seed pins view order + init but not bit-exact (CUDA atomics).
