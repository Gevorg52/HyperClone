# WP-S2 — Object-mode background study (GPU_V4) — RECORD (run 2026-07-04)

**Verdict (per-mode recommendation, the GATE):**
- **OBJECT scans → KEEP the background at training (full-scene = current production). Do NOT
  switch to object-only.** Object-only training measurably DEGRADES object-region quality
  (obj-LPIPS +24%, obj-PSNR −2.6 dB, obj-SSIM −0.069 vs full-scene), and **alpha-loss does NOT
  recover it** (arm c ≈ arm b on the object).
- **Alpha-loss verdict: NOT required (leave OFF, production default).** Its job — kill
  black-opaque background floaters — is largely MOOT under the production MCMC + opacity_reg
  recipe: object-only WITHOUT alpha already has only **7** black-opaque out-of-hull Gaussians
  (MCMC drives the 421k black-supervised background init points to near-TRANSPARENT, not
  black-opaque). Alpha removes those last 7 (→0) but doesn't help object quality and adds
  SOR-outliers.
- **WORLDS scans → unaffected (full-scene stays; that is what these object captures already do).**
- **Hypothesized regression CONFIRMED (modest):** object-only shows ~33% MORE pose-refinement
  translation drift than full-scene — background texture anchors pose_opt; removing it lets poses
  drift, which (with reduced multi-view photometric constraint) is the most likely cause of the
  object-region quality loss.

**Net:** the queued WP-S2 premise — "production is object-only black-bg; is that optimal / does the
shipped alpha-loss lever help" — is answered: production is NOT object-only (it is full-scene, per
WP-G1), full-scene is BETTER for object-region quality than object-only, and the alpha-loss lever
is unnecessary on this recipe. Object-only + alpha buys a marginally cleaner exterior at the cost of
object fidelity — not worth it.

---

## 0. Data-mode verification FIRST (self-evident what each arm trains on)

Measured the factor-2 frames the trainer actually reads
(`training/gsplat_prod/data/images_2`, 569 frames, 2000×3000). Evidence:
`qa/wps2_datamode.json` + `qa/wps2_datamode_strip.jpg` (4 frames, full + corner crop).

| metric | value | reading |
|---|---|---|
| near-black px (max(RGB)<16/255) | **11.4%** | a visible background, NOT premultiplied black |
| mean corner luminance (0–255) | **89.5** | corners are bright room (wall/furniture) |
| mean frame luminance (0–255) | 97.0 | full scene |
| **DATA_MODE_VERDICT** | **FULL_SCENE** | confirms WP-G1; refutes DEVELOPER_NOTES "black-bg" |

So **arm (a) full-scene = CURRENT PRODUCTION (the control); object-only (b)/(c) is the NEW thing
under test** — the plan's original arm labels were inverted (WP-G1 correction, honoured here). The
strip shows the GPU on a polka-dot cloth in a real room (chair, desk, wood floor, couch, wall).

## 1. Object masks (BiRefNet, headless)

No pre-computed object masks existed (`sam_segments/` empty; SAM 2.1 not installed in the Windows
env). Generated with the project's existing masking engine — **BiRefNet 'general'**
(`core.masking.run_birefnet_batch`, `ZhengPeng7/BiRefNet`, soft alpha, threshold 0, erosion 0) — run
headless on the 569 factor-2 frames → `wps2/masks_2/DSC*.png` (stem+resolution matched to `images_2`;
the trainer resolves `masks_2/` first for factor 2). Evidence: `qa/wps2_masks.json`,
`qa/wps2_mask_overlays/` (image|mask|overlay for 5 views spanning the orbit),
`qa/wps2_masktest/overlay_DSC00730.jpg`.

- coverage: mean object fraction **20.7%** (min 7.1%, max 53.7%) over a 64-frame sample.
- quality (spot-check, 5 spread views): the GPU (3 fans, backplate, PCIe bracket, IO shield) is
  cleanly captured; background (cloth, chair, desk, wall) excluded, across the whole trajectory.

## 2. Arms — exact recipe (WP-S1 STACKED config, 15k, seed 0, cap 1M, factor 2)

Established 15k A/B protocol (WP-F4): **no C2F** (coarse_factor unset → `coarse_until`→0 so
mono-depth fires at step 2000), factor 2 throughout. Dense-init is baked into `sparse/0` (WP-S1 GPU_V4
model, **882,902** pts) — no trainer flag. Driver: `_infusion/wps2_chain.sh`.

**COMMON (all 3 arms):**
```
--data_factor 2 --max_steps 15000 --test_every 8 --seed 0 --sh_degree 3 --ssim_lambda 0.2
--strategy mcmc --cap_max 1000000 --opacity_reg 0.01 --scale_reg 0.01 --antialiased
--normalize_world_space --pose_opt --pose_opt_lr 1e-5
--mono_depth_lambda 0.05 --mono_depth_start 2000 --mono_depth_use_weight
--erank_lambda 0.01 --erank_start 7000
--lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000 --lpips_gated
--save_steps 15000
```
| arm | data_dir | arm-specific flags | trains on | train time |
|---|---|---|---|---|
| **(a) a_full** (control = prod) | `wps1/stack/data` | (none) — full-frame mono-depth, no alpha | FULL-SCENE images_2 | 2436 s |
| **(b) b_obj** | `wps2/obj/data` | `--mono_depth_object_masked` | premult BLACK-bg images_2 (+masks) | 2735 s |
| **(c) c_obj_alpha** | `wps2/obj/data` | `--mono_depth_object_masked --alpha_loss_lambda 0.1` | premult BLACK-bg images_2 (+masks) | 2686 s |

`--alpha_obj_lambda` stayed **0** (default, not passed) in all arms — confirmed from the live
process argv. All arms verified at launch: 882,902 dense-init pts, 497 train / 72 held-out, pose_opt
+ gated-LPIPS + erank 0.01 ON; b/c mono-depth object-masked (569/569 reliable, 497/497 covered).

### New lever added (default OFF): `--mono_depth_object_masked`
Object-mode mono-depth restricted to inside the object mask (`pv &= mask>0.5`) so object-only
training never anchors Gaussians to background depth. `core/training/gsplat_trainer.py` (~4 lines) +
`gsplat_backend.py` plumbing. Default OFF → production (full-frame mono-depth) unchanged. **Priors
decision (justified):** reused the SAME MoGe-2 cache for all arms (no re-run, no offline undistort);
the object-region depth priors are byte-identical to arm (a), so the ONLY difference for b/c is that
background depth supervision is dropped — the correct "object-only" semantics, and a clean isolation
of the full-scene-vs-object variable.

## 3. GATE TABLE 1 — OBJECT-REGION metrics (primary) — all arms vs the SAME full-scene GT object px

`_infusion/wps2_obj_eval.py`: eval_dir junctions FULL-SCENE `images_2` (GT) + object `masks_2` + the
dense sparse; each arm rendered black-bg antialiased on the 72 held-out views; OBJECT-region =
mask-premultiplied object-bbox VGG-LPIPS (normalize=True, fair across arms — bg is black in both
inputs) + pixelwise PSNR/SSIM over the object mask. JSON: `_infusion/wps2_obj_metrics.json`;
strips `qa/wps2_strips/` (GT|a|b|c full + 2× object crop, 8 views).

| metric (object region) | (a) full-scene | (b) obj-only | (c) obj+alpha | Δ b−a | Δ c−a |
|---|---|---|---|---|---|
| **OBJ LPIPS ↓ (PRIMARY)** | **0.2277** | 0.2824 | 0.2826 | **+0.0547 (+24.0%)** | +0.0549 (+24.1%) |
| OBJ PSNR ↑ | **23.505** | 20.866 | 21.044 | **−2.64 dB** | −2.46 dB |
| OBJ SSIM ↑ | **0.7368** | 0.6674 | 0.6695 | −0.0694 | −0.0673 |
| obj frac (of frame) | 19.3% | 19.3% | 19.3% | (same views/masks) | |

**Full-scene (a) wins the object region on all three metrics.** Object-only (b) is +24% worse
LPIPS / −2.6 dB PSNR. **Alpha-loss (c) does NOT recover object quality** (c ≈ b: +0.18 dB PSNR,
identical LPIPS). Visual (`qa/wps2_strips/view27_objcrop2x.jpg`): a renders the edge-on heatsink
fins + gold contacts comparably or slightly sharper; b/c show a faint iridescent haze on the metal
edge. (Full-frame LPIPS/PSNR are NOT comparable across arms — b/c render black backgrounds so their
full-frame numbers reflect black-vs-real-bg, not quality; that is why the OBJECT-scoped metric is the
gate. bg-region PSNR: a=25.1 dB, b/c=5.8 dB by construction.)

## 4. GATE TABLE 2 — black-opaque floaters + interaction stats (the hazard this study exists to measure)

`_infusion/wps2_analysis.py` — object hull from the object-classified dense-init points (normalized
frame; classification by projecting every init point into all 569 views vs the masks, validated in
`qa/wps2_initclass_overlay.jpg`); black-opaque = OUTSIDE object hull AND max(rgb)<0.15 AND
sigmoid(opa)>0.5. JSON: `_infusion/wps2_floater_analysis.json`.

**Interaction stat — dense-init points outside the object mask:** of 882,902 dense-init points,
**421,347 (47.7%) are BACKGROUND** (outside the object across the majority of views) — i.e. ~421k
points that get BLACK-supervised in arms (b)/(c). This is the black-opaque-floater hazard the study
targets.

| per arm (15k ckpt, 1,000,000 GS) | (a) full-scene | (b) obj-only | (c) obj+alpha |
|---|---|---|---|
| Gaussians OUTSIDE object hull | **49.1%** (491k) | **0.5%** (5.1k) | **0.0%** (0) |
| — of which opaque (σ>0.5) | 17,816 | 7 | 0 |
| — **BLACK-OPAQUE (dark+opaque)** | 4,740* | **7** | **0** |
| opacity-mass outside hull | 53.4% | 0.57% | 0.00% |
| mean max-RGB of outside GS | 0.652 (COLORED) | −0.01 (dark) | — |

*(a)'s 491k outside-hull Gaussians are LEGIT COLORED background (mean max-RGB 0.65); its 4,740
"black-opaque" are genuinely dark background regions (black desk/monitor/shadow), not floaters.*

**Finding that reframes the study:** the black-opaque background-floater hazard **barely
materializes** under the production MCMC + opacity_reg recipe. Object-only WITHOUT alpha (b) has only
**7** black-opaque out-of-hull Gaussians — MCMC + opacity_reg drove the 421k black-supervised init
points to **near-TRANSPARENT** instead (see §5: low-opacity fraction 58% and med-opacity 0.008 in b
vs 12% / 0.043 in a). Alpha-loss (c) removes the last 7 (→0), a **marginal** cleanup, not the
essential floater-killer it was hypothesized to be.

## 5. Standard floater indicators (`floater_report.py`) — `_infusion/wps2_floaters.txt`

| arm | needle% | SOR% | giant% | low-opa% | med-opa | mean-erank |
|---|---|---|---|---|---|---|
| a_full | 12.92 | 0.02 | 11.20 | 11.8 | 0.043 | 2.177 |
| b_obj | 7.38 | 0.22 | 1.59 | **58.0** | **0.008** | 2.148 |
| c_obj_alpha | 7.54 | **3.48** | 0.87 | **60.9** | **0.007** | 2.155 |

Object-only has fewer giants/needles (it models less) but **5× more low-opacity ghosts** (the
transparent-ized background points — the MCMC "solution" to a black background). Alpha (c) trades a
few more SOR-outliers (0.22→3.48%) for slightly fewer giants; no meaningful cleanliness win over b.

## 6. Pose-stability observation (hypothesis: background texture anchors pose_opt) — CONFIRMED (modest)

Per-view CameraOptModule deltas (`pose_embeds`), mean over 569 views; scene_scale 1.454.

| arm | trans-norm mean (%scene) | trans p95 (%scene) | rot° mean | rot° p95 |
|---|---|---|---|---|
| a_full | **0.0908** | 0.208 | **0.0783** | 0.177 |
| b_obj | 0.1211 (**+33%**) | 0.267 | 0.0832 (+6%) | 0.202 |
| c_obj_alpha | 0.1176 (**+30%**) | 0.260 | 0.0815 (+4%) | 0.215 |

Object-only arms drift **~33% more in translation** (and modestly more in rotation) than full-scene.
This supports the hypothesis: the background provides photometric correspondences that anchor
pose_opt; masking it out lets the refined poses wander. Combined with the reduced multi-view
constraint on the object, this is the most plausible mechanism for the object-region quality loss in
§3. (Absolute drift is small — 0.0013 vs 0.0018 world units — but the ~33% relative increase is
consistent across seeds-of-one and matches the object-quality regression.)

## 7. Temporal scorers (tsed / raft_warp / flicker), object-framed held-out trajectory
`qa/wps2_{a_full,b_obj,c_obj_alpha}/scorecard.json` (qa_report `--scorers`, in-process, deterministic).

| arm | tsed ratio ↑ | tsed epi-med ↓ | raft-warp mean ↓ | flicker ↑ | obs% |
|---|---|---|---|---|---|
| a_full | 1.000 | 0.132 | 0.0137 | 0.9888 | 100.0 |
| b_obj | 1.000 | 0.108 | 0.0107 | 0.9965 | 55.0 |
| c_obj_alpha | 1.000 | 0.103 | 0.0112 | 0.9963 | 16.7 |

**All arms tsed=1.0** (no epipolar violations → no gross temporal floaters in any arm). b/c score
marginally lower raft/flicker, but that is an **artifact**, not a win: object-only renders a large
static BLACK background (visibility obs% 55%/17% vs a's 100%; c's alpha-loss makes even more of the
frame transparent), and a static black region trivially inflates temporal consistency. These scorers
are therefore not a fair cross-arm quality comparison here; the object-region metrics (§3) are.

## 8. Persisted artifacts (committable; `_dataset/` is gitignored — ckpts recomputable from the argv)
- `_infusion/wps2_RECORD.md` (this file)
- `_infusion/wps2_datamode.py` · `qa/wps2_datamode.json` · `qa/wps2_datamode_strip.jpg`
- `_infusion/wps2_make_masks.py` · `qa/wps2_masks.json` · `qa/wps2_mask_overlays/` · `qa/wps2_masktest/`
- `_infusion/wps2_premultiply.py` · `qa/wps2_initclass_overlay.jpg`
- `_infusion/wps2_chain.sh` (exact 3-arm argv + seed) · `_infusion/wps2_{a_full,b_obj,c_obj_alpha}.log`
- `_infusion/wps2_obj_eval.py` · `_infusion/wps2_obj_metrics.json` · `qa/wps2_strips/`
- `_infusion/wps2_analysis.py` · `_infusion/wps2_floater_analysis.json`
- `_infusion/wps2_floaters.txt` · `qa/wps2_{a_full,b_obj,c_obj_alpha}/scorecard.json`
- `_infusion/wps2_eval.sh` (eval driver) · `_infusion/wps2_wait_eval.sh` (orphan-robust runner)
- code: `core/training/gsplat_trainer.py` + `gsplat_backend.py` (`--mono_depth_object_masked`, default OFF)
- THIRD_PARTY.md: BiRefNet + the new lever logged.

## 9. Deviations / honesty
- **15k / cap-1M / no-C2F** = the established fast A/B protocol (WP-F4), NOT the 30k/cap-3M/C2F
  production run. All three arms use identical settings, so the A/B is internally valid; absolute
  numbers are lower-budget than a production run (payoff signs, not production-final magnitudes).
- **Single scene (GPU_V4)**, single seed. GPU_V4 was chosen because it has a real, partly-dark
  background (the only scene where the black-opaque-floater hazard could plausibly appear). The
  result (full-scene ≥ object-only; alpha unnecessary) is a clean negative for object-only on this
  scene; a second scene would strengthen generality but the mechanism (pose-anchor + MCMC
  transparent-izes bg) is recipe-level, not scene-specific.
- **Object hull** built from the object-classified init points (majority-vote projection, 98th-pct
  outlier-trim, +10% margin). "Outside hull" is thus a slightly generous object region; the
  arm-relative comparison (a≫b>c on outside-hull count) is robust to the exact margin.
- Object-region LPIPS uses mask-premultiplied object-bbox crops (1 forward/view). The earlier
  tiled-over-background variant was pathologically slow (~1000 tiles/view on the annular bg) and
  redundant — replaced; bg is covered by pixelwise PSNR/SSIM + the floater analysis.
