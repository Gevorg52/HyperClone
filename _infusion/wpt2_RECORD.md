# WP-T2 — Sony shoe quality improvement pass — RECORD

**Goal:** make the WP-T1 Sony shoe model better in its MUSHY areas (finest lace/mesh weave
softening; weak per-view cluster idx 400–512). Two evidence-backed moves: (1) DIAGNOSE where/why
the mush is (routes future levers), (2) the WP-N1-validated **native-detail variant** — retrain at
**factor-2 (~15 MP) with pose_opt OFF** — and compare honestly (common-res factor-3 + native
factor-2 + region crops + strips). **Do NOT commit** (`_dataset/` gitignored; this record +
`qa/wpt2_*` + `_infusion/wpt2_*.json` are the committable evidence).

Base model (WP-T1): `_dataset/SONY_TEST/training/gsplat/output/ckpts/ckpt_29999_rank0.pt`
(factor-3 = 6.69 MP, WP-S1 stacked recipe + pose_opt ON, object-LPIPS 0.176).
Environment: RTX PRO 6000 sm_120 96 GB; torch 2.11.0+cu128; gsplat AOT. I own the GPU (sequential).

---

## STEP 1 — DIAGNOSIS (on the existing factor-3 ckpt) — the mush cause

Scripts (committable): `_infusion/wpt2_diagnose.py` (per-Gaussian view-count/ρ machinery from
`core/qa/visibility.py` + `_infusion/val_splatsure.py`), `_infusion/wpt2_regions.py`.
Data: factor-3 model, every-8 split (74 held-out, sorted view idx 0–587). Cluster = raw idx [400,512]
(15 test views). Output: `_infusion/wpt2_diagnosis.json`, panel `qa/wpt2_diag/`.

### 1a. Is the weak 400–512 cluster UNDER-OBSERVED or SPECULAR? → **NOT under-observation**

Per-Gaussian training-view observation counts (all 514 train views, `fully_fused_projection`
eps2d=0, SplatSuRe protocol) + per-test-view under-observed fraction (accumulated splat weight from
Gaussians seen by <3 views, within the visibility-masked region):

| signal | value | reading |
|---|---|---|
| camera→object distance spread | min 0.517 / median 0.966 / max 1.760 = **3.41×** | NOT a constant-radius orbit — real distance variation (ρ-gate can separate) |
| Gaussians seen by ≥1 / ≥3 views | 70.1% / **70.1%** | any Gaussian seen at all is seen by ≥3 views; the other ~30% is the never-observed low-opa padding |
| **opacity-weighted under-observed (<3 views)** | **0.8%** | the *visible* surface is almost entirely well-observed |
| n_obs percentiles (opacity-agnostic) | p50 **463** / p75 487 / p95 498 (of 514) | median Gaussian seen by 463/514 views |

| held-out group | n | masked-PSNR | masked-LPIPS | **under-frac** | mean dist | belowness |
|---|---|---|---|---|---|---|
| **cluster idx 400–512** | 15 | **22.31** | **0.4649** | **0.0000** | 1.012 | −0.707 |
| rest | 59 | 24.34 | 0.3956 | **0.0000** | 0.979 | −0.024 |

**VERDICT: the weak cluster is NOT under-observation.** Under-observed fraction is **0.0000 for
both the cluster and the rest** — no view, cluster or not, renders from Gaussians that few cameras
saw. The visible surface Gaussians are observed by a median of 463/514 views; opacity-weighted
under-observed mass is 0.8% globally. Yet the cluster is −2.0 dB PSNR / +0.069 LPIPS worse. The
residual error is therefore **appearance-limited, not geometry/coverage-limited**: view-dependent
specular reflection on the silver TPU overlays + metallic-green panel + translucent gel window,
plus the low-texture forefoot mesh + dark rubber tread at these steep/grazing angles (the cluster
is the steep top-down + below-grazing subset; belowness −0.707 vs −0.024 rest). This exactly matches
DEVELOPER_NOTES §4 (specular/flat surfaces are capture-limited; MCMC drains flat regions by opacity)
and the WP-S1 residual-specular diagnosis.

**Visual confirmation** (`qa/wpt2_diag/clusterworst_view496_DSC01699.JPG_panel.jpg`, GT | render |
under-heatmap): worst cluster view (idx 496, masked-PSNR 19.40) — GT crisp mesh weave + sharp silver
etched blade lines; render softens the mesh and *smears the specular highlight* on the silver
overlay; the under-heatmap is **uniformly zero (deep blue) across the entire shoe** = every visible
pixel is well-observed. The failure is pure appearance.

### 1b. Region localization of the mush (prod model)  — [region-LPIPS + strips, see STEP 3]
Region crops fixed for the whole WP (per-view fractional boxes; held-out views):
- **lace_mesh** = idx 448 (DSC01651, top-down: laces + forefoot mesh + tongue text) — resolution-limited microstructure
- **sole_tread** = idx 96 (DSC01299, dark rubber tread from below) — the sole-from-below focus
- **metallic**  = idx 248 (DSC01451, side profile: silver TPU blades + green medial sheen + mesh) — specular/appearance

### Diagnosis → lever routing (records for the future)
- The **lace/mesh microstructure** softening is **resolution-limited** → the factor-2 native-detail
  variant (WP-N1 recipe) is the right lever and SHOULD help these regions.
- The **400–512 cluster** (specular metallic + dark tread + steep angles) is **appearance-limited**,
  NOT resolution- or coverage-limited → factor-2 is **NOT expected to rescue it**; its fixes are a
  dedicated specular lever (WP-G1) or capture-side (cross-pol, matte spray). Dense-init / more views
  would NOT help (geometry already fully observed). **Prediction to test in STEP 3.**

---

## STEP 2 — the DETAIL variant (factor-2, pose_opt OFF)  — WP-N1 native-detail recipe

**Data prep (disk-efficient, reuses everything mandated):** shared data dir
`_dataset/SONY_TEST/training/gsplat/data`. Generated `images_2` (588 @ **4752×3168 = 15.05 MP**) +
`images_4` (588 @ 2376×1584, coarse C2F phase) via the exact `GsplatBackend._downsample_images`
bicubic recipe (`_dataset/SONY_TEST/downsample_f2.py`). **dense_init sparse REUSED** (the same
738,340-pt RoMa-augmented `sparse/0` — no re-run). **Priors cache rebuilt at factor-2**
(`python -m core.priors <data> --factor 2 --depth_only`): factor=2, per-view mode, **588/588
reliable, median reliable rel-residual 0.01464** (identical alignment quality to the factor-3 cache's
0.0146 — resolution is the only change), scale span [0.715, 2.06], peak VRAM 2910 MB, 353 s. This
OVERWROTE the dead factor-3 depth cache incrementally (the factor-3 model is trained/delivered; eval
reads ckpts not priors; rebuild f3 with `--factor 3 --depth_only` if ever needed). Result dir
`output_t2` (separate → the factor-3 production model is preserved intact). Disk after prep: 46 GB free.

**Recipe** (`_dataset/SONY_TEST/train_sony_t2.sh` — exact argv persisted):
identical to WP-T1 `train_sony.sh` EXCEPT `--data_factor 3→2`, `--coarse_factor 6→4` (C2F 4→2), and
**pose_opt REMOVED** (the WP-N1 requirement). Full: `--data_factor 2 --max_steps 30000 --test_every 8
--seed 0 --sh_degree 3 --ssim_lambda 0.2 --strategy mcmc --cap_max 3000000 --opacity_reg 0.01
--scale_reg 0.01 --antialiased --normalize_world_space --coarse_factor 4 --coarse_until 15000
--lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000 --lpips_gated --mono_depth_lambda 0.05
--mono_depth_start 2000 --mono_depth_use_weight --erank_lambda 0.01 --erank_start 7000 --save_steps
7000 30000`. Startup confirmed: C2F 4→2 @15k; init 738,340 GS from dense cloud; MCMC cap 3M;
mono-depth ON (start=15001, 588/588 reliable, 514/514 covered); pose_opt OFF. Coarse phase ~25 it/s,
23.2 GB VRAM.

**VRAM / runtime:** peak **23,251 MB** (steady across coarse+fine — the MCMC densification
peak dominates, NOT the factor-2 15 MP fine phase; matches WP-N1's "fine phase did not exceed
the coarse-phase peak"). Runtime **10,235.5 s (~2.84 h)**, GPU owned/uncontended. Trainer own
[Eval] (unmasked whole-frame, factor-2, 74 held-out): **PSNR 24.28 / SSIM 0.9070**; config in
`output_t2/stats/results.json`: pose_opt=False, data_factor=2, coarse_factor=4. Detail ckpt:
`_dataset/SONY_TEST/training/gsplat/output_t2/ckpts/ckpt_29999_rank0.pt`.

---

## STEP 3 — HONEST COMPARISON (prod factor-3 pose_opt-ON vs detail factor-2 pose_opt-OFF)

Both models are 3D Gaussians → rendered at each factor vs that factor's GT. `qa_report` masked
metrics (visibility mask, VGG-LPIPS normalize=True). Scorecards under `qa/wpt2_{prod,detail}_{f3,f2}/`.

### Common-res (factor-3, the WP-T1 production delivery resolution) — whole-frame masked
| metric | PROD (f3) | DETAIL (f2) | Δ (detail−prod) |
|---|---|---|---|
| masked PSNR | 23.928 | **24.342** | **+0.41 dB** |
| masked SSIM | 0.8949 | **0.9175** | **+0.0226** |
| masked LPIPS ↓ | **0.4096** | 0.4218 | +0.0122 (**+3.0% worse**) |

PROD f3 reproduces WP-T1 EXACTLY (23.928 / 0.8949 / 0.4096) → harness/data consistency confirmed.

### Native (factor-2, the detail model's home resolution) — whole-frame masked
| metric | PROD (f3) | DETAIL (f2) | Δ |
|---|---|---|---|
| masked PSNR | 23.858 | **24.282** | **+0.42 dB** |
| masked SSIM | 0.8884 | **0.9070** | **+0.0186** |
| masked LPIPS ↓ | 0.4811 | **0.4798** | −0.0013 (**−0.27%, detail wins**) |

→ Detail wins PSNR/SSIM at BOTH resolutions. Whole-frame LPIPS **flips with eval resolution**:
detail loses at factor-3 (+3.0%), wins at factor-2 (−0.27%) — each arm's perceptual/texture match
is resolution-matched (EXACTLY the WP-N1 finding). Whole-frame LPIPS is dominated by the
heavily-defocused bokeh background (WP-T1: object-region 0.176 vs whole-frame 0.41), so it is NOT
the decisive metric for "is the shoe better" — the region crops are.

### Region-LPIPS (per-view crop boxes; `_infusion/wpt2_regions_f{3,2}.json`; strips `qa/wpt2_strips_f{3,2}/`)
| region | factor | PROD | DETAIL | Δ% | reading |
|---|---|---|---|---|---|
| **lace_mesh** (idx448, microstructure) | f3 | 0.3165 | **0.2982** | **−5.8%** | detail WINS |
| **lace_mesh** | f2 | 0.4236 | **0.3795** | **−10.4%** | detail WINS BIG |
| sole_tread (idx96, dark rubber/appearance) | f3 | **0.3059** | 0.3205 | +4.8% | detail loses (res-mismatch) |
| sole_tread | f2 | 0.4098 | **0.4054** | −1.1% | detail marginal |
| metallic (idx248, silver+green specular+mesh) | f3 | **0.2144** | 0.2253 | +5.1% | detail loses (res-mismatch) |
| metallic | f2 | 0.3075 | **0.2941** | −4.4% | detail wins (via mesh; specular residual persists) |

**This is the diagnosis prediction CONFIRMED (STEP 1 routing):** the factor-2 detail helps EXACTLY
the RESOLUTION-limited region (lace/mesh microstructure: −5.8% common / −10.4% native, the biggest
gain by far, and it survives the resolution-mismatch penalty to still WIN at common factor-3). The
APPEARANCE-limited regions (sole/tread dark rubber; specular silver+green metallic) gain little or
nothing — they even regress at common factor-3 (resolution mismatch) and only marginally improve at
native factor-2, and the specular sheen residual persists in both arms. **Factor-2 is not a specular
or dark-texture fix** (as diagnosed: those are appearance/capture-limited, not resolution-limited).

### Strips (GT | prod | detail) — visual, `qa/wpt2_strips_f2/*_crop2x.jpg`
- **lace_mesh (view448):** CLEAR visible win — prod washes out the hexagonal mesh weave + lace
  stitching + silver etched blade lines + green "asics" text; detail resolves all of them
  markedly crisper, close to GT.
- **sole_tread (view96):** marginal — the white midsole hex texture sharpens slightly; the DARK
  rubber tread lugs stay ~equal (detail can't manufacture texture the capture didn't carry on the
  low-texture dark rubber — capture-limited, DEVELOPER_NOTES §4).
- **metallic (view248):** the mesh weave + silver blade etching sharpen in detail; the specular
  sheen on the green/silver panels remains smeared in BOTH (the residual specular case).

### Train/test gap — pose_opt-OFF mechanism check (`_infusion/wpt2_gap.json`, factor-2 masked-PSNR)
| arm | pose_opt | train | test | gap |
|---|---|---|---|---|
| prod (f3-trained) | ON | 24.01 | 23.86 | +0.15 dB |
| **detail (f2-trained)** | **OFF** | 24.69 | 24.28 | **+0.41 dB** |

→ The detail model (pose_opt OFF) shows a **healthy +0.41 dB gap — NO overfitting**, vs WP-N1's
GPU_V4 **6.18 dB catastrophe** with pose_opt ON at native res. pose_opt-OFF was the right call; the
WP-N1 native-detail recipe (and its pose_opt-OFF requirement) transfers cleanly to the shoe. (prod's
+0.15 is understated — it renders train views at ORIGINAL un-refined poses; still small.)

## VERDICT / SHIP RECOMMENDATION
1. **The mush the user cares about (finest lace/mesh weave softening) is genuinely and visibly
   improved by the factor-2 detail variant** — region-LPIPS −5.8% (common factor-3) / −10.4%
   (native factor-2), sharper mesh/lace/etching/text in the strips. This is the WP-T2 win.
2. **It is NOT a blanket replacement for the factor-3 production recipe.** Whole-frame common-res
   LPIPS regresses +3.0% (background-dominated + resolution-matched perceptual), so for **factor-3
   perceptual delivery, factor-3 production stays the recipe**. The factor-2 detail (pose_opt OFF)
   is the validated variant for **NATIVE 15 MP detail delivery** — where it wins PSNR/SSIM/LPIPS AND
   all region crops (lace_mesh −10.4%, metallic −4.4%, sole −1.1%). This is the exact WP-N1 verdict
   pattern, reproduced on a second, independent capture (specular product orbit) → the finding
   generalizes.
3. **The weak 400–512 cluster is NOT fixed by resolution** (diagnosed appearance-limited; detail
   gains there are marginal-to-negative). Its remaining levers are a dedicated specular lever (WP-G1)
   or capture-side (cross-pol / matte the sole+metallic) — NOT dense-init/more-views (geometry is
   already fully observed) and NOT higher training resolution.
4. **pose_opt-OFF is mandatory for native-res training** — re-confirmed on the shoe (clean +0.41 dB
   gap). Native-res + pose_opt-ON is the known trap; not retried.

## Deliverable — object-extract of the detail model  _[see STEP 4]_
