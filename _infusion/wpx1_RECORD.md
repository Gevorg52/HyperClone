# WP-X1 — Master-splat re-test (>3M) on GPU_V4

Executes EXECUTION_PLAN §WP-X1: re-run cap_max 3M/4M/6M/8M on GPU_V4 **with** the
WP-Q2a erank anti-needle hinge + raised scale_reg (0.02) + coarse-to-fine, on the
96 GB RTX PRO 6000, to test whether the historical 8M divergence (DEVELOPER_NOTES §6:
PSNR 23→10 collapse past 3M, giant-Gaussian tile-intersection blowup, 33 GB OOM on the
24 GB card) is fixed by the anti-needle/scale-bound attack — i.e. whether an
unconstrained master splat >3M now beats the 3M control on the QA suite, OR the 3M
ceiling is re-confirmed with the regs in place.

## Environment / provenance (verified 2026-07-02)
- Machine: RTX PRO 6000 Blackwell, sm_120, 96 GB, driver 576.57. Windows system
  Python 3.11.0, torch 2.11.0+cu128, gsplat 1.5.3 (AOT sm_120). Trainer at git HEAD
  + uncommitted WP-F1 mono-depth flags (left OFF/default). Trainer code NOT edited.
- Data (prepared, factor 2): `_dataset/GPU_V4/apex_images/training/gsplat_prod/data`
  — 569 images (497 train / 72 test @ test_every 8), 123,289 SfM points,
  scene_scale 1.454. `images_2` (train res) + `images_4` (coarse res) present.
- Baseline for context — GPU_V4 **production 2M** ckpt (`gsplat_prod/output`,
  no erank, scale_reg 0.01): masked PSNR **25.278** / SSIM **0.8514** / LPIPS
  **0.4125** (72 views, observed 99.999%), banked `qa/GPU_V4/scorecard.json`.
- WP-Q2a erank A/B (15k, cap 1M) for context: needle 89.63%→16.59%, mean-erank
  1.53→2.05 at ~equal quality (`training/wpq2_ab/AB_RECORD.md`).

## Recipe (shared across all arms) — the GPU_V4 production recipe + WP-X1 regs
factor 2 · mcmc · antialiased (Mip-Splatting) · pose_opt lr 1e-5 · coarse-to-fine
(coarse_factor 4, coarse_until 15000) · 30k steps · opacity_reg 0.01 · ssim_lambda
0.2 · sh_degree 3 · normalize_world_space · **--seed 0** · save_steps 7000/15000/30000.
WP-X1 regs on every arm: **--erank_lambda 0.01 --erank_start 7000 --scale_reg 0.02**.
LPIPS OFF (production GPU_V4 run had no LPIPS; keeps the cap effect isolated & the
comparison honest — judge by masked-LPIPS + floaters + VRAM per the plan).

Arms (differ ONLY in cap_max):
- **Arm 0 (control)**: `--cap_max 3000000`
- **Arm 1**: `--cap_max 4000000`
- **Arm 2**: `--cap_max 6000000`
- **Arm 3**: `--cap_max 8000000`

Full per-arm argv + seed persisted to `training/wpx1/<tag>/argv.txt` at launch time
— but `_dataset/` is gitignored, so the canonical committable copy is HERE (gate
evidence rule). All four arms ran EXACTLY this argv, differing only in `--cap_max`
(3000000 / 4000000 / 6000000 / 8000000) and the `--result_dir` tag (cap3m/4m/6m/8m):

```
python core/training/gsplat_trainer.py
  --data_dir  _dataset/GPU_V4/apex_images/training/gsplat_prod/data
  --result_dir _dataset/GPU_V4/apex_images/training/wpx1/<tag>/output
  --data_factor 2  --max_steps 30000  --test_every 8  --sh_degree 3
  --ssim_lambda 0.2  --strategy mcmc  --cap_max <CAP>
  --opacity_reg 0.01  --scale_reg 0.02
  --erank_lambda 0.01  --erank_start 7000
  --antialiased  --pose_opt  --pose_opt_lr 1e-5
  --coarse_factor 4  --coarse_until 15000
  --normalize_world_space  --seed 0
  --save_steps 7000 15000 30000
```
(Interpreter: system `python` = Python 3.11.0 / torch 2.11.0+cu128. All other trainer
flags at their argparse defaults; WP-F1 mono-depth flags at default 0/OFF. Arms run
STRICTLY sequentially, GPU verified idle (<1.7 GB desktop, 0% util) before each launch.)

## DEVIATION — MCMC noise_lr (plan asked to lower it 10×)
The plan says "lowered MCMC noise_lr". **The trainer does NOT expose noise_lr**:
`MCMCStrategy(cap_max=…, refine_stop_iter=…, verbose=False)` is constructed with no
`noise_lr` arg and there is no `--noise_lr` CLI flag (verified: `grep noise_lr
core/training/gsplat_trainer.py` + `MCMCStrategy` signature `noise_lr: float =
500000.0`). This WP is "pure experiments — do not edit trainer code AT ALL", so per
the task's own fallback ("if not exposed, note it and proceed") **all arms use the
gsplat default noise_lr = 5e5**. Recorded as a deviation; the erank hinge + scale_reg
0.02 remain the active anti-divergence levers being tested.

## Results

### Arm 0 — cap 3M (control) — COMPLETE 2026-07-02, no divergence
- Ran to 30k, exit 0. Wall time **1571.2 s (26.2 min)**. Count reached **3,000,000**
  (cap hit during coarse phase). Peak `torch.cuda.max_memory_allocated` **5043 MB**
  (trainer `mem=` log; coarse phase 3890 MB, fine phase 5043 MB).
- Trainer eval (unmasked, 72 views): PSNR 25.05 / SSIM 0.8495
  (`training/wpx1/cap3m/output/stats/results.json`).
- QA harness (`qa/wpx1_cap3m/scorecard.json`, ckpt_29999, factor 2, every8, 72 views):
  masked PSNR **25.052** / SSIM **0.8495** / LPIPS **0.4101**, observed 100.0%.
  Scorers: TSED ratio **1.000** (median epi 0.155 px, 452 mean matches),
  RAFT warp-err mean **0.01437** (valid frac 0.978), flicker **0.9887** (static clip).
- Floaters (`qa/wpx1_cap3m/floater.txt`): needle **14.89%**, SOR-outlier **0.09%**,
  giant **8.99%**, low-opa 30.6%, mean-erank **2.085**. The erank hinge holds at 3M
  (WP-Q2 reference at cap 1M: needle 16.59% ON / 89.63% OFF).
- Context vs production 2M ckpt (no erank, scale_reg 0.01, same 30k C2F recipe):
  masked 25.278/0.8514/0.4125 → control trades −0.23 dB PSNR for −0.0024 LPIPS
  with the new regs at 3M. Cross-cap comparison below is like-for-like (all arms
  share the identical recipe + seed 0; only cap_max differs).

### Arm 1 — cap 4M — COMPLETE 2026-07-02, no divergence
- Ran to 30k, exit 0. Wall time **1956.9 s (32.6 min)**. Count reached **4,000,000**.
  Peak `torch.cuda.max_memory_allocated` **6404 MB**; whole-GPU peak (nvidia-smi
  30 s poll incl. desktop ~0.8–1.6 GB + allocator reserve + transient raster
  workspace) **29,168 MiB** (`training/wpx1/cap4m/vram_poll.csv`).
- Trainer eval: PSNR 24.85 / SSIM 0.8478 (`training/wpx1/cap4m/output/stats/results.json`).
- QA harness (`qa/wpx1_cap4m/scorecard.json`, ckpt_29999, 72 views): masked PSNR
  **24.855** / SSIM **0.8478** / LPIPS **0.4105**, observed 100.0%. Scorers: TSED
  ratio **1.000** (median epi 0.155 px), RAFT warp-err mean **0.01424**, flicker
  **0.9887**.
- Floaters (`qa/wpx1_cap4m/floater.txt`): needle **14.05%**, SOR-outlier **0.10%**,
  giant **9.21%**, low-opa 36.1%, mean-erank **2.094**.
- vs 3M control: LPIPS +0.0004 (equal), PSNR −0.20 dB, floaters equal — the extra
  1M Gaussians neither diverge nor help; mild dilution (mirrors the v3 distill
  finding at 4M).

### Arm 2 — cap 6M — COMPLETE 2026-07-02, no divergence
- Ran to 30k, exit 0. Wall time **2358.8 s (39.3 min)**. Count reached **6,000,000**.
  Peak `torch.cuda.max_memory_allocated` **9363 MB**; whole-GPU nvidia-smi poll peak
  **19,520 MiB** (`training/wpx1/cap6m/vram_poll.csv`).
- Trainer eval: PSNR 24.03 / SSIM 0.8384 (`training/wpx1/cap6m/output/stats/results.json`).
- QA harness (`qa/wpx1_cap6m/scorecard.json`, ckpt_29999, 72 views): masked PSNR
  **24.027** / SSIM **0.8384** / LPIPS **0.4166**, observed 100.0%. Scorers: TSED
  ratio **1.000** (median epi 0.163 px), RAFT warp-err mean **0.014376**, flicker
  **0.9886**.
- Floaters (`qa/wpx1_cap6m/floater.txt`): needle **12.03%**, SOR-outlier **0.07%**,
  giant **9.39%**, low-opa **46.3%** (dead fraction growing with cap), mean-erank 2.111.
- vs 3M control: LPIPS +0.0065 worse, PSNR −1.03 dB, SSIM −0.011 — clean (no
  divergence, floaters flat) but clearly DILUTED: the extra Gaussians under-train
  and nearly half are near-dead (low-opa 46.3% vs 30.6% at 3M).

### VRAM measurement note
The nvidia-smi whole-GPU number includes desktop (~0.8–1.6 GB) + CUDA context +
the caching allocator's reserved-but-free pages; cap4m showed a sustained ~27–29 GB
reserve (fragmentation spike in its fine phase) while its live-tensor peak was only
6.4 GB. The load-bearing, cap-comparable measure is the trainer-logged
`torch.cuda.max_memory_allocated` (live tensors), which scales sanely with cap:
5.0 GB (3M) → 6.4 GB (4M) → 9.4 GB (6M). Both are recorded per arm.

### Arm 3 — cap 8M (the historical diverger) — COMPLETE 2026-07-02, NO divergence, NO OOM
- Ran to 30k, exit 0. Wall time **2949.2 s (49.2 min)**. Count reached **8,000,000**.
  Peak `torch.cuda.max_memory_allocated` **13,221 MB** (vs 33 GB OOM on the 24 GB
  card in 2026-06); whole-GPU nvidia-smi poll peak **38,821 MiB**
  (`training/wpx1/cap8m/vram_poll.csv`).
- **Trajectory (the interesting result — full log `training/wpx1/cap8m/train.log`):**
  train-view PSNR peaked **25.27 @ step 8k** (4.8M GS), then FELL to **~21–22**
  as the count grew 4.8M→8M and stayed there through steps 10k–25k (loss 0.067→0.084,
  sagging — the softened signature of the historical 23→10 collapse, WITHOUT the
  collapse), then **RECOVERED +3.5 dB to 25.30 by step 30k** immediately after
  `refine_stop_iter=25000` stopped MCMC noise/relocation. This pins the residual
  8M harm on **MCMC noise injection at high caps** — precisely the lever the plan
  wanted lowered (noise_lr), which the trainer doesn't expose (see DEVIATION above).
- Trainer eval: PSNR 21.16 / SSIM 0.8025 (`training/wpx1/cap8m/output/stats/results.json`).
- QA harness (`qa/wpx1_cap8m/scorecard.json`, ckpt_29999, 72 views): masked PSNR
  **21.161** / SSIM **0.8025** / LPIPS **0.4503**, observed 100.0%. Scorers: TSED
  ratio **1.000** (median epi 0.151 px), RAFT warp-err mean **0.017935** (worst arm),
  flicker **0.9884**.
- Floaters (`qa/wpx1_cap8m/floater.txt`): needle **8.86%** (LOWEST of all arms),
  SOR-outlier **0.04%** (lowest), giant **8.37%** (lowest), **low-opa 63.2%**
  (nearly 2/3 of the model is near-dead), mean-erank 2.144.

### Eval-tooling note (honesty)
First eval pass of Arm 0 accidentally scored **ckpt_6999** (a `sort -t_` full-path
bug in the eval wrapper; fixed to sort by basename step number). The 7k-step numbers
(masked 24.10/0.8401/0.4409, needle 82.92% — pre-erank-start) were discarded;
`qa/wpx1_cap3m/scorecard.json` now records ckpt_29999 (3.0M GS) — verify via its
`ckpt` field.

---

## FINAL TABLE (all: ckpt_29999, factor-2 held-out every8, 72 views; GPU_V4)

| metric | **3M (control)** | 4M | 6M | 8M |
|---|---|---|---|---|
| count reached | 3,000,000 | 4,000,000 | 6,000,000 | 8,000,000 |
| wall time | 1571 s | 1957 s | 2359 s | 2949 s |
| peak VRAM (torch alloc) | 5043 MB | 6404 MB | 9363 MB | 13,221 MB |
| peak VRAM (nvidia-smi whole-GPU) | n/p | 29,168 MiB* | 19,520 MiB | 38,821 MiB |
| divergence events | none | none | none | none (train-PSNR sag 15k–25k, +3.5 dB recovery after noise stop; see Arm 3) |
| masked PSNR ↑ | **25.052** | 24.855 | 24.027 | 21.161 |
| masked SSIM ↑ | **0.8495** | 0.8478 | 0.8384 | 0.8025 |
| masked LPIPS ↓ | **0.4101** | 0.4105 | 0.4166 | 0.4503 |
| TSED ratio ↑ | 1.000 | 1.000 | 1.000 | 1.000 |
| RAFT warp-err ↓ | 0.01437 | **0.01424** | 0.01438 | 0.01793 |
| flicker ↑ | 0.98868 | 0.98865 | 0.98860 | 0.98841 |
| needle (erank<2) ↓ | 14.89% | 14.05% | 12.03% | **8.86%** |
| SOR-outlier ↓ | 0.09% | 0.10% | 0.07% | **0.04%** |
| giant-blob ↓ | 8.99% | 9.21% | 9.39% | **8.37%** |
| low-opacity (<0.01) | 30.6% | 36.1% | 46.3% | 63.2% |
| mean erank | 2.085 | 2.094 | 2.111 | 2.144 |

\* cap4m's whole-GPU figure is sustained allocator reserve after a fine-phase
fragmentation spike (live tensors only 6.4 GB) — see VRAM measurement note.

Scorecards: `qa/wpx1_cap{3,4,6,8}m/scorecard.json` (+ `floater.txt` each).
Visual: `_infusion/wpx1_zoom/sags_z2_view{0,12,24}.jpg` — 2× zoom strips, all four
arms [OFF|ON eps2d rule]; every cap is CLEAN (no needle streaks / floaters / blobs;
the erank hinge holds at all caps), but 6M/8M are visibly SOFTER (text + PCB traces).

## VERDICT — the 3M ceiling is RE-CONFIRMED; the divergence mechanism is FIXED

1. **Gate outcome (per plan §WP-X1): no cap >3M beats the 3M control** on the QA
   suite at fixed eval res. Quality declines monotonically with cap: masked LPIPS
   0.4101 → 0.4105 → 0.4166 → 0.4503; masked PSNR −0.20/−1.03/−3.89 dB at 4M/6M/8M.
   4M is a statistical tie on LPIPS/floaters (mild dilution); 6M/8M are clear losses.
2. **The 2026-06 divergence mechanism IS fixed by erank+scale_reg.** 8M trained to
   completion with no PSNR 23→10 collapse, no OOM (13.2 GB live vs 33 GB then), and
   the FEWEST needles/outliers/giants of any arm. The failure mode changed from
   *catastrophic* (giant-Gaussian tile blowup) to *benign dilution* (fixed photon
   budget spread over more, under-trained, increasingly dead Gaussians —
   low-opa 30.6%→63.2%).
3. **The residual 8M quality loss is MCMC-noise-driven, not tile-blowup-driven**:
   the Arm-3 train-PSNR sag disappears (+3.5 dB in 5k steps) the moment noise
   injection stops at refine_stop=25000. A lowered `noise_lr` (the plan's third
   lever, not exposed by the trainer) is the plausible fix if a >3M master splat is
   ever needed — that is a NEW, cheap, one-flag experiment for a future WP, not a
   re-run of this one.
4. **Master-splat implication (Hyperscape lesson):** an unconstrained >3M master for
   delivery-decimation is now SAFE to train (nothing explodes, VRAM trivial on 96 GB)
   but currently pointless on this scene — it delivers a cleaner-but-softer model.
   Use 3M as the master; revisit >3M only after a noise_lr flag exists, or on a
   scene whose capture actually starves 3M.

WP-X1 CLOSED (either-outcome gate satisfied: ceiling re-confirmed, honestly recorded).
Conclusion appended to `docs/DEVELOPER_NOTES.md` §6.
