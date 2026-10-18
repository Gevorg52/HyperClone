# WP-SH4 Phase B Record — SH Degree 4 vs Degree 3 Quality A/B (2026-10-18)

## 1. Executive Summary & Gate Decision

**Verdict:** **KEEP `--sh_degree 3` AS PRODUCTION DEFAULT; KEEP `--sh_degree 4` OPT-IN (WARM-START ONLY)** (`verdict = KEEP_SH3_DEFAULT_OPT_IN_SH4_WARM_START_ONLY`).

Using contiguous 38° azimuth orbit-arc holdout sets (`_infusion/wpsh4/phaseB/gpu_holdout_arc.txt`: 60/569 views = 10.54%; `laptop_holdout_arc.txt`: 35/304 views = 11.51%) to force true angular extrapolation of the 9 degree-4 ($L=4$) basis functions across all elevation rings, we executed:
- **Experiment B1 (Warm-Start Fine-Phase Refinement, 5,000 steps at factor 2, `--refine_stop_iter 0`, `--opacity_reg 0.0`, `--scale_reg 0.01`, `--seed 0`):** Resumed the converged 30k SH3 checkpoints on `GPU_V4` (`wpf5/base`) and `Laptop` (`opreg0/or0`) at `--sh_degree 3` vs `--sh_degree 4`.
- **Experiment B2 (From-Scratch Arc-Withheld A/B on `GPU_V4`, 15,000 steps C2F, `--coarse_factor 4 --coarse_until 7500 --data_factor 2 --cap_max 2000000 --opacity_reg 0.0 --scale_reg 0.01 --sh_degree_interval 1000 --seed 0`):** Trained from scratch at `--sh_degree 3` vs `--sh_degree 4`.

### Summary Against the 4 Promotion Gate Criteria

| Criterion | Requirement | `B1_GPU_V4` (Warm 5k) | `B1_Laptop` (Warm 5k) | `B2_GPU_V4` (Scratch 15k) | Gate Status |
|---|---|---|---|---|---|
| **1. Specular Quality** | $\Delta\text{PSNR}_{\text{spec10}} \ge +0.30\text{ dB}$ **or** $\Delta\text{LPIPS}_{\text{box}} \le -0.008$ | `+0.083 dB` (`+0.203 dB` strict unseen) / `+0.0014` LPIPS | `-0.064 dB` / `-0.0004` LPIPS | `+0.224 dB` / `+0.0074` LPIPS | **FAIL** (peaks at `+0.224 dB`, short of `+0.30 dB`) |
| **2. Held-Out Non-Regression** | $\Delta\text{PSNR}_{\text{masked}} \ge -0.05\text{ dB}$ **and** $\Delta\text{LPIPS}_{\text{masked}} \le +0.002$ | `+0.030 dB` (`+0.080 dB` full) / `-0.00005` LPIPS (**PASS**) | **`-0.060 dB`** (`-0.102 dB` full) / `-0.0003` LPIPS (**FAIL**) | **`-0.152 dB`** / **`+0.0033`** LPIPS (**FAIL**) | **FAIL** (`Laptop` B1 and `GPU_V4` B2 regress) |
| **3. Angular Extrapolation Stability** | $\Delta(\text{train} - \text{holdout}) \le +0.25\text{ dB}$ | `+0.058 dB` (**PASS**) | **`+0.259 dB`** (`+0.476 dB` strict unseen) (**FAIL**) | `+0.185 dB` (**PASS**) | **FAIL** (`Laptop` B1 overfits across the 38° gap) |
| **4. Postshot Truncated Twin (`truncate_sh_ply.py`)** | $\text{PSNR}_{\text{trunc3}} - \text{PSNR}_{\text{SH3}} \ge -0.15\text{ dB}$ | **`+0.058 dB`** (`+0.097 dB` full) (**PASS**) | **`+0.072 dB`** (`+0.088 dB` full) (**PASS**) | **`-0.500 dB`** masked (`-0.521 dB` spec) (**FAIL**) | **PASS for B1 Warm-Start / FAIL for B2 Scratch** |

---

## 2. Quantitative Results (`_infusion/wpsh4/phaseB_gate.json`)

### 2.1 Experiment B1 — Warm-Start Fine-Phase Refinement (5,000 steps from 30k SH3)

| Metric | `B1_GPU_V4` SH3 | `B1_GPU_V4` SH4 | $\Delta$ (`SH4 - SH3`) | `B1_Laptop` SH3 | `B1_Laptop` SH4 | $\Delta$ (`SH4 - SH3`) |
|---|---:|---:|---:|---:|---:|---:|
| **Holdout Arc Views** | 60 (10.54%) | 60 (10.54%) | — | 35 (11.51%) | 35 (11.51%) | — |
| **Full-Frame PSNR (dB)** | 25.418 | **25.498** | **`+0.080`** | **24.078** | 23.976 | **`-0.102`** |
| **Full-Frame SSIM** | **0.8598** | 0.8597 | `-0.0001` | **0.8391** | 0.8378 | `-0.0013` |
| **Object-Masked PSNR (dB)** | 27.199 | **27.229** | **`+0.030`** | **24.000** | 23.939 | **`-0.060`** |
| **Object-Masked SSIM** | **0.8277** | 0.8272 | `-0.0005` | **0.8300** | 0.8289 | `-0.0011` |
| **Object-Masked LPIPS (bbox)** | 0.3654 | **0.3653** | `-0.0001` | 0.2680 | **0.2677** | `-0.0003` |
| **Specular Top-10% PSNR (dB)** | 24.351 | **24.434** | **`+0.083`** | **24.978** | 24.914 | **`-0.064`** |
| **Specular Top-10% (SH4 mask) PSNR** | 24.529 | **24.622** | **`+0.093`** | **25.186** | 25.084 | **`-0.102`** |
| **Specular Top-10% LPIPS (comp)** | 0.0963 | **0.0963** | `-0.0000` | 0.0801 | **0.0801** | `-0.0000` |
| **Specular 512×512 Box LPIPS** | **0.3551** | 0.3566 | `+0.0014` | 0.2354 | **0.2351** | `-0.0004` |
| **Strict-Unseen Arc Specular PSNR** | 23.589 | **23.792** | **`+0.203`** | 21.920 | **22.009** | `+0.089` |
| **Train Sample PSNR (24 views)** | 27.127 | **27.265** | `+0.138` | 25.208 | **25.365** | `+0.157` |
| **Train − Holdout Gap (dB)** | 1.709 | 1.767 | **`+0.058`** | 1.130 | 1.389 | **`+0.259`** |
| **Truncated SH3 Twin Masked PSNR** | — | **27.257** | **`+0.058`** vs SH3 | — | **24.071** | **`+0.072`** vs SH3 |
| **`sh_l1_to_l3_rms` / `sh_l4_rms`** | 0.0319 / 0.0 | 0.0314 / 0.0125 | — | 0.0367 / 0.0 | 0.0363 / 0.0129 | — |
| **`fog_layer_alpha_mean`** | 0.00824 | **0.00808** | `-1.98%` | 0.00854 | **0.00842** | `-1.33%` |
| **`op_lt_005_frac`** | 0.3232 | 0.3190 | `-0.0041` | 0.2679 | 0.2653 | `-0.0026` |
| **Wall Time (5k steps) / Speed** | 383.4s (13.5 it/s) | 402.6s (12.8 it/s) | `+5.0%` time | 146.2s (35.1 it/s) | 177.3s (28.9 it/s) | `+21.3%` time |
| **Peak VRAM (MB)** | 4,667 MB | 6,413 MB | **`+37.4%`** | 3,764 MB | 5,819 MB | **`+54.6%`** |
| **PLY Size / SPZ v4 (8/8) Size** | 708.0 MB / — | 1,032.0 MB / 134.9 MB | `7.65x` SPZ | 708.0 MB / — | 1,032.0 MB / 136.5 MB | `7.56x` SPZ |
| **SPZ v4 (8/8) Orig-vs-Decoded PSNR** | — | **45.26 dB** (min 41.91) | — | — | **49.20 dB** (min 47.38) | — |

### 2.2 Experiment B2 — From-Scratch Arc-Withheld A/B on `GPU_V4` (15,000 steps C2F, Cap 2M)

| Metric | `B2_GPU_V4` SH3 (`--sh_degree 3`) | `B2_GPU_V4` SH4 (`--sh_degree 4`) | $\Delta$ (`SH4 - SH3`) |
|---|---:|---:|---:|
| **Full-Frame PSNR (dB) / SSIM** | 23.909 / **0.8440** | **23.951** / 0.8375 | `+0.042 dB` / `-0.0065` |
| **Object-Masked PSNR (dB) / SSIM** | **26.434** / **0.8134** | 26.281 / 0.7810 | **`-0.152 dB`** / **`-0.0323`** |
| **Object-Masked LPIPS (bbox)** | **0.3949** | 0.3982 | **`+0.0033`** (worse) |
| **Specular Top-10% PSNR (dB)** | 22.516 | **22.740** | **`+0.224 dB`** |
| **Specular 512×512 Box LPIPS** | **0.3868** | 0.3942 | **`+0.0074`** (worse) |
| **Train Sample PSNR (24 views)** | 26.755 | **26.982** | `+0.227 dB` |
| **Train − Holdout Gap (dB)** | 2.846 | 3.030 | **`+0.185 dB`** |
| **Truncated SH3 Twin Masked PSNR** | — | 25.934 | **`-0.500 dB`** vs SH3 (`-0.521 dB` spec) |
| **`sh_l1_to_l3_rms` / `sh_l4_rms`** | 0.0336 / 0.0 | 0.0298 / **0.0282** | $L=4$ RMS $2.26\times$ larger than B1 |
| **`fog_layer_alpha_mean` / `op < 0.05`** | 0.01178 / 0.2087 | 0.01186 / 0.2088 | `+0.65%` fog (negligible) |
| **Wall Time (15k steps) / Peak VRAM** | 695.5s / 3,890 MB | 706.4s / 4,609 MB | `+1.6%` time / `+18.5%` VRAM |
| **PLY Size / SPZ v4 (8/8) Size & PSNR** | 472.0 MB / — | 688.0 MB / 81.4 MB (`8.45x`) | **50.62 dB** (min 49.38 dB) |

---

## 3. Key Technical Findings

1. **Why Contiguous-Arc Holdout Was Essential (`Laptop` B1 Overfit Diagnostic):**
   - On `Laptop` (304 views), unlocking $L=4$ during B1 improved training-view PSNR by **`+0.157 dB`** (`25.208` $\to$ `25.365 dB`), yet **degraded held-out 38° arc PSNR by `-0.102 dB` full / `-0.060 dB` masked**, widening the train–holdout gap by **`+0.259 dB`**.
   - Even more tellingly, when the 9 degree-4 bands of `b1_laptop_sh4` are zeroed out (`sh4_truncated_to_sh3`), held-out masked PSNR on `Laptop` **jumps from `23.939 dB` to `24.071 dB` (`+0.132 dB`)**, beating both full SH4 and native SH3! This proves that on sparser camera orbits (304 views), the 9 $L=4$ basis functions fit high-frequency angular lobes to the training cameras that oscillate destructively across a 38° unseen azimuth gap.
   - On denser camera orbits (`GPU_V4`, 569 views), `B1_GPU_V4` does not overfit across the 38° arc (`+0.080 dB` full PSNR, `+0.083 dB` specular top-10% PSNR, and `+0.203 dB` specular PSNR on the 8 strict-unseen views), but the gain remains below the `+0.30 dB` threshold required to justify a +45.8% PLY size and +37–55% VRAM increase by default.

2. **Warm-Start (B1) vs From-Scratch (B2) Co-Adaptation & Postshot Truncation:**
   - When trained from scratch with `--sh_degree_interval 1000` (unlocking $L=4$ at step 4,000 during coarse MCMC densification), the optimizer distributes view-dependent energy heavily into $L=4$ (`sh_l4_rms = 0.0282` vs `0.0125` in B1) at the expense of $L \le 3$ (`sh_l1_to_l3_rms` drops from `0.0336` to `0.0298`).
   - Consequently, truncating a **from-scratch** SH4 PLY to degree 3 for Postshot consumers (`tools/truncate_sh_ply.py`) causes a **`-0.500 dB` masked PSNR drop** (`-0.521 dB` on specular pixels), failing Criterion 4.
   - Conversely, **warm-starting** SH4 from a converged SH3 checkpoint (Experiment B1) keeps $L=4$ purely residual (`sh_l4_rms = 0.0125`), so `tools/truncate_sh_ply.py` produces a degree-3 Postshot twin that scores **`+0.058 dB` (`GPU_V4`) and `+0.072 dB` (`Laptop`) ABOVE native SH3**.

---

## 4. Provenance & Artifacts

- **Holdout lists:** `_infusion/wpsh4/phaseB/gpu_holdout_arc.txt` (60/569 views), `_infusion/wpsh4/phaseB/laptop_holdout_arc.txt` (35/304 views)
- **Evaluator:** `_infusion/wpsh4_phaseB_eval.py`
- **Machine-readable gate:** `_infusion/wpsh4/phaseB_gate.json`
- **Training logs & checkpoints:** `_infusion/wpsh4/phaseB/{b1_gpu_sh3,b1_gpu_sh4,b1_laptop_sh3,b1_laptop_sh4,b2_gpu_sh3,b2_gpu_sh4}.log`
