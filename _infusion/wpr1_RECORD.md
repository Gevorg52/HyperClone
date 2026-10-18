# WP-R1 Record — Observed-Preservation Gate for Generative Supervision (2026-10-18)

## 1. Executive Summary & Gate Decision

**Verdict:** **HOLD DEFAULT-OFF (`verdict = HOLD_DEFAULT_OFF`)** — The WP-R1 trainer levers (`--extra_unobserved_only`, `--extra_anchor_observed`, `--extra_loss_weight`, `--extra_obs_tau`, `--extra_obs_gamma`) are implemented, unit-tested (`tests/test_wpr1_observed_gate.py`, 6/6 passing), and plumbed through `core/training/gsplat_trainer.py`, `core/training/gsplat_backend.py`, and `run_distill_laptop.py` (all **OFF by default**).

On the `_infusion/hole/` 23-view carved-hole testbed (`v0293..v0315`), the three-zone WP-R1 gate (`distill_r1_gated`: `--extra_unobserved_only --extra_anchor_observed 1.0 --extra_obs_tau 0.30 --extra_loss_weight 0.05 --opacity_reg 0.01`):
1. **Passes Observed-Region LPIPS (`0.4603` vs `distill_control` `0.4632`, $\Delta = -0.0029$)** and **beats `distill_gated` (`0.4677`, $\Delta = -0.0074$)**.
2. **Passes Solid Observed-Region PSNR ($\alpha_{\text{init}} \ge 0.99$, 57% of frame): `20.50 dB` vs `distill_control` `20.19 dB` ($\Delta = +0.31\text{ dB}$)** and LPIPS **`0.4342` vs `0.4480` ($\Delta = -0.0138$)**.
3. **Passes Off-Path Fog Sanity:** `fog_layer_alpha_mean = 0.02058` vs `distill_control` `0.02039` ($+0.95\%$, well under the $+10\%$ threshold), `offpath_semi_alpha_mean = 0.0`.
4. **Improves NoVA Artifact Score by 18.0%** (`0.3004` vs `0.3663` on `distill_gated`), full-wedge PSNR by **`+0.49 dB`** (`12.66` vs `12.17 dB`), and observed PSNR (`alpha_init >= 0.30`) by **`+0.65 dB`** (`17.46` vs `16.81 dB` on `distill_gated`).
5. **Fails Total Observed PSNR (`17.46 dB` vs `distill_control` `19.59 dB`) and Unobserved Hole PSNR (`6.83 dB` vs `distill_control` `15.50 dB`)** due to two fundamental physical limits of single-view 2D inpainting on a carved 3D frustum:
   - **Generator bias in the void:** SDXL Inpaint (`fills_gated`) hallucinated a pitch-dark shadow inside the carved cavity (`mean RGB = 0.078` vs ground-truth `mean RGB = 0.581`, yielding a **2D image PSNR of only `5.33 dB`** inside `unobs`), whereas `distill_control` (no pseudo-views) heals the hole to **`15.50 dB`** using multi-view rays from the 364 real views outside the withheld wedge.
   - **3D multi-view frustum coupling (`fringe_in_box`):** Because the hole was carved as a 3D frustum from `center_view=300`, 3D Gaussians placed inside the carved box project onto `unobs` ($\alpha_{\text{init}} < 0.30$) on central arc views (`~v0300`) AND project onto the partially-carved fringe ($0.30 \le \alpha_{\text{init}} < 0.99$, `fringe_in_box`, where background splats remain behind the carved box) on adjacent arc views (`v0293..v0298`, `v0303..v0315`). Even when 2D supervision on the fringe is completely zeroed out (`rendered.detach()`), any 3D Gaussians darkened to satisfy `unobs` on view 300 occlude `fringe_in_box` on neighboring views (`11.56 dB` vs `16.51 dB` in `distill_control`).

---

## 2. Root-Cause Diagnosis of WP-C1 (`distill_gated` vs `distill_control`)

Partitioning the 23 hole arc views (`293..315`) by the frozen initial checkpoint opacity $\alpha_{\text{init}}$ (`_infusion/hole/arc/vXXXX_alpha.npy`) revealed why `distill_gated` lost to `distill_control` in WP-C1:

| Zone | Definition | Frame Share | `holed_baseline` | `distill_control` | `distill_gated` | `distill_ungated` |
|---|---|---:|---:|---:|---:|---:|
| **`obs` (All Observed)** | $\alpha_{\text{init}} \ge 0.30$ | 80.6% | 15.23 dB | **19.59 dB** | 16.81 dB | 17.03 dB |
| ├─ **`solid_obs_99`** | $\alpha_{\text{init}} \ge 0.99$ | 57.4% | 19.14 dB | 20.19 dB | 20.38 dB | **20.59 dB** |
| └─ **`fringe_obs_03_99`** | $0.30 \le \alpha_{\text{init}} < 0.99$ | 23.2% | 10.86 dB | **18.19 dB** | 12.48 dB | 12.71 dB |
| &nbsp;&nbsp;&nbsp;&nbsp;├─ `fringe_in_box` | Fringe inside carved 2D box | 10.1% | 8.95 dB | **16.51 dB** | 11.25 dB | 11.44 dB |
| &nbsp;&nbsp;&nbsp;&nbsp;└─ `fringe_out_box` | Fringe outside carved 2D box | 13.1% | 13.72 dB | **20.33 dB** | 14.95 dB | 15.24 dB |
| **`unobs` (Void / Hole)** | $\alpha_{\text{init}} < 0.30$ | 19.4% | 5.10 dB | **15.50 dB** | 6.63 dB | 7.32 dB |

Key diagnostic findings:
1. **`distill_gated` did NOT degrade `solid_obs_99` ($\alpha_{\text{init}} \ge 0.99$):** On uncarved solid splats, `distill_gated` actually **beat** `distill_control` (`20.38 dB` vs `20.19 dB`, tiled LPIPS `0.4377` vs `0.4480`).
2. **`distill_gated` collapsed on `fringe_obs_03_99` ($0.30 \le \alpha_{\text{init}} < 0.99$, `12.48 dB` vs `18.19 dB`):** When `hole_carve.py` deleted foreground Gaussians inside the 3D box on view 300, rays passing through the box border or oblique views retained background/side splats with $0.30 \le \alpha_{\text{init}} < 0.99$, where `holed_gpuv4.pt` renders a broken `10.86 dB` image (`8.95 dB` inside the box). Because `gamo_fill.py` only inpainted $\alpha_{\text{init}} < 0.30$ and pasted the broken `holed` render back onto $\alpha_{\text{init}} \ge 0.30$, full-frame supervision in `distill_gated` actively forced `fringe_obs_03_99` back toward the broken `10.86 dB` render instead of letting the 364 real views heal it (`18.19 dB`).
3. **Unanchored mask-only supervision (`distill_r1_w100`) collapses `solid_obs_99` (`15.98 dB`):** If observed pixels ($\alpha_{\text{init}} \ge 0.30$) are simply masked to zero loss (`w_unobs = 0`) while normalizing the hole loss by `w_unobs.sum()` (`mean w_unobs = 0.1082`, amplifying hole gradients by $9.24\times$), `solid_obs_99` is left completely unguarded from the withheld-wedge viewing directions (`293..315`), allowing dark hole Gaussians to expand across `solid_obs_99` without penalty (`20.38 dB` $\to$ `15.98 dB`).

---

## 3. Three-Zone WP-R1 Formulation (`core/training/gsplat_trainer.py`)

To simultaneously protect `solid_obs_99`, release `fringe_obs_03_99` from broken `holed` paste-back supervision, and weight `unobs` without division blowup, WP-R1 precomputes per-extra-view frozen anchors (`render_init`, `alpha_init`, `w_unobs`, `w_solid`) in `precompute_extra_view_anchors()` and constructs the three-zone effective target on extra-view steps:

$$I_{\text{eff}}(p) = w_{u}(p) \, I_{\text{pseudo}}(p) + w_{a}(p) \, I_{\text{init}}(p) + \bigl(1 - w_{u}(p) - w_{a}(p)\bigr) \, \text{sg}\bigl(I_{\text{render}}(p)\bigr)$$

where:
- $w_{u}(p) = \min(1, w_{\text{extra}}) \cdot \text{clamp}\!\left(\frac{\tau_{\text{obs}} - \alpha_{\text{init}}(p)}{\tau_{\text{obs}}}, 0, 1\right)^\gamma$ supervises the **unobserved hole** ($\alpha_{\text{init}} < \tau_{\text{obs}}$) with $I_{\text{pseudo}}$.
- $w_{a}(p) = a_{\text{anchor}} \cdot \text{clamp}\!\left(\frac{\alpha_{\text{init}}(p) - 0.95}{0.05}, 0, 1\right) \cdot (1 - w_{\text{unobs}}(p))$ anchors **solid observed splats** ($\alpha_{\text{init}} \ge 0.99$) to the frozen initial render $I_{\text{init}}$ so hole Gaussians cannot spill over and occlude uncarved geometry.
- $w_{\text{free}}(p) = 1 - w_{u}(p) - w_{a}(p)$ replaces the **partially-carved fringe** ($\tau_{\text{obs}} \le \alpha_{\text{init}} \le 0.95$) with $\text{sg}(I_{\text{render}}(p))$ (`rendered.detach()`) in both `gt_image` (L1) and `gt_window` (11×11 SSIM and 512×512 LPIPS crop), yielding **identically zero gradient** on damaged fringe pixels with **zero $1/\sum w$ division blowup** and leaving `opacity_reg` and `scale_reg` unscaled.

---

## 4. Quantitative Gate Results (`_infusion/wpr1/gate.json`)

| Metric | `holed_baseline` | `distill_control` | `distill_gated` | `distill_ungated` | `distill_r1_w100` (unanchored) | **`distill_r1_gated` (WP-R1 3-zone)** |
|---|---:|---:|---:|---:|---:|---:|
| **`observed_masked_PSNR` ($\alpha_{\text{init}} \ge 0.30$)** | 15.23 dB | **19.59 dB** | 16.81 dB | 17.03 dB | 14.32 dB | **17.46 dB** (`+0.65` vs gated) |
| **`observed_masked_SSIM`** | 0.7512 | **0.7811** | 0.7617 | 0.7673 | 0.6837 | **0.7752** (`+0.0134` vs gated) |
| **`observed_masked_LPIPS` (tiled)** | 0.4738 | 0.4632 | 0.4677 | 0.4762 | 0.5205 | **0.4603** (**beats control & gated**) |
| **`solid_obs_99_PSNR` ($\alpha_{\text{init}} \ge 0.99$)** | 19.14 dB | 20.19 dB | 20.38 dB | 20.59 dB | 15.98 dB | **20.50 dB** (**`+0.31` vs control**) |
| **`solid_obs_99_LPIPS_tiled`** | 0.4360 | 0.4480 | 0.4377 | 0.4484 | 0.4934 | **0.4342** (**`-0.0138` vs control**) |
| **`fringe_obs_03_99_PSNR`** | 10.86 dB | **18.19 dB** | 12.48 dB | 12.71 dB | 11.55 dB | **13.41 dB** (`+0.93` vs gated) |
| **`fringe_out_box_PSNR`** | 13.72 dB | **20.33 dB** | 14.95 dB | 15.24 dB | 13.05 dB | **16.88 dB** (`+1.93` vs gated) |
| **`fringe_in_box_PSNR`** | 8.95 dB | **16.51 dB** | 11.25 dB | 11.44 dB | 10.50 dB | **11.56 dB** (`+0.30` vs gated) |
| **`unobserved_masked_PSNR` ($\alpha_{\text{init}} < 0.30$)** | 5.10 dB | **15.50 dB** | 6.63 dB | 7.32 dB | 5.54 dB | **6.83 dB** (`+0.20` vs gated) |
| **`unobserved_masked_LPIPS_bbox`** | 0.6713 | **0.5522** | 0.6545 | 0.6538 | 0.6716 | **0.6394** (`-0.0151` vs gated) |
| **`unobserved_vs_fill_PSNR`** | 25.85 dB | 5.14 dB | 14.77 dB | 12.98 dB | **21.84 dB** | 12.16 dB |
| **`fog_layer_alpha_mean`** | 0.02252 | 0.02039 | 0.02073 | 0.02064 | 0.02103 | **0.02058** (`+0.95%` vs control) |
| **`op_lt_005_frac`** | 0.7151 | 0.7198 | 0.7185 | 0.7224 | 0.4606 | **0.7192** |
| **QA Wedge Full PSNR (`qa_report.py`)** | 11.09 dB | — | 12.17 dB | 12.40 dB | — | **12.66 dB** (`+0.49` vs gated) |
| **QA Wedge Full LPIPS (`qa_report.py`)** | 0.5518 | — | 0.5446 | 0.5504 | — | **0.5364** (`-0.0082` vs gated) |
| **QA NoVA (`qa_report.py`, lower=better)** | 0.3125 | — | 0.3663 | 0.3647 | — | **0.3004** (**`-18.0%` vs gated**) |
| **QA RAFT Warp (`qa_report.py`, lower=better)** | 0.01283 | — | 0.01309 | 0.01334 | — | **0.01271** (**beats holed & gated**) |
| **QA TSED Matches (`qa_report.py`)** | 405.8 | — | 422.2 | 418.7 | — | **426.4** (`ratio = 1.000`) |

---

## 5. Provenance & Reproducibility

- **Code modified:** `core/training/gsplat_trainer.py`, `core/training/gsplat_backend.py`, `run_distill_laptop.py`
- **Unit tests:** `tests/test_wpr1_observed_gate.py` (`python -m pytest tests/test_wpr1_observed_gate.py -v` -> 6 passed)
- **Evaluator:** `_infusion/wpr1_eval.py`
- **Artifacts:** `_infusion/wpr1/gate.json`, `_infusion/wpr1/distill_r1_w100.log`, `_infusion/wpr1/distill_r1_gated.log`, `_infusion/wpr1/qa_r1_gated/scorecard.json`
