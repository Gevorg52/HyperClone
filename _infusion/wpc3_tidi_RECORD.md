# WP-C3 Part 1 Record: TIDI-GS Cross-View Evidence, Importance & Isolation Floater Pruner

**Date:** 2026-04-18  
**Status:** GATE PASSED across all 3 scenes (`Laptop`, `GPU_V4`, `360`)  
**Deliverables:**
- `tools/prune_tidi.py` — Standalone CLI & importable Python API (`compute_tidi_scores`, `prune_ckpt_tidi`, `compute_view_blending_weights`, `tidi_keep_mask`, `TidiConfig`)
- `tests/test_prune_tidi.py` — 3 unit tests verifying backward-pass blending weight exactness vs forward one-hot oracle, detection of `opacity=0.12` synthetic floaters missed by `tools/prune_ply.py`, and preservation of multi-view surface Gaussians (`3 passed`)
- `_infusion/wpc3_tidi_eval.py` — Full 3-scene gate evaluation harness (`unpruned`, `prune_op05`, `prune_tidi_cons`, `prune_tidi`)
- `_infusion/wpc3/tidi_gate.json` — Raw gate metrics JSON (`"all_gates_passed": true`)

---

## 1. Root-Cause Findings

### 1.1 Residual Off-Path Fog (`0.05 <= opacity < 0.12`)
WP-A1 (`_infusion/wpa1/residual_audit_results.json`) established that simple global opacity thresholding (`tools/prune_ply.py --min_opacity 0.05`) removes ultra-low-opacity dust (`opacity < 0.05`) with zero quality loss (`-0.027 dB` mPSNR), but leaves behind:
1. **Mid-opacity off-path fog (`0.05 <= opacity < 0.12`)** that sits in front of the scene surface from off-axis views or only contributes to 1 training camera (`active_views < 2`). Raising the global opacity threshold to `0.15` or `0.20` destroys thin multi-view surface geometry and specular highlights (`-0.31 dB` to `-1.19 dB` mPSNR).
2. **Spatially isolated 3D floaters (`floating_sor_002`: `100k–204k` splats)** and **outside-bbox runaway splats (`>3x` scene extent)**.

### 1.2 Major Discovery: "Mip-Splatting Zombie Dust" (`383k` to `1.44M` Dead Splats in `prune_op05`)
Cross-view rasterization instrumentation in `tools/prune_tidi.py` uncovered a previously undiagnosed failure mode of combining `--antialiased` (Mip-Splatting 2D filter) with `--scale_reg 0.01` and `--opacity_reg 0.0`:
- During MCMC training, `--scale_reg 0.01` continuously shrinks the 3D scales of unused/occluded Gaussians toward `1e-8 .. 1e-7`.
- When a Gaussian's projected 2D covariance $\Sigma_{2D}$ becomes much smaller than the Mip-Splatting 2D filter variance ($\epsilon_{2D} = 0.3\text{ px}^2$), Mip-Splatting's amplitude compensation multiplies its effective 2D opacity by:
  $$\alpha_{\text{eff}} = \alpha_{\text{raw}} \cdot \sqrt{\frac{\det(\Sigma_{2D})}{\det(\Sigma_{2D} + \epsilon_{2D} I)}} \ll \frac{1}{255}$$
- Because $\alpha_{\text{eff}} < 1/255$, `gsplat`'s projection kernel culls the Gaussian (`radii == 0`) across **every** training view (`proj_views == 0`).
- Once culled, the Gaussian receives **zero gradient** for the remainder of training, freezing its raw parameter opacity at whatever value it had when its scale collapsed (`0.05 <= sigmoid(opacity) <= 0.99`).
- Neither MCMC dead-splat relocation (`op <= 0.005`) nor `tools/prune_ply.py` (`op < 0.05`) ever detects or removes these "zombie" Gaussians!
- **Prevalence in `prune_op05` (`opacity >= 0.05`) checkpoints:**
  - **Laptop (`opreg0/or0`):** `383,128` splats with `proj_views == 0` + `205,409` splats with `active_views < 2` (**26.8% of `prune_op05` is dead/single-view**)
  - **GPU_V4 (`wpf5/base`):** `158,747` splats with `proj_views == 0` + `215,919` splats with `active_views < 2` (**18.6% of `prune_op05` is dead/single-view**)
  - **360 (`wpf5_base`):** `1,440,562` splats with `proj_views == 0` + `165,375` splats with `active_views < 2` (**66.5% of `prune_op05` is dead/single-view**, including `1.34M` giant sky-shell zombies with `max_scale > 0.25`!)

---

## 2. Mathematical Formulation & Architecture (`tools/prune_tidi.py`)

### 2.1 Exact Single-Backward Blending Weight Identity
Standard `gsplat.rasterization` does not expose per-Gaussian forward blending weights $w_i(v) = \sum_{p \in v} T_{p,i} \alpha_{p,i}$ in Python. However, if we pass a 1-channel unit per-Gaussian color $c_{i,0} = 1$ (`colors = torch.ones((N, 1), requires_grad=True)`) with `render_mode="RGB+ED"` (`packed=False`):
- Channel 0 of `renders` is the accumulated alpha $A_p = \sum_{i} T_{p,i} \alpha_{p,i} c_{i,0}$.
- Channel 1 of `renders` is the expected depth $D_{\text{exp}}(u, v) = \sum_{i} T_{p,i} \alpha_{p,i} z_i$.
- Calling `.backward()` on $\mathcal{L} = \sum_{p} \text{renders}[p, 0]$ yields two exact per-Gaussian quantities in a **single** rasterization pass per view:
  1. **Exact forward alpha-blending weight (`colors.grad[:, 0]`):**
     $$\frac{\partial \mathcal{L}}{\partial c_{i,0}} = \sum_{p} T_{p,i} \alpha_{p,i} = w_i(v)$$
     Because $c_{j,0} = 1$ is independent of $c_{i,0}$ for $j \neq i$, `colors.grad` does **not** suffer from the $\prod_{j > i}(1 - \alpha_{p,j}) = T_{\text{final}} / T_{p,i+1}$ downstream occlusion factor that attenuates `opacities.grad` when opaque splats lie behind $i$. Verified in `tests/test_prune_tidi.py::test_backward_blending_weights_match_forward_oracle` to match a per-splat one-hot forward pass to `< 3e-5` max error (`r > 0.999999`).
  2. **Accumulated-alpha sensitivity (`opacities * opacities.grad`):**
     $$\alpha_i \frac{\partial \mathcal{L}}{\partial \alpha_i} = \sum_{p} T_{p,i} \alpha_{p,i} \prod_{j > i}(1 - \alpha_{p,j})$$

### 2.2 Depth Consistency (`infront_views` vs `surface_views`)
For every view $v$ where Gaussian $i$ projects inside the frustum (`radii > 0`) and has non-trivial blending contribution ($w_i(v) \ge 10^{-3}$), we bilinearly sample the rendered expected depth map $D_{\text{exp}}(u_i, v_i)$ at the Gaussian's projected 2D center $(u_i, v_i)$:
- **Surface-consistent (`surface_views`):** $|z_i - D_{\text{exp}}(u_i, v_i)| \le \max(0.05 \cdot D_{\text{exp}}, 3 \cdot s_{\max, i})$
- **Mid-air floater in front (`infront_views`):** $z_i < D_{\text{exp}}(u_i, v_i) - \max(0.15 \cdot D_{\text{exp}}, 4 \cdot s_{\max, i})$

### 2.3 3D Spatial Isolation on Cross-View-Supported Subset + Sky-Dome Protection
1. **Supported-Subset k-NN (`knn8_dist`):** Computing 3D k-NN (`k=8`) on the raw point cloud fails when dense clouds of `proj_views == 0` zombie splats shield nearby floaters. `compute_tidi_scores` builds the `scipy.spatial.cKDTree` strictly on the **cross-view-supported subset** (`opacity >= 0.05` & `active_views >= 2` & `importance_total >= 0.10`), exposing isolated 3D floaters (`knn8_dist > sor_thresh_abs = 0.02 * diag_98`).
2. **Sky-Dome Protection in Bbox Pruning (`rel_bbox_radius`):** On full-scene outdoor captures (`Laptop`), ~66 distant sky/horizon Gaussians sit at `rel_bbox_radius in [3.0, 10.0]` (`100` to `400,000` pixels of blending weight across `5–38` views). Dropping them degrades mPSNR by `-0.048 dB` and mLPIPS by `+0.0019`. `tidi_keep_mask` protects distant sky-dome splats that satisfy `rel_bbox_radius <= 10.0`, `active_views >= 5`, and `importance_total >= 100.0`, while unconditionally pruning extreme runaway outliers (`rel_bbox_radius > 10.0` $\rightarrow$ `0` remaining).

---

## 3. Gate Evaluation Results Across All 3 Scenes

Evaluated via `python _infusion/wpc3_tidi_eval.py --scene all` (`_infusion/wpc3/tidi_gate.json`).

### 3.1 Summary Table (`prune_tidi` vs `unpruned` and `prune_op05`)

| Scene | Variant | Splats Kept | vs `unpruned` | vs `prune_op05` | mPSNR (dB) | $\Delta$mPSNR vs `op05` | mSSIM | mLPIPS (VGG) | $\Delta$mLPIPS vs `op05` | Sub-group LPIPS | `res_fogA` (`0.05..0.12`) | `floating_sor_002` | `>3x` Bbox (`>10x`) | PLY Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Laptop** | `unpruned` | `3,000,000` | `100.0%` | — | `23.0367` | `+0.0271` | `0.8311` | `0.2794` | `-0.0007` | `0.2400` (deck) | `0.2638` | `274,352` (`9.15%`) | `1,878` (`1,187`) | `708.0 MB` |
| **Laptop** | `prune_op05` | `2,199,888` | `73.3%` | `100.0%` | `23.0096` | `0.0000` | `0.8304` | `0.2801` | `0.0000` | `0.2412` (deck) | `0.2314` | `204,165` (`6.81%`) | `381` (`92`) | `519.2 MB` |
| **Laptop** | `prune_tidi_cons` | `1,570,076` | `52.3%` | `-28.6%` | `23.0102` | **`+0.0006`** | `0.8304` | `0.2803` | **`+0.0002`** | `0.2411` (deck) | `0.0497` (`-78.5%`) | `203,824` (`6.79%`) | `299` (`0`) | `370.5 MB` |
| **Laptop** | **`prune_tidi`** | **`1,551,279`** | **`51.7%`** | **`-29.5%`** | **`23.0100`** | **`+0.0004`** | **`0.8304`** | **`0.2804`** | **`+0.0003`** | **`0.2412` (deck)** | **`0.0000` (`-100%`)** | **`189,889` (`6.33%`)** | **`297` (`0`)** | **`366.1 MB`** |
| **GPU_V4** | `unpruned` | `3,000,000` | `100.0%` | — | `24.7436` | `+0.0266` | `0.8437` | `0.3698` | `-0.0007` | `0.2997` (shiny) | `0.2359` | `129,702` (`4.32%`) | `1,265` (`1,055`) | `708.0 MB` |
| **GPU_V4** | `prune_op05` | `2,019,517` | `67.3%` | `100.0%` | `24.7170` | `0.0000` | `0.8430` | `0.3705` | `0.0000` | `0.3009` (shiny) | `0.1765` | `100,954` (`3.37%`) | `59` (`35`) | `476.6 MB` |
| **GPU_V4** | `prune_tidi_cons` | `1,608,516` | `53.6%` | `-20.4%` | `24.7170` | **`0.0000`** | `0.8430` | `0.3706` | **`+0.0001`** | `0.3009` (shiny) | `0.0269` (`-84.8%`) | `98,809` (`3.29%`) | `21` (`0`) | `379.6 MB` |
| **GPU_V4** | **`prune_tidi`** | **`1,602,034`** | **`53.4%`** | **`-20.7%`** | **`24.7170`** | **`0.0000`** | **`0.8430`** | **`0.3707`** | **`+0.0002`** | **`0.3010` (shiny)** | **`0.0000` (`-100%`)** | **`94,025` (`3.13%`)** | **`18` (`0`)** | **`378.1 MB`** |
| **360** | `unpruned` | `3,000,000` | `100.0%` | — | `21.4196` | `+0.0075` | `0.7146` | `0.5524` | `0.0000` | `0.5819` (bbox) | `0.0515` | `32,618` (`1.09%`) | `4,393` (`698`) | `708.0 MB` |
| **360** | `prune_op05` | `2,413,894` | `80.5%` | `100.0%` | `21.4121` | `0.0000` | `0.7144` | `0.5524` | `0.0000` | `0.5820` (bbox) | `0.0329` | `23,132` (`0.77%`) | `2,240` (`488`) | `569.7 MB` |
| **360** | `prune_tidi_cons` | `794,549` | `26.5%` | `-67.1%` | `21.4113` | **`-0.0008`** | `0.7144` | `0.5524` | **`0.0000`** | `0.5820` (bbox) | `0.0054` (`-83.6%`) | `23,441` (`0.78%`) | `846` (`0`) | `187.5 MB` |
| **360** | **`prune_tidi`** | **`793,497`** | **`26.4%`** | **`-67.1%`** | **`21.4113`** | **`-0.0008`** | **`0.7144`** | **`0.5524`** | **`0.0000`** | **`0.5820` (bbox)** | **`0.0000` (`-100%`)** | **`22,936` (`0.76%`)** | **`816` (`0`)** | **`187.3 MB`** |

### 3.2 Gate Verification Checklist
- **$\Delta\text{mLPIPS} \le +0.0005$ vs `prune_op05`:**
  - Laptop: `+0.0003` (**PASS**)
  - GPU_V4: `+0.0002` (**PASS**)
  - 360: `0.0000` (**PASS**)
- **$\Delta\text{mPSNR} \ge -0.05\text{ dB}$ vs `prune_op05`:**
  - Laptop: `+0.0004 dB` (**PASS** — slightly higher PSNR than `prune_op05`!)
  - GPU_V4: `0.0000 dB` (**PASS**)
  - 360: `-0.0008 dB` (**PASS**)
- **Floater & Splat Count Reduction vs `prune_op05`:**
  - Laptop: `-648,609` splats (`-29.5%`), `res_fogA: 0.2314 -> 0.0000`, `floating_sor_002: 204,165 -> 189,889` (`-14,276` isolated 3D floaters), `>10x` bbox: `92 -> 0` (**PASS**)
  - GPU_V4: `-417,483` splats (`-20.7%`), `res_fogA: 0.1765 -> 0.0000`, `floating_sor_002: 100,954 -> 94,025` (`-6,929` isolated 3D floaters), `>10x` bbox: `35 -> 0` (**PASS**)
  - 360: `-1,620,397` splats (`-67.1%`), `res_fogA: 0.0329 -> 0.0000`, `giant_pct (>0.25): 44.69% -> 10.80%`, PLY size reduced from `569.7 MB` to `187.3 MB` (**PASS**)

---

## 4. Exported Pruned PLY Artifacts & CLI Reproduction

### 4.1 Exported PLY Files on Disk
1. **Laptop:** `_dataset/Laptop/apex_images/training/opreg0/or0/output/ply/point_cloud_29999_pruned_tidi.ply` (`1,551,279` splats, `366.10 MB`)
2. **GPU_V4:** `_dataset/GPU_V4/apex_images/training/wpf5/base/output/ply/point_cloud_29999_pruned_tidi.ply` (`1,602,034` splats, `378.08 MB`)
3. **360:** `_dataset/360/train/wpf5_base/output/ply/point_cloud_29999_pruned_tidi.ply` (`793,497` splats, `187.27 MB`)

### 4.2 CLI Usage
```powershell
# Laptop (moderate default preset)
python tools/prune_tidi.py `
  --ckpt _dataset/Laptop/apex_images/training/opreg0/or0/output/ckpt/ckpt_29999_rank0.pt `
  --data_dir _dataset/Laptop/apex_images `
  --factor 2 `
  --preset moderate `
  --out_ply _dataset/Laptop/apex_images/training/opreg0/or0/output/ply/point_cloud_29999_pruned_tidi.ply

# GPU_V4 (moderate default preset)
python tools/prune_tidi.py `
  --ckpt _dataset/GPU_V4/apex_images/training/wpf5/base/output/ckpt/ckpt_29999_rank0.pt `
  --data_dir _dataset/GPU_V4/apex_images `
  --factor 2 `
  --preset moderate `
  --out_ply _dataset/GPU_V4/apex_images/training/wpf5/base/output/ply/point_cloud_29999_pruned_tidi.ply

# 360 (moderate default preset, factor 1)
python tools/prune_tidi.py `
  --ckpt _dataset/360/train/wpf5_base/output/ckpt/ckpt_29999_rank0.pt `
  --data_dir _dataset/360/train `
  --factor 1 `
  --preset moderate `
  --out_ply _dataset/360/train/wpf5_base/output/ply/point_cloud_29999_pruned_tidi.ply
```
