# WP-Q1 phase-1 gate record (2026-07-02)

Produced by `qa_report.py` (core/qa/ harness, components 1+8). All runs factor-2,
split every8 unless noted. Determinism verified: identical `scorecard.json`
sha256 across repeat runs.

| run | corruption | obs% | mPSNR | mSSIM | mLPIPS |
|---|---|---|---|---|---|
| GPU_V4 (ckpt_29999, 2M GS) | none | 100.0 | 25.278 | 0.8514 | 0.4125 |
| GPU_V4_floaters | 2000 floaters @ scale 0.01 | 100.0 | 12.487 | 0.6209 | 0.7573 |
| GPU_V4_floaters_mild | 2000 floaters @ scale 0.003 | 100.0 | 20.309 | 0.7874 | 0.4851 |
| GPU_V4_blur | render blur sigma=2 | 100.0 | 25.724 | 0.8559 | 0.4319 |
| Laptop (ckpt_29999, 3M GS) | none | 100.0 | 22.319 | 0.8120 | 0.3540 |
| Laptop_wedge (views 131–169) | none | 100.0 | 23.618 | 0.8427 | 0.3491 |

**Blur-corruption note (expected, verified — not a harness failure):** for
blur-type corruption, **LPIPS is the judging axis**. PSNR/SSIM can move slightly
POSITIVE under blur by construction (blur suppresses high-frequency residual
error), which is exactly why the plan mandates judging perceptual work by
LPIPS/region metrics, never PSNR alone.

100% observed on both production scenes is correct (dense captures). Mask
flagging validated by slab-deletion: removing the z>=p55 Gaussian slab drops a
mid-trajectory view to 45.2% observed with the removed region tinted red
(`_flagdemo_hole.png`), and the full masked-metrics path runs on that partial
mask (LPIPS bbox 0.4802 / tiled 0.2970, PSNR 23.242, SSIM 0.8308).
Sliver-mask robustness (thin bboxes, sub-64px blobs, frames smaller than the
LPIPS crop) covered by expansion-within-bounds + replicate padding.
