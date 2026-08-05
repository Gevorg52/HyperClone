# Loading the WP-O5 8K splat in Postshot

_A copy of this file sits next to the PLYs in `_dataset/360/train/wpo5_run_D/ply/`
(that directory is gitignored, so this committed copy is the reference.)_

## Which file to load

| file | size | Gaussians | use |
|---|---|---|---|
| **`point_cloud_29999_aligned_pruned_op05.ply`** | 567.5 MB | 2 404 841 | **start here** — level, metric, 20 % smaller, renders identically to the full one |
| `point_cloud_29999_aligned.ply` | 708.0 MB | 3 000 000 | same thing unpruned |
| `point_cloud_29999.ply` | 708.0 MB | 3 000 000 | the raw training output — tilted floor, arbitrary origin/scale. Superseded. |

The pruned file drops every Gaussian below opacity 0.05. **That is a file-size change, not a
quality change** — we measured it: the render is visually identical at every viewpoint tested,
on-path and off. Use it because it loads faster, not because it looks better.

## What "aligned" means

`point_cloud_29999.ply` came out of the trainer in a normalised SfM frame: arbitrary origin,
arbitrary scale, and tilted (the capture is a hand-held 360 stick; WP-O2 measured panorama
zenith tilt at median 11°, max 49°). That is why the floor was not level and navigation felt
disorienting. The aligned files have one similarity transform baked in:

- **Up axis: Y-down** — i.e. up is **−Y**, Y increases toward the floor. This is the standard
  INRIA / gsplat / COLMAP 3DGS convention, which is what **Postshot itself exports** and what
  SuperSplat and PlayCanvas assume (they apply a fixed 180° Z rotation to match it). If the
  scene still loads upside-down, that is the fix: rotate 180° about Z, or use Postshot's
  **Align Up Vector** on the Rdnc Field, then **Bake Transform**.
- **Floor plane is at Y = 0**, fitted by RANSAC to the reconstructed floor.
- **Origin** is the centre of the walked area, on the floor.
- **Units are approximately metres** (see the assumption below). Camera height along the walk
  is a median of **2.15 m** above the floor — that is the selfie-stick height, and it is where
  the splat is best supported. The walked area is about **18 × 14 m**.

**The metric-scale assumption, stated plainly:** the ingest samples 2 panoramas per second, so
one step between capture stations is 0.5 s of walking, taken as **0.65 m**. Everything metric
downstream inherits that. The measured median step is 0.1537 normalised units, giving
**4.2286 m per unit**. If the true walking speed differed, every distance scales with it —
the *shape* is unaffected.

**The transform is lossless.** It was verified per run by rendering the same views from the
original model and from the aligned model with correspondingly transformed cameras: worst case
over 4 views was **max |pixel difference| 0.018 / 255-scale ≈ 4.6**, mean 7.8e-6, **min PSNR
74.2 dB** — i.e. float32 rasterizer noise, nothing else. Means, scales, quaternions **and the
spherical-harmonic coefficients** were all rotated properly (the SH rotation is the part that
is usually skipped, which silently breaks view-dependent shading as you orbit).
Full numbers: `_infusion/wpo5/align_armD.json`.

## Where the good spaces are

Open **`_infusion/wpo5/eval/GUIDE.jpg`**. It is a top-down floor plan of the walked path in
metres with **8 numbered stops** and an arrow at each, beside the view you get standing at that
stop looking along the arrow (eye height 2.15 m, 75° FOV). The stops are the positions with the
most surrounding capture stations, i.e. the best-supported places in the model.
`GUIDE.json` has their exact metric coordinates if you want to fly straight there.

Rough tour: **stop 1** is the far end of the corridor looking back (the IV-bag rack room with
the bed is visible); **stops 2–3** are the corridor and the doorway into the theatre;
**stops 5–8** are inside the rooms off the loop.

## Honest expectations

- **Stay within ~1.5 m of the walked path.** Out to about ±1–1.5 m laterally the rooms stay
  structurally correct — walls planar, doorways and fixtures in the right places. That is a
  real improvement over the previous `_077` splat, which smeared at the same displacement.
- **Beyond that it falls apart.** At a position 4.6 m from the nearest capture station —
  still inside the building, inside the area the operator walked *around* — every arm we
  trained renders incoherent blur. The capture is a single walking pass; there is no parallax
  to reconstruct those viewpoints from. This is a capture-geometry limit, not a setting.
- **It is soft, not sharp.** Held-out masked PSNR 21.5 dB / SSIM 0.71. Much better than the
  earlier proxy capture (17.5 dB / 0.43) because this one is real 8K, but the scene is
  texture-poor (plain tiled walls) and that starved the SfM: 4.7 inlier matches per image pair
  where the previous scene gave 39.2, so the model was initialised from 16× fewer points per
  panorama.
- **Free-fly far from the path will look bad and that is expected.** If you want a splat that
  survives free navigation of this space, the capture needs multiple passes at different
  heights/offsets, not a different training setting.

## If it still loads wrong

1. Wrong way up → 180° about Z, or **Align Up Vector** → **Bake Transform**.
2. Wrong scale → the file is metric under the 0.65 m/step assumption; rescale uniformly.
3. Ground plane not at the floor → the floor is Y = 0 by construction; check the up axis first.
