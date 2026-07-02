# WP-Q2b SA-GS eps2d gate strips — notes (2026-07-02)

Strips: `sags_z{2,4}_view{0,12,24}.jpg` (zoom-in) and `sags_z0p5_view{0,12,24}.jpg`
(zoom-out), each [OFF eps2d=0.3 | ON SA-GS rule] side-by-side on the GPU_V4 30k
prod checkpoint. Inspection crops in `crops/`.

**Caveat on the zoom-out result:** the 0.5x "ON" column bundles TWO levers by
design (per the execution plan's `--eval_ss` recipe): the SA-GS eps2d rule
(`0.3*(r*ss)^2`) AND 4x supersampling with box-downsample. The +9.9 dB
vs-band-limited-reference win at 0.5x is the **combined recipe**, not eps2d
alone. The zoom-in strips (z2, z4, ss=1) are the clean isolated eps2d A/B:
there the rule suppresses the needle-streak/dilation artifacts but softens
genuine detail proportionally (it reproduces training-time appearance,
magnified) — a visual trade-off, not a free win.
