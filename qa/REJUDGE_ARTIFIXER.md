# ArtiFixer wedge re-judgment — scorecard bank (2026-07-02)

Product-metric re-judgment of the June ArtiFixer3D artifacts (first scientific use
of the WP-Q1 suite). Driver `_infusion/wedge_rejudge.py` (normalize=False consumer
of `core/qa/`; core/qa unchanged). Full analysis + verdict:
`_infusion/wedge_rejudgment_RECORD.md`.

Scorecards: `qa/rejudge_wedge_base/`, `qa/rejudge_wedge_af3d/`,
`qa/rejudge_easy_base/`, `qa/rejudge_easy_af3d/` (each = scorecard.json +
report.html + per-view thumbnails). PLYs re-exported from the surviving WSL
3DGRUT checkpoints → `artifixer-checkpoints/ply/laptop_{base_3dgrut,artifixer3d}_wedge.ply`.
Scene: `artifixer-checkpoints/rejudge/scene_laptop720/` (720p, factor=1, split
wedge:130:170 = frames 0131–0170 / every8; sh_degree=3; seed=0).

| run | obs% | mPSNR | mSSIM | mLPIPS | crossscore↑ | nova↓ | tsed↑ | raft_warp↓ | nvs_sqa↑ | flicker↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| wedge base 3DGRUT | 100.0 | 21.224 | 0.8226 | 0.3626 | 0.8010 | 0.1743 | 0.9091 | 0.0474 | −2.9400 | 0.9948 |
| wedge **ArtiFixer3D** | 100.0 | 21.522 | 0.8262 | 0.3496 | 0.8244 | 0.1611 | 0.8788 | 0.0527 | −2.9281 | 0.9950 |
| easy base 3DGRUT | 100.0 | 22.401 | 0.8207 | 0.3420 | 0.7943 | 0.1611 | 0.9091 | 0.0626 | −2.9464 | 0.9945 |
| easy **ArtiFixer3D** | 100.0 | 22.563 | 0.8249 | 0.3059 | 0.8336 | 0.1408 | 0.9394 | 0.0525 | −2.9307 | 0.9951 |

**Verdict:** wedge = qualified win (6/9 metrics better; cleanliness/plausibility
up — CrossScore/NOVA/NVS-SQA/mLPIPS; consistency slightly down — TSED −0.030,
RAFT +0.005). easy = clean sweep (9/9). The wedge is **100% observed** (angular
gap, not a hole) → GT metrics were valid; June's number stands. Bidirectional
ArtiFixer3D confirmed as a **sharpen-observed** tool, NOT a completion-lane hole
filler. Baselines here are the ArtiFixer 3DGRUT recons — DISTINCT from the
`qa/Laptop*` production-splat baselines (different trainer/scene); do not
cross-compare absolute values.
