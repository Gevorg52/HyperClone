# Reference source snapshots (fetched 2026-07-01 during execution-plan research)

**POC stance (user directive): licenses are NOT a filter — these snapshots are fair game to copy,
vendor, and adapt.** Log anything ported into product code in `apex-sfm/THIRD_PARTY.md`
(component → source → license) so the future funded rebuild knows what to re-license/re-implement.

Contents: `moge/` (MoGe-2 API: v2.py, README, LICENSE=MIT) · `dnsplatter/` (depth-alignment
reference, Apache) · `gs/` + `drgs/` (depth-scale references, Inria) · `edgs/` (configs + README;
clone the full repo for `source/corr_init.py` if the WP-F2 alternative route is taken — note its
H/W-swap bug #14) · `erank_gs/` (official eff-rank loss code) · `sags/` (SA-GS forward.cu +
render scripts — reference for the eps2d rule; porting its CUDA is unnecessary) · `ppisp/` +
`ppisp_files/` + `gsplat_pr858.diff` (Apache — the WP-F5 wiring reference) · `bilagrid_*`
(fused-bilagrid, MIT) · `paper.txt` (In-Depth-We-Trust full text, arXiv 2604.05715 — no code
exists anywhere; implement from these formulas).
