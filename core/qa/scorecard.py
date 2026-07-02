"""Per-scene QA scorecard (WP-Q1 comp.8): dataclass + deterministic JSON writer
+ an HTML strip for eyeballing, and the plug-in interface phase-2 scorers use.

The scorecard carries the phase-1 masked metrics plus a `scorers` registry dict
(crossscore / nova / tsed / raft_warp / nvs_sqa / flicker) that all default to
null.  A phase-2 scorer is any object implementing the `Scorer` protocol
(`.name` + `.score(ctx) -> dict`); the report driver calls each with a
`SceneContext` and stores the returned dict under `scorers[name]`.  Nothing about
the JSON schema changes when scorers are added — they only fill in the reserved
null slots.

Determinism: `write_json` sorts keys and rounds floats; NaN -> null.  Given the
same inputs the file is byte-identical run-to-run (a WP gate requirement).
"""
from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:                                        # Protocol is 3.8+, present on 3.11
    from typing import Protocol, runtime_checkable
except ImportError:                          # pragma: no cover
    Protocol = object

    def runtime_checkable(x):
        return x


# Reserved scorer slots — phase-2 fills these in; phase-1 leaves them null.
SCORER_REGISTRY: List[str] = [
    "crossscore",   # CrossScore  — predicted per-pixel SSIM (no-reference)
    "nova",         # NOVA        — cosine distance vs nearest real photo
    "tsed",         # TSED        — thresholded symmetric epipolar consistency ratio
    "raft_warp",    # RAFT        — optical-flow warp error (temporal, moving cam)
    "nvs_sqa",      # NVS-SQA     — learned JOD (within-scene)
    "flicker",      # VBench temporal flickering (static-camera clips)
]


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class ViewScore:
    index: int
    name: str
    observed_pct: float
    masked_psnr: float
    masked_ssim: float
    masked_lpips: float
    # relative paths (to the scorecard dir) of the eyeballing thumbnails
    render: str = ""
    gt: str = ""
    overlay: str = ""


@dataclass
class Scorecard:
    scene: str
    ckpt: str
    data_dir: str
    factor: int
    split: str
    n_views: int
    observed_area_pct: float
    masked_psnr: float
    masked_ssim: float
    masked_lpips: float
    alpha_thresh: float = 0.3
    # corruption knobs used (0 == pristine); recorded so runs are self-describing
    corruption: Dict[str, Any] = field(default_factory=dict)
    # reserved for phase-2 scorers; all null in phase-1
    scorers: Dict[str, Any] = field(
        default_factory=lambda: {k: None for k in SCORER_REGISTRY})
    views: List[ViewScore] = field(default_factory=list)

    def to_dict(self, round_to: int = 6) -> dict:
        return _sanitize(asdict(self), round_to)

    def write_json(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")


def _sanitize(obj, round_to):
    """Recursively round floats and turn NaN/Inf into None (valid, deterministic
    JSON). sort_keys handles dict ordering; lists keep their (sorted) order."""
    if isinstance(obj, dict):
        return {k: _sanitize(v, round_to) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, round_to) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, round_to)
    return obj


# ---------------------------------------------------------------------------
# Phase-2 plug-in interface
# ---------------------------------------------------------------------------

@dataclass
class SceneContext:
    """Everything a scorer needs, handed to `Scorer.score`.

    Phase-1 fills this in and passes it to every registered scorer.  A scorer may
    read the per-view thumbnails (`views[*].render/gt/overlay`) or, for metrics
    that need extra viewpoints (RAFT temporal, TSED pairs), render arbitrary poses
    itself from `splats` using `data` (poses/K/indices) + the project renderer.
    """
    scene: str
    ckpt: str
    data_dir: str
    factor: int
    sh_degree: int
    device: str
    out_dir: str
    data: dict                 # load_colmap_data(...) — poses, Ks, real image_paths, indices
    splats: dict               # loaded gaussian params, renderable via render_splats
    test_indices: List[int]    # sorted held-out indices scored in phase-1
    views: List[ViewScore]     # phase-1 per-view records (with thumbnail paths)
    # corruption knobs of THIS run (0/empty == pristine). Render-based scorers read
    # blur_sigma so blur-type corruption stays direction-correct (splats already
    # carry injected floaters); mirrors Scorecard.corruption.
    corruption: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Scorer(Protocol):
    """A phase-2 scorer. `name` MUST be one of SCORER_REGISTRY.

    `score` returns a JSON-serialisable dict (e.g. {"mean": ..., "per_view": [...],
    "map": "<relpath.png>"}) that is stored verbatim under scorers[name].
    """
    name: str

    def score(self, ctx: "SceneContext") -> dict:
        ...


def run_scorers(card: "Scorecard", ctx: "SceneContext", scorers,
                verbose: bool = True) -> None:
    """Run each scorer and merge its result into the reserved slot (the ONLY
    scorer-driving loop — qa_report.py calls this). Each result dict gets a
    "runtime_s". Failures are recorded (not fatal) so one missing checkpoint
    never sinks the scorecard."""
    import time
    for s in scorers:
        t0 = time.time()
        try:
            res = s.score(ctx)
            if isinstance(res, dict):
                res["runtime_s"] = round(time.time() - t0, 2)
            card.scorers[s.name] = res
            if verbose:
                head = ({k: v for k, v in res.items() if not isinstance(v, list)}
                        if isinstance(res, dict) else res)
                rt = res.get("runtime_s") if isinstance(res, dict) else None
                print(f"  [scorer] {s.name:11s} ok  ({rt}s)  -> {head}")
        except Exception as e:                        # pragma: no cover
            card.scorers[s.name] = {"error": f"{type(e).__name__}: {e}",
                                    "runtime_s": round(time.time() - t0, 2)}
            if verbose:
                print(f"  [scorer] {s.name:11s} FAILED  {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# HTML strip
# ---------------------------------------------------------------------------

def write_html(card: "Scorecard", path) -> None:
    """Self-contained HTML: header with aggregate metrics + a per-view table
    (render | mask-overlay | gt | masked metrics). Image paths are relative to
    the HTML file's directory."""
    def fmt(x, nd=4):
        return "n/a" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.{nd}f}"

    rows = []
    for v in card.views:
        rows.append(
            "<tr>"
            f"<td>{v.index}<br><span class=nm>{html.escape(v.name)}</span></td>"
            f"<td><img src='{html.escape(v.render)}'></td>"
            f"<td><img src='{html.escape(v.overlay)}'></td>"
            f"<td><img src='{html.escape(v.gt)}'></td>"
            f"<td class=num>{fmt(v.observed_pct, 1)}%</td>"
            f"<td class=num>{fmt(v.masked_psnr, 3)}</td>"
            f"<td class=num>{fmt(v.masked_ssim)}</td>"
            f"<td class=num>{fmt(v.masked_lpips)}</td>"
            "</tr>")

    corr = ", ".join(f"{k}={v}" for k, v in card.corruption.items() if v) or "none"
    doc = _HTML_TEMPLATE.format(
        scene=html.escape(card.scene),
        ckpt=html.escape(card.ckpt),
        split=html.escape(card.split),
        factor=card.factor,
        nviews=card.n_views,
        obs=fmt(card.observed_area_pct, 1),
        psnr=fmt(card.masked_psnr, 3),
        ssim=fmt(card.masked_ssim),
        lpips=fmt(card.masked_lpips),
        athresh=card.alpha_thresh,
        corruption=html.escape(corr),
        rows="\n".join(rows),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(doc, encoding="utf-8")


_HTML_TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>QA scorecard — {scene}</title>
<style>
 body{{margin:0;background:#141414;color:#ddd;font:14px system-ui}}
 header{{padding:14px 18px;background:#1d1d1d;border-bottom:1px solid #333}}
 h1{{margin:0 0 6px;font-size:18px}}
 .kv{{color:#9ad;}} .mut{{color:#888}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border-bottom:1px solid #2a2a2a;padding:6px 8px;vertical-align:top}}
 th{{position:sticky;top:0;background:#1d1d1d;text-align:left}}
 img{{max-width:300px;height:auto;display:block;background:#000;image-rendering:auto}}
 td.num{{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}
 .nm{{color:#777;font-size:11px}}
 .legend{{margin-top:6px;color:#c66}}
</style></head><body>
<header>
 <h1>QA scorecard — {scene}</h1>
 <div class=mut>ckpt <span class=kv>{ckpt}</span> · split <span class=kv>{split}</span>
  · factor <span class=kv>{factor}</span> · views <span class=kv>{nviews}</span>
  · alpha&lt;{athresh} = unobserved · corruption <span class=kv>{corruption}</span></div>
 <div style="margin-top:8px">
  observed area <b>{obs}%</b> &nbsp;|&nbsp;
  masked PSNR <b>{psnr}</b> &nbsp; SSIM <b>{ssim}</b> &nbsp;
  LPIPS <b>{lpips}</b> <span class=mut>(observed region only; lower LPIPS = better)</span></div>
 <div class=legend>overlay: red tint = UNOBSERVED (excluded from masked metrics)</div>
</header>
<table>
<tr><th>view</th><th>render</th><th>mask overlay</th><th>GT</th>
 <th>obs%</th><th>PSNR</th><th>SSIM</th><th>LPIPS</th></tr>
{rows}
</table></body></html>
"""
