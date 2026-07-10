"""gui2 — Deliver screen logic (UI-4, design §3④ + §4).

Pure, Gradio-free helpers so the whole Deliver screen is unit-testable and gateable
without a browser (mirrors the qc.py / review.py split — app.py is only the shell +
wiring). Everything here reads the on-disk artifacts UI-1..3 already produce:

  * the delivery record  <run_dir>/deliverables/<name>_<mode>_delivery.json
    (core/delivery.py schema: `deliverable`{n_gaussians,sh_degree,sog_mb,ply_mb},
     `qa`{delivered_object|delivered, spatial_floaters, ...}, `stages`{vote,prune,...})
  * the COLMAP model      <run_dir>/export/sparse/0  (registered-image count)
  * an optional QA scorecard  qa/wpd1_<name>_<object|full>/scorecard.json (TSED if run)

Four sections the app draws from this module:
  1. quality_card(run_dir)   -> 4 friendly numbers + honest flags (design §3④)
  2. deliverable_cards(...)  -> per-deliverable name/size/count/SH (+ viewer scene URL)
  3. coverage_report_for(...) -> "improve further" re-capture map (core.enhancement.coverage)
  4. cleanup_master(...)     -> "improve further" floater prune (core.training.prune)

The star-scale mapping (design §3④, banked object-region-LPIPS range) lives here as the
single source: <=0.18 excellent (5*) ... >=0.35 poor (1*), linear between; the raw number
is always surfaced for experts.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from urllib.parse import quote

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_DATASET = _REPO / "_dataset"

_IMG_EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# Star scale (design §3④): object-region LPIPS -> 5-star, from the banked range.
# <=0.18 excellent (5.0*)  ...  >=0.35 poor (1.0*).  Linear in between, clamped.
# Chosen so the delivered Sony shoe (obj-LPIPS 0.235) lands at ~3.7* = "Very good",
# matching the eyeball verdict, and the whole [0.18,0.35] band spans the 5..1 range.
# ---------------------------------------------------------------------------

LPIPS_EXCELLENT = 0.18      # <= this -> 5 stars
LPIPS_POOR = 0.35           # >= this -> 1 star


def lpips_to_stars(lpips: float | None) -> float | None:
    """Map an object-region / masked LPIPS to a [1.0, 5.0] star score (float)."""
    if lpips is None or not math.isfinite(float(lpips)):
        return None
    span = LPIPS_POOR - LPIPS_EXCELLENT
    stars = 5.0 - 4.0 * (float(lpips) - LPIPS_EXCELLENT) / span
    return max(1.0, min(5.0, stars))


def star_label(stars: float | None) -> str:
    if stars is None:
        return "not scored"
    if stars >= 4.5:
        return "Excellent"
    if stars >= 3.5:
        return "Very good"
    if stars >= 2.5:
        return "Good"
    if stars >= 1.5:
        return "Fair"
    return "Poor"


def star_string(stars: float | None) -> str:
    """★ full, ½ half, ☆ empty — padded to 5. Robust glyphs (no exotic half-star)."""
    if stars is None:
        return "—"
    full = int(math.floor(stars + 1e-6))
    half = 1 if (stars - full) >= 0.5 else 0
    full = min(full, 5)
    empty = max(0, 5 - full - half)
    return "★" * full + ("½" if half else "") + "☆" * empty


# ---------------------------------------------------------------------------
# On-disk readers
# ---------------------------------------------------------------------------

def _colmap_num_images(sparse_dir: Path) -> int | None:
    """Registered-image count of a COLMAP model (num_reg_images = images.bin leading
    uint64), falling back to counting images.txt entries. None if unreadable.
    (Same logic as app._colmap_num_images, replicated so this module stays
    Gradio-free / independently importable.)"""
    binp = sparse_dir / "images.bin"
    if binp.is_file():
        try:
            with open(binp, "rb") as f:
                return int(struct.unpack("<Q", f.read(8))[0])
        except Exception:
            return None
    txtp = sparse_dir / "images.txt"
    if txtp.is_file():
        try:
            n = sum(1 for ln in txtp.read_text().splitlines()
                    if ln.strip() and not ln.startswith("#"))
            return n // 2
        except Exception:
            return None
    return None


def _count_imgs(d: Path) -> int | None:
    if not d.is_dir():
        return None
    n = sum(1 for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    return n or None


def _pipeline_state_image_count(run_dir: str) -> int | None:
    """Input frame count from the project's pipeline_state.json image_list (present in
    both the folder-drop and video-extract flows). Last-resort total source."""
    for p in Path(run_dir).glob("apex_*/pipeline_state.json"):
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
            il = st.get("image_list")
            if il:
                return len(il)
        except Exception:
            continue
    return None


def _registered_total(run_dir: str) -> tuple[int | None, int | None]:
    """(registered, total) views for the run.

    Registered = COLMAP num_reg_images. Total = the number of frames HANDED to
    reconstruction — NOT the registered subset. Primary source is ``export/images/``
    (populated in BOTH the folder-drop and video-extract flows and holding every input,
    not just the registered ones); fall back to ``input_images/`` (only under the run
    dir when the runner extracted video frames) then to pipeline_state.json's
    image_list. ``total`` is left None if none of these are readable — it is NEVER
    defaulted to ``registered`` (that produced a confidently-wrong 100% for the
    external-folder-drop flow, where input_images does not live under the run dir)."""
    rd = Path(run_dir)
    reg = _colmap_num_images(rd / "export" / "sparse" / "0")
    total = (_count_imgs(rd / "export" / "images")
             or _count_imgs(rd / "input_images")
             or _pipeline_state_image_count(run_dir))
    return reg, total


def find_records(run_dir: str) -> list[dict]:
    """All delivery records under <run_dir>/deliverables/, newest first."""
    ddir = Path(run_dir) / "deliverables"
    if not ddir.is_dir():
        return []
    recs = []
    for p in sorted(ddir.glob("*_delivery.json"),
                    key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            recs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return recs


def primary_record(run_dir: str, mode: str | None = None) -> dict | None:
    """The record backing the quality card: the newest whose mode matches (or the
    newest of any mode when `mode` is None / no match)."""
    recs = find_records(run_dir)
    if not recs:
        return None
    if mode:
        for r in recs:
            if r.get("mode") == mode:
                return r
    return recs[0]


def _tsed_ratio(name: str, mode: str) -> float | None:
    """Consistent-pair ratio from a QA scorecard's TSED scorer, if one was ever run.
    delivery writes qa_report output under qa/wpd1_<name>_<object|full>/. TSED is a
    phase-2 scorer (null unless explicitly requested), so this is usually None and the
    card falls back to spatial compactness (design §3④)."""
    label = "object" if mode == "object" else "full"
    p = _REPO / "qa" / f"wpd1_{name}_{label}" / "scorecard.json"
    if not p.is_file():
        return None
    try:
        sc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    t = (sc.get("scorers") or {}).get("tsed")
    if not isinstance(t, dict):
        return None
    pairs = t.get("per_pair") or []
    if pairs:
        cons = sum(1 for pp in pairs if pp.get("consistent"))
        return cons / len(pairs)
    for k in ("consistent_ratio", "ratio", "consistent_frac"):
        if k in t:
            try:
                return float(t[k])
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Quality card (design §3④): 4 friendly numbers + honest flags
# ---------------------------------------------------------------------------

def quality_card(run_dir: str, mode: str | None = None) -> dict:
    """Map the delivery record `qa` block (+ a light recon count + optional scorecard)
    into the 4 friendly numbers the Deliver screen shows. Returns a structured dict;
    fields are None when their source is absent (the HTML renderer degrades cleanly)."""
    rec = primary_record(run_dir, mode)
    card: dict = {"has_record": rec is not None, "mode": mode}
    if rec is None:
        return card
    mode = rec.get("mode") or mode or "object"
    card["mode"] = mode
    card["name"] = rec.get("name")
    qa = rec.get("qa") or {}

    # --- 1) sharpness: object-region (or scene-masked) LPIPS -> stars ---
    if mode == "object":
        lp = ((qa.get("delivered_object") or {}).get("obj_masked_lpips"))
        sharp_kind = "region-LPIPS"
    else:
        lp = ((qa.get("delivered") or {}).get("masked_lpips"))
        sharp_kind = "masked-LPIPS"
    stars = lpips_to_stars(lp)
    card["sharpness"] = {
        "lpips": (float(lp) if lp is not None and math.isfinite(float(lp)) else None),
        "stars": stars, "stars_str": star_string(stars),
        "label": star_label(stars), "kind": sharp_kind,
        "title": "Object sharpness" if mode == "object" else "Scene sharpness",
    }

    # --- 2) geometric consistency: TSED if a scorecard has it, else compactness % ---
    tsed = _tsed_ratio(rec.get("name", ""), mode)
    floaters = qa.get("spatial_floaters") or {}
    if tsed is not None:
        card["geometry"] = {"kind": "tsed", "value_pct": round(100.0 * tsed, 1),
                            "text": f"{100.0 * tsed:.0f}% of view-pairs consistent (TSED)"}
    elif "within_0.5_scale_pct" in floaters:
        v = float(floaters["within_0.5_scale_pct"])
        card["geometry"] = {"kind": "compactness", "value_pct": round(v, 1),
                            "text": f"{v:.1f}% of Gaussians in the object core (compactness)"}
    else:
        card["geometry"] = {"kind": None, "value_pct": None, "text": "not scored"}

    # --- 3) floaters: object -> bg bleed alpha; env -> SOR removal % ---
    if mode == "object":
        do = qa.get("delivered_object") or {}
        bleed = do.get("mean_bg_bleed_alpha")
        beyond = floaters.get("beyond_0.5_scale_count")
        pct = (float(bleed) * 100.0 if bleed is not None
               and math.isfinite(float(bleed)) else None)
        if pct is None:
            flabel = "not scored"
        elif pct <= 3.0:
            flabel = "Clean"
        elif pct <= 8.0:
            flabel = "Minor bleed"
        else:
            flabel = "Noticeable bleed"
        card["floaters"] = {
            "kind": "bleed", "label": flabel, "bleed_pct": pct,
            "beyond_core": (int(beyond) if beyond is not None else None),
            "text": (f"{pct:.1f}% background bleed" if pct is not None else "not scored"),
        }
    else:
        prune = (rec.get("stages") or {}).get("prune") or {}
        nb = prune.get("n_before")
        removed = prune.get("removed_outliers")
        pct = (100.0 * removed / nb if nb else None) if removed is not None else None
        if removed is None:
            flabel = "Not scored"
        elif pct is not None and pct < 1.0:
            flabel = "Clean"
        else:
            flabel = "Cleaned"
        card["floaters"] = {
            "kind": "sor", "label": flabel,
            "removed": (int(removed) if removed is not None else None),
            "removed_pct": (round(pct, 2) if pct is not None else None),
            "text": (f"{removed:,} outliers removed (SOR)" if removed is not None
                     else "not scored"),
        }

    # --- 4) views used: registered / total ---
    reg, total = _registered_total(run_dir)
    card["views"] = {"registered": reg, "total": total,
                     "text": (f"{reg} of {total} frames registered"
                              if reg is not None and total is not None
                              else (f"{reg} registered" if reg is not None
                                    else "unknown"))}

    # --- honest flags (design §3④: keep simple; per-view PSNR spread not banked
    #     in the record, so the "underside weakly observed" flag is omitted rather
    #     than faked — flags derive only from fields that actually exist) ---
    flags: list[str] = []
    fl = card["floaters"]
    if fl["kind"] == "bleed" and fl.get("bleed_pct") is not None and fl["bleed_pct"] > 5.0:
        flags.append(f"Some background bleed remains at the object edges "
                     f"({fl['bleed_pct']:.1f}% out-of-mask alpha).")
    if fl.get("beyond_core") is not None and fl["beyond_core"] > 50:
        flags.append(f"{fl['beyond_core']} distant Gaussians sit beyond the object "
                     "core (possible floaters).")
    if reg is not None and total is not None and total - reg > 0:
        miss = total - reg
        if miss > max(1, int(0.02 * total)):
            flags.append(f"{miss} of {total} frames did not register — coverage may "
                         "be uneven where they were dropped.")
    card["flags"] = flags
    return card


# ---------------------------------------------------------------------------
# Quality-card HTML
# ---------------------------------------------------------------------------

def _metric_block(title: str, big: str, sub: str, raw: str = "") -> str:
    raw_html = f'<div class="qraw">{raw}</div>' if raw else ""
    return (f'<div class="qmetric"><div class="qlabel">{title}</div>'
            f'<div class="qbig">{big}</div><div class="qsub">{sub}</div>'
            f'{raw_html}</div>')


def quality_card_html(card: dict) -> str:
    if not card.get("has_record"):
        return ('<div class="apex-panel">Quality report appears here once delivery '
                'finishes (needs the delivery QA pass — leave it on).</div>')
    sh = card.get("sharpness") or {}
    ge = card.get("geometry") or {}
    fl = card.get("floaters") or {}
    vw = card.get("views") or {}

    sharp_raw = (f'{sh.get("kind","")} {sh["lpips"]:.3f} (lower is sharper)'
                 if sh.get("lpips") is not None else "not scored")
    blocks = [
        _metric_block(sh.get("title", "Sharpness"),
                      f'<span class="qstars">{sh.get("stars_str","—")}</span>',
                      sh.get("label", "not scored"), sharp_raw),
        _metric_block("Geometric consistency",
                      (f'{ge["value_pct"]:.0f}%' if ge.get("kind") == "tsed"
                       else (f'{ge["value_pct"]:.1f}%' if ge.get("value_pct") is not None
                             else "—")),
                      ge.get("text", "not scored")),
        _metric_block("Floaters", fl.get("label", "—"),
                      fl.get("text", "not scored"),
                      (f'{fl["beyond_core"]} beyond core'
                       if fl.get("beyond_core") not in (None, 0) else "")),
        _metric_block("Views used",
                      (f'{vw["registered"]}/{vw["total"]}'
                       if vw.get("registered") is not None and vw.get("total") is not None
                       else "—"),
                      vw.get("text", "unknown")),
    ]
    flags = card.get("flags") or []
    if flags:
        items = "".join(f"<li>{f}</li>" for f in flags)
        flags_html = (f'<div class="apex-flags"><b>Notes:</b><ul>{items}</ul></div>')
    else:
        flags_html = ('<div class="apex-flags apex-flags-ok">No quality concerns '
                      'flagged.</div>')
    return (f'<div class="apex-quality">{"".join(blocks)}</div>{flags_html}')


# ---------------------------------------------------------------------------
# Deliverable cards
# ---------------------------------------------------------------------------

def scene_url_for(path: str) -> str | None:
    """Same-origin /scenes URL for a file under _dataset (the UI-2 static mount),
    else None (only _dataset files are viewer-servable via /scenes)."""
    try:
        rel = Path(path).resolve().relative_to(_DATASET.resolve()).as_posix()
        return "/scenes/" + quote(rel)
    except ValueError:
        return None


def deliverable_cards(items: list[dict], run_dir: str) -> list[dict]:
    """Group the runner's flat deliverable list (name/path/size_mb per file) into one
    card per deliverable base (e.g. `<scan>_object`), attaching the record's Gaussian
    count + SH level + mode. For object mode this naturally yields separate cards for
    the standard and any detail / reviewed / cleaned variants present."""
    recs_by_base: dict[str, dict] = {}
    for r in find_records(run_dir):
        d = r.get("deliverable") or {}
        for key in ("sog", "ply"):
            v = d.get(key)
            if v:
                recs_by_base[Path(v).stem] = r
    groups: dict[str, dict] = {}
    for it in items:
        base = Path(it["name"]).stem
        g = groups.setdefault(base, {"base": base, "sog": None, "ply": None,
                                     "sog_mb": None, "ply_mb": None})
        ext = Path(it["name"]).suffix.lower()
        if ext == ".sog":
            g["sog"] = it["path"]
            g["sog_mb"] = it.get("size_mb")
        elif ext == ".ply":
            g["ply"] = it["path"]
            g["ply_mb"] = it.get("size_mb")
    cards = []
    for base, g in groups.items():
        rec = recs_by_base.get(base)
        dblk = (rec.get("deliverable") or {}) if rec else {}
        g["n_gaussians"] = dblk.get("n_gaussians")
        g["sh_degree"] = dblk.get("sh_degree")
        g["mode"] = (rec or {}).get("mode")
        g["scene_url"] = scene_url_for(g["sog"]) if g.get("sog") else None
        cards.append(g)
    # stable order: object variants first, then by name
    cards.sort(key=lambda c: (c.get("mode") != "object", c["base"]))
    return cards


def _pretty_variant(base: str) -> str:
    """A friendly label for a deliverable base like `sony_shoe_detail_object` ->
    'Object · detail'. Strip the mode suffix, then name any known variant token."""
    b = base
    tag = "Deliverable"
    for suffix, name in (("_object", "Object"), ("_full", "Full scene"),
                         ("_environment", "Full scene")):
        if b.endswith(suffix):
            tag = name
            b = b[: -len(suffix)]
            break
    tokens = b.split("_")
    variant = next((v for v in ("detail", "cleaned", "reviewed") if v in tokens), None)
    return f"{tag} · {variant}" if variant else tag


def deliverable_cards_html(cards: list[dict]) -> str:
    if not cards:
        return ('<div class="apex-panel">No deliverables yet. They appear here when '
                'processing finishes.</div>')
    ico = {"object": "\U0001F9F8", "full": "\U0001F3E0"}
    out = []
    for c in cards:
        glyph = ico.get(c.get("mode") or "", "\U0001F4E6")
        parts = []
        if c.get("sog_mb") is not None:
            parts.append(f'.sog {c["sog_mb"]} MB')
        if c.get("ply_mb") is not None:
            parts.append(f'.ply {c["ply_mb"]} MB')
        if c.get("n_gaussians") is not None:
            parts.append(f'{c["n_gaussians"]:,} Gaussians')
        if c.get("sh_degree") is not None:
            parts.append(f'SH {c["sh_degree"]}')
        sub = " &middot; ".join(parts) if parts else "—"
        out.append(
            f'<div class="card"><span class="ico">{glyph}</span>'
            f'<div><div class="name">{_pretty_variant(c["base"])} '
            f'<span class="muted">({c["base"]})</span></div>'
            f'<div class="sub">{sub}</div></div></div>')
    return f'<div class="apex-deliv">{"".join(out)}</div>'


# ---------------------------------------------------------------------------
# "Improve further" #2 — re-capture / coverage report (core.enhancement.coverage)
# ---------------------------------------------------------------------------

def coverage_report_for(data_dir: str, factor: int, out_png: str) -> tuple[str, str, float]:
    """Run the angular-coverage analysis on a run's cameras and render its map.
    Returns (human-readable report text, out_png path, coverage_fraction). CPU-only
    (poses + points; no GPU), so the 'improve further' re-capture button is cheap."""
    import numpy as np
    import cv2
    from core.training.gsplat_trainer import load_colmap_data
    from core.enhancement import coverage as CV

    data = load_colmap_data(data_dir, factor=factor, normalize=True)
    c2w = np.asarray(data["camtoworlds"], dtype=float)
    cams = c2w[:, :3, 3]
    forwards = c2w[:, :3, 2]                       # camera +z (OpenCV)
    pts = data.get("points")
    if pts is not None and len(pts):
        O = np.median(np.asarray(pts, dtype=float), axis=0)
    else:
        O = CV.object_center_from_cameras(cams, forwards)
    res = CV.analyze_coverage(cams, O)
    text = CV.coverage_report(res)
    img = CV.render_coverage_map(res)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), img)
    return text, str(out_png), float(res.coverage_pct)


def resolve_data_dir(path: str) -> str | None:
    """Accept either a training data dir (has sparse/) or a run dir (has
    training/gsplat/data) and return the COLMAP data dir, else None."""
    p = Path(path)
    if (p / "sparse").is_dir():
        return str(p)
    cand = p / "training" / "gsplat" / "data"
    if (cand / "sparse").is_dir():
        return str(cand)
    return None


# ---------------------------------------------------------------------------
# "Improve further" #3 — floater cleanup (core.training.prune) on the master
# ---------------------------------------------------------------------------

def resolve_reviewed_mask_dir(run_dir: str, log=None) -> str | None:
    """Return <run_dir>/masks_reviewed IFF UI-3 wrote a valid completion marker whose
    count matches the on-disk PNGs, else None (delivery then falls back to --gen_masks).

    Single source shared by the runner's delivery stage AND the Deliver-screen floater-
    cleanup re-delivery, so both honor reviewed masks identically. `log` is an optional
    one-arg line sink (e.g. runner._log_line)."""
    from gui2.review import MASKS_COMPLETE_MARKER
    _log = log or (lambda _m: None)
    reviewed = Path(run_dir) / "masks_reviewed"
    if not reviewed.is_dir():
        return None
    marker = reviewed / MASKS_COMPLETE_MARKER
    pngs = list(reviewed.glob("*.png"))
    if not marker.is_file():
        _log("[delivery] masks_reviewed present but no completion marker (partial or "
             "legacy accept); falling back to --gen_masks (BiRefNet)")
        return None
    try:
        want = int(marker.read_text().strip())
    except (ValueError, OSError):
        want = -1
    if pngs and want == len(pngs):
        _log(f"[delivery] using {len(pngs)} reviewed masks (accept marker ok): {reviewed}")
        return str(reviewed)
    _log(f"[delivery] masks_reviewed marker/count mismatch (marker={want}, "
         f"on-disk={len(pngs)}); falling back to --gen_masks (BiRefNet)")
    return None


def build_delivery_flags(settings: dict, run_dir: str, *,
                         delivery_refit: bool | None = None,
                         delivery_no_qa: bool = False, log=None) -> list[str]:
    """The full trailing delivery CLI flags (everything after --ckpt/--data_dir/--name/
    --factor/--out): the mode's assemble_delivery_flags with reviewed masks resolved,
    plus the per-run refit / no_qa choices. Single source for BOTH the runner's delivery
    stage and the floater-cleanup re-delivery so neither silently drops reviewed masks or
    the caller's refit choice."""
    from gui2 import defaults as D
    reviewed_dir = resolve_reviewed_mask_dir(run_dir, log=log)
    flags = list(D.assemble_delivery_flags(settings, reviewed_mask_dir=reviewed_dir))
    if delivery_refit is not None:
        flags.append("--refit" if delivery_refit else "--no-refit")
    if delivery_no_qa:
        flags.append("--no_qa")
    return flags


def cleanup_master(master_ckpt: str, work_dir: str, opacity_min: float = 0.02,
                   sor_std: float = 2.0, device: str = "cuda") -> dict:
    """Prune low-opacity + spatial-outlier Gaussians from the MASTER checkpoint
    (faithful: only removes). Writes a cleaned ckpt + PLY under work_dir so a
    re-delivery can run from the cleaned master. Returns prune stats (+ paths)."""
    from core.training.prune import prune_splats
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    out_ply = str(Path(work_dir) / "master_cleaned.ply")
    out_ckpt = str(Path(work_dir) / "master_cleaned.pt")
    stats = prune_splats(master_ckpt, out_ply, out_ckpt,
                         opacity_min=opacity_min, sor_std=sor_std, device=device)
    stats["cleaned_ckpt"] = out_ckpt
    stats["cleaned_ply"] = out_ply
    return stats
