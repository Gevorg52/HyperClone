"""WP-O1 gate evidence collector — copies/derives the committable artifacts from the
(gitignored) ingest output dirs into `_infusion/wpo1/`.

    python _infusion/wpo1_evidence.py

Reads   _dataset/360/{flowershop_077,theater_027_8k,flowershop_mov,erp_path_check}
Writes  _infusion/wpo1/*.jpg  (contact sheets, sharpness strips, 8K samples + native crops)
        _infusion/wpo1/report_*.json      (full ingest reports, incl. argv)
        _infusion/wpo1/integrity.json     (programmatic mask/image conformance checks)

NOTE `_infusion/.gitignore` ignores *.png — every image written here is JPEG on purpose.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_infusion" / "wpo1"
RUNS = {
    "077": ROOT / "_dataset/360/flowershop_077",
    "027_8k": ROOT / "_dataset/360/theater_027_8k",
    "mov": ROOT / "_dataset/360/flowershop_mov",
    "erp_path": ROOT / "_dataset/360/erp_path_check",
}


def integrity(run: Path) -> dict:
    """Verify the output layout the rest of the pipeline consumes."""
    imgs = sorted((run / "images").glob("*.jpg"))
    masks = sorted((run / "masks").glob("*.png"))
    vals, bad_shape, mask_dtype = set(), 0, set()
    for ip, mp in zip(imgs, masks):
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        mask_dtype.add(str(m.dtype))
        vals |= set(np.unique(m).tolist())
        h, w = cv2.imread(str(ip), cv2.IMREAD_REDUCED_COLOR_8).shape[:2]
        if (m.shape[0] // 8, m.shape[1] // 8) != (h, w):
            bad_shape += 1
    return {
        "n_images": len(imgs), "n_masks": len(masks),
        "stems_pair_1to1": [p.stem for p in imgs] == [p.stem for p in masks],
        "mask_dtypes": sorted(mask_dtype),
        "mask_values": sorted(vals),
        "mask_shape_mismatches": bad_shape,
        "image_size": list(cv2.imread(str(imgs[0])).shape[:2][::-1]) if imgs else None,
        "images_mb": round(sum(p.stat().st_size for p in imgs) / 1e6, 1),
    }


def operator_coverage(run: Path, frames: list) -> dict:
    """Excluded/operator-only fractions over the contact-sheet frames."""
    rep = json.loads((run / "ingest_report.json").read_text())
    sel = [rep["frames"][i] for i in frames] if frames else rep["frames"]
    ex = np.array([f["excluded_frac"] for f in sel])
    op = np.array([f["operator_frac"] for f in sel])
    return {"n": len(sel),
            "excluded_frac": {"min": round(float(ex.min()), 4), "median": round(float(np.median(ex)), 4),
                              "max": round(float(ex.max()), 4)},
            "operator_frac": {"min": round(float(op.min()), 4), "median": round(float(np.median(op)), 4),
                              "max": round(float(op.max()), 4)},
            "operator_sources": {s: sum(1 for f in sel if f["operator_source"] == s)
                                 for s in sorted({f["operator_source"] for f in sel})}}


def zoom_sheet(run: Path, idxs: list, out: Path, cols: int = 3, tile_w: int = 760) -> None:
    """Mask-overlay zoom on the rear hemisphere (x 0.28-0.72, y 0.33-1.0) — the region the
    operator can occupy. Makes per-frame operator coverage visually checkable."""
    tiles = []
    for i in idxs:
        img = cv2.imread(str(run / "images" / f"pano_{i:05d}.jpg"))
        m = cv2.imread(str(run / "masks" / f"pano_{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]
        c = img[int(H * .33):, int(W * .28):int(W * .72)].copy()
        mc = m[int(H * .33):, int(W * .28):int(W * .72)]
        ov = c.copy(); ov[mc == 0] = (0, 0, 255)
        c = cv2.addWeighted(c, .5, ov, .5, 0)
        c = cv2.resize(c, (tile_w, int(tile_w * c.shape[0] / c.shape[1])))
        cv2.putText(c, f"pano_{i:05d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        tiles.append(c)
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(str(out), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tag, run in RUNS.items():
        if not run.is_dir():
            print(f"MISSING run dir: {run}")
            return 1
        rep = json.loads((run / "ingest_report.json").read_text())
        (OUT / f"report_{tag}.json").write_text(json.dumps(rep, indent=1), encoding="utf8")
        cs = run / "qa" / "contact_sheet.jpg"
        if cs.is_file():
            cv2.imwrite(str(OUT / f"contact_{tag}.jpg"), cv2.imread(str(cs)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 84])
        sp = run / "qa" / "sharpness.png"
        if sp.is_file():                              # *.png is gitignored here -> re-encode
            cv2.imwrite(str(OUT / f"sharpness_{tag}.jpg"), cv2.imread(str(sp)),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        summary[tag] = {
            "source": rep["source"]["path"],
            "argv": rep["argv"],
            "layout": rep["source"]["layout"],
            "erp_size": [rep["stitch"]["out_width"], rep["stitch"]["out_height"]],
            "n_selected": rep["sampling"]["n_selected"],
            "effective_pano_fps": rep["sampling"]["effective_pano_fps"],
            "sharpness": rep["sharpness"]["all"] | {
                "selected_median": rep["sharpness"]["selected"]["median"],
                "selected_p10": rep["sharpness"]["selected"]["p10"],
                "selected_median_gain_pct": rep["sharpness"]["selected_median_gain_pct"]},
            "masks": {k: rep["masks"][k] for k in
                      ("operator_mode", "nadir_deg", "seam_deg", "geometric_excluded_frac",
                       "mean_excluded_frac", "operator_area_guard_frames",
                       "operator_lost_guard_frames")},
            "timing_s": rep["timing_s"],
            "integrity": integrity(run),
            "contact_sheet_coverage": operator_coverage(run, rep["qa"]["contact_sheet_frames"]),
        }

    # 8K proof: 3 sample frames (downscaled overview) + native-resolution 1:1 crops
    r8 = RUNS["027_8k"]
    picks = json.loads((r8 / "ingest_report.json").read_text())["qa"]["contact_sheet_frames"][:3]
    for k, i in enumerate(picks):
        img = cv2.imread(str(r8 / "images" / f"pano_{i:05d}.jpg"))
        H, W = img.shape[:2]
        cv2.imwrite(str(OUT / f"sample8k_{k}_overview_{W}x{H}.jpg"),
                    cv2.resize(img, (1600, 800), interpolation=cv2.INTER_AREA),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        cv2.imwrite(str(OUT / f"sample8k_{k}_crop1to1.jpg"),
                    img[int(H * .40):int(H * .40) + 768, int(W * .78):int(W * .78) + 1024],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    zoom_sheet(RUNS["077"], [2, 81, 92, 107, 114, 150], OUT / "zoom_operator_077.jpg")
    (OUT / "integrity.json").write_text(json.dumps(summary, indent=1), encoding="utf8")
    print(json.dumps({k: {"n": v["n_selected"], "erp": v["erp_size"],
                          "integrity": v["integrity"]} for k, v in summary.items()}, indent=1))
    print(f"\nevidence -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
