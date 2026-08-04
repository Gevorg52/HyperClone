"""WP-O1 — 360° capture ingest (Insta360 dual-fisheye or pre-stitched ERP) -> images/ + masks/.

Turns ONE 360° video file into the plain image folder the rest of the pipeline already
understands, plus a per-frame mask in the existing mask convention and a JSON report.

    <out_dir>/images/pano_00000.jpg      q95 equirectangular frames
    <out_dir>/masks/pano_00000.png       uint8, 255 = VALID, 0 = EXCLUDED (repo convention,
                                         see core/training/gsplat_trainer.py:421-428)
    <out_dir>/ingest_report.json         probe + stitch params + per-frame sharpness/selection
    <out_dir>/qa/contact_sheet.jpg       mask-overlay QA sheet (--contact_sheet N)

DEFAULT-OFF: the module does nothing unless `--capture-type insta360` is passed. No
production module imports it; production behaviour is unchanged with the flag off.

Pipeline (docs/PLAN_360.md §5 WP-O1):

  1. PROBE (ffprobe) -> layout
       2 square video streams          -> dual_fisheye (in-camera .insv/.mov)
       1 video stream at 2:1           -> erp (pre-stitched, stitching skipped)
       anything else                   -> hard error
  2. SAMPLE — window-relative max-sharpness. The clip is cut into fixed windows of
     round(fps / --target_fps) frames and the sharpest frame of each window is kept
     (var-of-Laplacian on the central crop of the SCENE-facing lens, decoded at reduced
     size through one raw-gray ffmpeg pipe). There is NO absolute sharpness threshold:
     DEVELOPER_NOTES §5 records that absolute culls conflate "dark/low-texture" with
     "blurry" and are invalid for video. Window selection reuses core.frame_select.
  3. STITCH — one ffmpeg pass, `select` BEFORE `v360` so the expensive remap only runs on
     the selected frames:  [0:0][0:1]hstack -> v360=dfisheye:e:ih_fov=200:iv_fov=200
     (validated 2026-08-04, PLAN_360.md §2). Resolution-agnostic: output defaults to
     2*input_height x input_height, so 2x1920² -> 3840x1920 and 2x3840² -> 7680x3840.
  4. MASK — per selected frame, one uint8 PNG compositing
       a. operator   : SAM 2.1 video propagation on the rear-lens region. With our stitch
                       the REAR lens (the one that sees the operator) lands at ERP CENTRE
                       (verified on frames t=2/20/60 s of _077), so the tracker is prompted
                       at ERP centre and its output is clamped to the rear hemisphere.
       b. nadir band : bottom --nadir_deg degrees (stick / hand / feet).
       c. seam bands : vertical bands at the +-90° yaw lens boundaries (x = W/4, 3W/4),
                       +-  --seam_deg degrees.

Geometry conventions used throughout (ERP, W x H):
    yaw(x)   = (x/W - 0.5) * 360      (-180 at left edge, 0 at centre, +180 at right edge)
    pitch(y) = (0.5 - y/H) * 180      (+90 zenith at the top row, -90 nadir at the bottom)

CLI:
    python -m core.ingest_360 --capture-type insta360 \
        --input D:/HYPERCLONE/FOOTAGE/FlowerShop/VID_20260721_133413_00_077.insv \
        --out_dir _dataset/360/flowershop_077 --target_fps 2.0 --contact_sheet 20
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from core.frame_select import laplacian_var, select_indices

FFMPEG = os.environ.get("APEX_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("APEX_FFPROBE", "ffprobe")

VALID = 255       # mask value for pixels the pipeline may use
EXCLUDED = 0      # mask value for pixels every downstream stage must ignore


# ---------------------------------------------------------------------------
# 1. probe
# ---------------------------------------------------------------------------

@dataclass
class SourceInfo:
    path: str
    layout: str                       # "dual_fisheye" | "erp"
    width: int                        # per-stream width
    height: int                       # per-stream height
    fps: float
    fps_frac: str
    nb_frames: int                    # from the container (may be approximate)
    duration: float
    container: str = ""
    streams: list = field(default_factory=list)

    @property
    def erp_size(self) -> tuple:
        """Default stitched output size: 2*H x H for dual fisheye, native for ERP."""
        if self.layout == "dual_fisheye":
            return (2 * self.height, self.height)
        return (self.width, self.height)


def probe_source(path: str, force_layout: str | None = None) -> SourceInfo:
    """ffprobe the file and decide the layout. Raises ValueError on anything unsupported."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{out.stderr[-2000:]}")
    probe = json.loads(out.stdout)
    vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise ValueError(f"{path}: no video stream found")

    keep = ("index", "codec_name", "width", "height", "r_frame_rate", "nb_frames",
            "duration", "pix_fmt", "bit_rate")
    slim = [{k: s.get(k) for k in keep} for s in vstreams]

    w = int(vstreams[0]["width"]); h = int(vstreams[0]["height"])
    layout = force_layout
    if layout is None:
        square = [abs(int(s["width"]) - int(s["height"])) <= 2 for s in vstreams]
        same = all(int(s["width"]) == w and int(s["height"]) == h for s in vstreams)
        if len(vstreams) == 2 and all(square) and same:
            layout = "dual_fisheye"
        elif len(vstreams) == 1 and abs(w / max(h, 1) - 2.0) < 0.02:
            layout = "erp"
        else:
            raise ValueError(
                f"{path}: unsupported 360 layout - {len(vstreams)} video stream(s), sizes "
                + ", ".join(f"{s['width']}x{s['height']}" for s in vstreams)
                + ". Expected 2 equal SQUARE streams (in-camera dual fisheye) or 1 stream at "
                  "2:1 aspect (pre-stitched equirectangular). Override with --force_layout.")

    fps_frac = vstreams[0].get("r_frame_rate") or "30/1"
    fps = float(Fraction(fps_frac))
    dur = float(vstreams[0].get("duration") or probe.get("format", {}).get("duration") or 0.0)
    nbf = int(vstreams[0].get("nb_frames") or round(dur * fps))
    return SourceInfo(path=path, layout=layout, width=w, height=h, fps=fps, fps_frac=fps_frac,
                      nb_frames=nbf, duration=dur,
                      container=probe.get("format", {}).get("format_name", ""), streams=slim)


# ---------------------------------------------------------------------------
# 2. sampling — window-relative max sharpness (NEVER an absolute threshold)
# ---------------------------------------------------------------------------

def _run_pipe(cmd: list, frame_bytes: int, on_frame, label: str, progress: int = 250):
    """Stream rawvideo frames out of ffmpeg, calling on_frame(idx, buf). Returns count."""
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf)
        i = 0
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                on_frame(i, buf)
                i += 1
                if progress and i % progress == 0:
                    print(f"  [{label}] {i} frames", flush=True)
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0:
            errf.seek(0)
            raise RuntimeError(f"ffmpeg ({label}) exited {rc}:\n"
                               f"{errf.read().decode('utf8', 'replace')[-2000:]}")
    return i


def scan_sharpness(info: SourceInfo, stream: int = 0, central_frac: float = 0.5,
                   max_side: int = 960, progress: int = 250) -> np.ndarray:
    """var-of-Laplacian per source frame, on the central crop of one lens/stream.

    Decoding one stream at a reduced central crop is ~13 s for 2255 frames of 1920².
    Values are comparable only WITHIN a clip (they depend on crop size) — which is all
    window-relative selection needs.
    """
    cw = max(16, int(round(info.width * central_frac)) // 2 * 2)
    ch = max(16, int(round(info.height * central_frac)) // 2 * 2)
    x0 = (info.width - cw) // 2
    y0 = (info.height - ch) // 2
    sw, sh = cw, ch
    if max(cw, ch) > max_side:
        s = max_side / max(cw, ch)
        sw = max(16, int(round(cw * s)) // 2 * 2)
        sh = max(16, int(round(ch * s)) // 2 * 2)
    vf = f"crop={cw}:{ch}:{x0}:{y0}"
    if (sw, sh) != (cw, ch):
        vf += f",scale={sw}:{sh}:flags=area"

    cmd = [FFMPEG, "-v", "error", "-i", info.path, "-map", f"0:{stream}", "-vf", vf,
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    vals: list = []
    _run_pipe(cmd, sw * sh,
              lambda i, b: vals.append(laplacian_var(
                  np.frombuffer(b, np.uint8).reshape(sh, sw))),
              "sharpness", progress)
    return np.asarray(vals, dtype=np.float64)


def window_size(fps: float, target_fps: float) -> int:
    return max(1, int(round(fps / max(target_fps, 1e-6))))


# ---------------------------------------------------------------------------
# 3. stitch / extract — one pass, select BEFORE v360
# ---------------------------------------------------------------------------

def build_filter_graph(info: SourceInfo, indices: list, out_w: int, out_h: int,
                       ih_fov: float, iv_fov: float, yaw: float) -> str:
    sel = "+".join(f"eq(n,{i})" for i in indices)
    if info.layout == "dual_fisheye":
        return (f"[0:0][0:1]hstack=inputs=2[hs];"
                f"[hs]select='{sel}'[sel];"
                f"[sel]v360=dfisheye:e:ih_fov={ih_fov}:iv_fov={iv_fov}:yaw={yaw}"
                f":w={out_w}:h={out_h}[out]")
    graph = f"[0:0]select='{sel}'[sel];"
    if yaw:
        graph += (f"[sel]v360=e:e:yaw={yaw}:w={out_w}:h={out_h}[out]")
    elif (out_w, out_h) != (info.width, info.height):
        graph += f"[sel]scale={out_w}:{out_h}:flags=lanczos[out]"
    else:
        graph += "[sel]null[out]"
    return graph


def extract_panos(info: SourceInfo, indices: list, images_dir: Path, out_w: int, out_h: int,
                  ih_fov: float, iv_fov: float, yaw: float, jpeg_quality: int = 95,
                  sam_dir: Path | None = None, sam_width: int = 1024,
                  progress: int = 25) -> list:
    """Stitch+write ONLY the selected frames. Returns the written image paths.

    Frames come back over a rawvideo pipe so JPEG quality is ours (cv2, q=95) rather than
    ffmpeg's mjpeg qscale, and the SAM-input downscale is produced in the same pass.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    if sam_dir is not None:
        sam_dir.mkdir(parents=True, exist_ok=True)
    graph = build_filter_graph(info, indices, out_w, out_h, ih_fov, iv_fov, yaw)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(graph)
        graph_file = fh.name
    paths: list = []
    sam_h = max(2, int(round(sam_width * out_h / out_w)) // 2 * 2)

    def on_frame(i, buf):
        img = np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3)
        p = images_dir / f"pano_{i:05d}.jpg"
        cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        paths.append(p)
        if sam_dir is not None:
            small = cv2.resize(img, (sam_width, sam_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(sam_dir / f"{i:05d}.jpg"), small,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    try:
        cmd = [FFMPEG, "-v", "error", "-i", info.path, "-filter_complex_script", graph_file,
               "-map", "[out]", "-fps_mode", "passthrough", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-"]
        n = _run_pipe(cmd, out_w * out_h * 3, on_frame, "stitch", progress)
    finally:
        os.unlink(graph_file)
    if n != len(indices):
        raise RuntimeError(f"stitch produced {n} frames, expected {len(indices)}")
    return paths


# ---------------------------------------------------------------------------
# 4. masks
# ---------------------------------------------------------------------------

def geometric_mask(w: int, h: int, nadir_deg: float = 12.0, zenith_deg: float = 0.0,
                   seam_deg: float = 4.0, seam_yaws=(-90.0, 90.0)) -> np.ndarray:
    """uint8 ERP mask, VALID everywhere except the nadir/zenith bands and seam bands."""
    m = np.full((h, w), VALID, np.uint8)
    if nadir_deg > 0:
        m[int(round(h * (1.0 - nadir_deg / 180.0))):, :] = EXCLUDED
    if zenith_deg > 0:
        m[:int(round(h * (zenith_deg / 180.0))), :] = EXCLUDED
    if seam_deg > 0:
        half = w * seam_deg / 360.0
        for yaw in seam_yaws:
            cx = w * (0.5 + yaw / 360.0)
            a = int(round(cx - half)); b = int(round(cx + half))
            for xs in range(a, b):
                m[:, xs % w] = EXCLUDED
    return m


def operator_region(w: int, h: int, max_yaw_deg: float = 90.0,
                    max_pitch_deg: float = 25.0) -> np.ndarray:
    """Bool clamp: where the operator CAN be (rear hemisphere, not up in the ceiling).

    With our dfisheye stitch the rear lens is centred at yaw 0, so the operator is bounded
    by |yaw| <= max_yaw_deg and pitch <= max_pitch_deg. Keeps a runaway tracker from
    swallowing the scene.
    """
    reg = np.zeros((h, w), bool)
    x0 = int(round(w * (0.5 - max_yaw_deg / 360.0)))
    x1 = int(round(w * (0.5 + max_yaw_deg / 360.0)))
    y0 = int(round(h * (0.5 - max_pitch_deg / 180.0)))
    reg[max(y0, 0):, max(x0, 0):min(x1, w)] = True
    return reg


def box_mask(w: int, h: int, box) -> np.ndarray:
    """Bool mask from a normalised (x0,x1,y0,y1) box — the guaranteed-core operator body."""
    m = np.zeros((h, w), bool)
    x0, x1, y0, y1 = box
    m[int(round(y0 * h)):int(round(y1 * h)), int(round(x0 * w)):int(round(x1 * w))] = True
    return m


def _parse_points(spec: str) -> np.ndarray:
    """'x,y;x,y' normalised -> (N,2) float array. Empty string -> (0,2)."""
    pts = [p for p in spec.split(";") if p.strip()]
    if not pts:
        return np.zeros((0, 2), np.float32)
    return np.array([[float(v) for v in p.split(",")] for p in pts], np.float32)


def sam2_operator_masks(sam_dir: Path, n_frames: int, pos: np.ndarray, neg: np.ndarray,
                        checkpoint: str = "sam2.1_hiera_large", device: str = "cuda"):
    """SAM 2.1 VIDEO propagation of the operator across the selected ERP frames.

    The selected frames are treated as the video (they are 0.5 s apart but the operator is
    nearly static in image space — they hold the camera — so propagation is well posed).
    Returns {frame_idx: bool mask at the SAM working resolution} or None if SAM 2.1 is
    unavailable / fails (caller falls back to the geometric box).
    """
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        from core.masking import _sam2_resolve_path
    except Exception as e:                                   # pragma: no cover - env dependent
        print(f"  [operator] SAM 2.1 unavailable ({e}) — falling back to the box", flush=True)
        return None

    cfg_map = {
        "sam2.1_hiera_large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
        "sam2.1_hiera_base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
        "sam2.1_hiera_small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
        "sam2.1_hiera_tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
    }
    if checkpoint not in cfg_map:
        raise ValueError(f"unknown SAM 2.1 checkpoint {checkpoint}")
    cfg, ckpt = cfg_map[checkpoint]
    ckpt = _sam2_resolve_path(ckpt)
    if not os.path.isfile(ckpt):
        print(f"  [operator] SAM 2.1 checkpoint missing: {ckpt} — falling back to the box")
        return None

    probe = cv2.imread(str(sam_dir / "00000.jpg"))
    sh, sw = probe.shape[:2]
    points = np.concatenate([pos, neg], 0) * np.array([sw, sh], np.float32)
    labels = np.concatenate([np.ones(len(pos), np.int32), np.zeros(len(neg), np.int32)])

    print(f"  [operator] SAM 2.1 video propagation over {n_frames} frames "
          f"({sw}x{sh}, {len(pos)} pos / {len(neg)} neg prompts)", flush=True)
    predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
    out = {}
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(sam_dir), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                        points=points, labels=labels)
        for idx, _ids, logits in predictor.propagate_in_video(state):
            out[idx] = (logits[0, 0] > 0.0).cpu().numpy()
    del predictor, state
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# QA contact sheet
# ---------------------------------------------------------------------------

def make_contact_sheet(image_paths: list, mask_paths: list, out_path: Path, n: int = 20,
                       seed: int = 0, cols: int = 4, tile_w: int = 640) -> list:
    """Grid of n RANDOM frames with the excluded region tinted red. Returns the picks."""
    rng = np.random.default_rng(seed)
    n = min(n, len(image_paths))
    picks = sorted(rng.choice(len(image_paths), size=n, replace=False).tolist())
    tiles = []
    for i in picks:
        img = cv2.imread(str(image_paths[i]))
        m = cv2.imread(str(mask_paths[i]), cv2.IMREAD_GRAYSCALE)
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        ov = img.copy()
        ov[m == EXCLUDED] = (0, 0, 255)
        img = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
        th = max(2, int(round(tile_w * img.shape[0] / img.shape[1])))
        img = cv2.resize(img, (tile_w, th), interpolation=cv2.INTER_AREA)
        cv2.putText(img, Path(image_paths[i]).stem, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    rows = []
    for r in range(0, len(tiles), cols):
        row = tiles[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return picks


def sharpness_plot(sharp: np.ndarray, selected: list, out_path: Path,
                   w: int = 1400, h: int = 320) -> None:
    """Per-frame sharpness strip, selected frames in green (same idea as frame_select)."""
    img = np.full((h, w, 3), 28, np.uint8)
    if len(sharp):
        top = np.percentile(sharp, 98) + 1e-6
        sn = np.clip(sharp / top, 0, 1)
        sel = set(selected)
        for i, v in enumerate(sn):
            x = int(i / max(len(sn) - 1, 1) * (w - 1))
            y = int(h - v * (h - 40))
            cv2.line(img, (x, h), (x, y), (90, 220, 90) if i in sel else (75, 75, 75), 1)
        cv2.putText(img, f"var-of-Laplacian per source frame (green = selected)  "
                         f"median {np.median(sharp):.0f}  p10 {np.percentile(sharp, 10):.0f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def ingest(args) -> dict:
    t_all = time.time()
    # --- probe (before touching out_dir, so a bad input leaves no debris) ---
    info = probe_source(args.input, args.force_layout)
    print(f"[probe] {Path(info.path).name}: layout={info.layout} "
          f"{info.width}x{info.height} @ {info.fps:.3f} fps, {info.nb_frames} frames, "
          f"{info.duration:.1f} s, {len(info.streams)} video stream(s)", flush=True)

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    qa_dir = out_dir / "qa"
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_w = args.out_width or info.erp_size[0]
    out_h = args.out_height or info.erp_size[1]
    seam_deg = args.seam_deg if args.seam_deg is not None else (
        4.0 if info.layout == "dual_fisheye" else 0.0)
    op_mode = args.operator_mask
    if op_mode == "auto":
        op_mode = "sam" if info.layout == "dual_fisheye" else "none"

    # --- sample ------------------------------------------------------------
    t0 = time.time()
    sstream = args.sharpness_stream if info.layout == "dual_fisheye" else 0
    sharp = scan_sharpness(info, stream=sstream, central_frac=args.central_frac,
                           max_side=args.sharpness_max_side, progress=args.progress)
    t_scan = time.time() - t0
    win = window_size(info.fps, args.target_fps)
    # window-relative maximum ONLY — no absolute threshold (DEVELOPER_NOTES §5)
    idx = select_indices(sharp, window=win, min_gap=1, keep_frac=1.0)
    print(f"[sample] {len(sharp)} frames scanned in {t_scan:.1f} s; window {win} frames "
          f"(~{info.fps/win:.2f} pano-fps) -> {len(idx)} selected", flush=True)
    print(f"[sample] sharpness all: min {sharp.min():.0f} p10 {np.percentile(sharp,10):.0f} "
          f"median {np.median(sharp):.0f} p90 {np.percentile(sharp,90):.0f} max {sharp.max():.0f}",
          flush=True)
    ssel = sharp[idx]
    print(f"[sample] sharpness selected: p10 {np.percentile(ssel,10):.0f} "
          f"median {np.median(ssel):.0f} (+{100*(np.median(ssel)/np.median(sharp)-1):.0f}% "
          f"vs all-frame median)", flush=True)
    if args.max_frames and len(idx) > args.max_frames:
        idx = idx[:args.max_frames]
        print(f"[sample] truncated to --max_frames {args.max_frames}", flush=True)

    # --- stitch ------------------------------------------------------------
    t0 = time.time()
    sam_dir = out_dir / "_sam_frames" if op_mode == "sam" else None
    paths = extract_panos(info, idx, images_dir, out_w, out_h, args.ih_fov, args.iv_fov,
                          args.yaw, args.jpeg_quality, sam_dir, args.operator_sam_width,
                          args.progress)
    t_stitch = time.time() - t0
    print(f"[stitch] {len(paths)} ERP frames at {out_w}x{out_h} in {t_stitch:.1f} s", flush=True)

    # --- masks -------------------------------------------------------------
    t0 = time.time()
    geo = geometric_mask(out_w, out_h, args.nadir_deg, args.zenith_deg, seam_deg)
    region = operator_region(out_w, out_h, args.operator_max_yaw_deg, args.operator_max_pitch_deg)
    core_box = None if op_mode == "none" else box_mask(out_w, out_h, args.operator_box)
    sam_out = None
    if op_mode == "sam":
        sam_out = sam2_operator_masks(sam_dir, len(paths),
                                      _parse_points(args.operator_pos),
                                      _parse_points(args.operator_neg),
                                      args.sam_checkpoint, args.device)
        if sam_out is None:
            op_mode = "box"

    masks_dir.mkdir(parents=True, exist_ok=True)
    kern = None
    if args.operator_dilate_px > 0:
        k = 2 * int(args.operator_dilate_px) + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    n_area_guard = n_lost_guard = 0
    per_frame = []
    for i, p in enumerate(paths):
        op = np.zeros((out_h, out_w), bool)
        # The core box is a SAFETY NET: unioned on every frame only with
        # --operator_box_union 1, otherwise only when SAM is absent or its guards fire.
        use_box = core_box is not None and (bool(args.operator_box_union) or sam_out is None)
        src = "none"
        if sam_out is not None and i in sam_out:
            s = sam_out[i].astype(np.uint8)
            s = cv2.resize(s, (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            s &= region
            frac = float(s.mean())
            if frac > args.operator_max_area_frac:
                n_area_guard += 1                      # runaway tracker -> box only
                s = np.zeros_like(s)
                use_box = True
                src = "box(area-guard)"
            elif frac < args.operator_min_area_frac:
                n_lost_guard += 1                      # tracking lost -> box carries it
                use_box = True
                src = "sam+box(lost)"
            else:
                src = "sam"
            op |= s
        if use_box:
            op |= core_box
            src = "box" if src == "none" else (f"{src}+box" if src == "sam" else src)
        if kern is not None and op.any():
            op = cv2.dilate(op.astype(np.uint8), kern).astype(bool)
            op &= region
        m = geo.copy()
        m[op] = EXCLUDED
        mp = masks_dir / (p.stem + ".png")
        cv2.imwrite(str(mp), m)
        per_frame.append({
            "i": i,
            "src_frame": int(idx[i]),
            "timestamp_s": round(idx[i] / info.fps, 4),
            "sharpness": round(float(sharp[idx[i]]), 2),
            "window": i,
            "window_frames": [int(i * win), int(min((i + 1) * win, len(sharp)) - 1)],
            "image": f"images/{p.name}",
            "mask": f"masks/{mp.name}",
            "excluded_frac": round(float((m == EXCLUDED).mean()), 5),
            "operator_frac": round(float(op.mean()), 5),
            "operator_source": src,
        })
    t_mask = time.time() - t0
    print(f"[mask] {len(paths)} masks in {t_mask:.1f} s "
          f"(operator={op_mode}, area-guard {n_area_guard}, lost-guard {n_lost_guard}); "
          f"mean excluded {np.mean([f['excluded_frac'] for f in per_frame])*100:.2f}%",
          flush=True)
    if sam_dir is not None and not args.keep_sam_frames:
        shutil.rmtree(sam_dir, ignore_errors=True)

    # --- QA + report -------------------------------------------------------
    picks = []
    if args.contact_sheet:
        picks = make_contact_sheet(paths,
                                   [masks_dir / (p.stem + ".png") for p in paths],
                                   qa_dir / "contact_sheet.jpg", args.contact_sheet,
                                   args.seed, args.contact_cols, args.contact_tile)
        print(f"[qa] contact sheet ({len(picks)} random frames): "
              f"{qa_dir / 'contact_sheet.jpg'}", flush=True)
    sharpness_plot(sharp, idx, qa_dir / "sharpness.png")

    report = {
        "wp": "WP-O1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": sys.argv,
        "source": {
            "path": os.path.abspath(info.path),
            "bytes": os.path.getsize(info.path),
            "layout": info.layout,
            "container": info.container,
            "width": info.width, "height": info.height,
            "fps": info.fps, "r_frame_rate": info.fps_frac,
            "nb_frames_container": info.nb_frames,
            "n_frames_decoded": int(len(sharp)),
            "duration_s": info.duration,
            "video_streams": info.streams,
        },
        "sampling": {
            "target_pano_fps": args.target_fps,
            "window_frames": win,
            "effective_pano_fps": round(info.fps / win, 4),
            "n_windows": int(np.ceil(len(sharp) / win)),
            "n_selected": len(paths),
            "sharpness_stream": sstream,
            "central_frac": args.central_frac,
            "sharpness_max_side": args.sharpness_max_side,
            "absolute_threshold": None,       # deliberately never used — DEVELOPER_NOTES §5
        },
        "sharpness": {
            "all": {k: round(float(v), 2) for k, v in
                    [("min", sharp.min()), ("p10", np.percentile(sharp, 10)),
                     ("median", np.median(sharp)), ("p90", np.percentile(sharp, 90)),
                     ("max", sharp.max())]},
            "selected": {k: round(float(v), 2) for k, v in
                         [("min", ssel.min()), ("p10", np.percentile(ssel, 10)),
                          ("median", np.median(ssel)), ("p90", np.percentile(ssel, 90)),
                          ("max", ssel.max())]},
            "selected_median_gain_pct": round(
                float(100 * (np.median(ssel) / np.median(sharp) - 1)), 2),
            "per_source_frame": [round(float(v), 2) for v in sharp],
        },
        "stitch": {
            "filter_graph": build_filter_graph(info, [0], out_w, out_h, args.ih_fov,
                                               args.iv_fov, args.yaw),
            "ih_fov": args.ih_fov, "iv_fov": args.iv_fov, "yaw": args.yaw,
            "out_width": out_w, "out_height": out_h, "jpeg_quality": args.jpeg_quality,
        },
        "masks": {
            "convention": "uint8 PNG, 255=VALID, 0=EXCLUDED, <image_stem>.png in masks/",
            "nadir_deg": args.nadir_deg, "zenith_deg": args.zenith_deg,
            "seam_deg": seam_deg, "seam_yaws_deg": [-90.0, 90.0] if seam_deg else [],
            "operator_mode": op_mode,
            "operator_checkpoint": args.sam_checkpoint if op_mode == "sam" else None,
            "operator_pos_points": args.operator_pos, "operator_neg_points": args.operator_neg,
            "operator_box": list(args.operator_box), "operator_box_union": args.operator_box_union,
            "operator_dilate_px": args.operator_dilate_px,
            "operator_max_yaw_deg": args.operator_max_yaw_deg,
            "operator_max_pitch_deg": args.operator_max_pitch_deg,
            "operator_area_guard_frames": n_area_guard,
            "operator_lost_guard_frames": n_lost_guard,
            "geometric_excluded_frac": round(float((geo == EXCLUDED).mean()), 5),
            "mean_excluded_frac": round(float(np.mean([f["excluded_frac"] for f in per_frame])), 5),
            "mean_operator_frac": round(float(np.mean([f["operator_frac"] for f in per_frame])), 5),
        },
        "qa": {"contact_sheet": "qa/contact_sheet.jpg" if args.contact_sheet else None,
               "contact_sheet_frames": picks, "contact_sheet_seed": args.seed,
               "sharpness_plot": "qa/sharpness.png"},
        "timing_s": {"scan": round(t_scan, 2), "stitch": round(t_stitch, 2),
                     "mask": round(t_mask, 2), "total": round(time.time() - t_all, 2)},
        "frames": per_frame,
    }
    rp = out_dir / "ingest_report.json"
    rp.write_text(json.dumps(report, indent=1), encoding="utf8")
    print(f"[done] {len(paths)} frames -> {images_dir}\n       report: {rp} "
          f"({report['timing_s']['total']:.1f} s total)", flush=True)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _box(s: str):
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("expected x0,x1,y0,y1 normalised")
    return tuple(v)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m core.ingest_360",
        description="WP-O1 360 ingest: Insta360 dual-fisheye/ERP video -> images/ + masks/ "
                    "+ ingest_report.json. DEFAULT-OFF: does nothing without "
                    "--capture-type insta360.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--capture-type", dest="capture_type", default="off",
                   choices=["off", "insta360"],
                   help="OFF by default. 'insta360' opts into the 360 ingest lane.")
    p.add_argument("--input", help="one .insv / .mov / .mp4 360 file")
    p.add_argument("--out_dir", help="output dataset dir (images/, masks/, ingest_report.json)")
    p.add_argument("--overwrite", action="store_true", help="delete out_dir first")
    p.add_argument("--force_layout", choices=["dual_fisheye", "erp"],
                   help="override the ffprobe layout auto-detection")

    g = p.add_argument_group("frame sampling (window-relative max sharpness)")
    g.add_argument("--target_fps", type=float, default=2.0,
                   help="target selected pano-frames per second (sets the window size)")
    g.add_argument("--central_frac", type=float, default=0.5,
                   help="central crop fraction used for the sharpness metric")
    g.add_argument("--sharpness_max_side", type=int, default=960,
                   help="downscale cap for the sharpness crop (keeps 8K scans cheap)")
    g.add_argument("--sharpness_stream", type=int, default=0,
                   help="which video stream to score; 0 = scene-facing lens (verified)")
    g.add_argument("--max_frames", type=int, default=0, help="cap selected frames (0 = no cap)")

    g = p.add_argument_group("stitch (ffmpeg v360 dfisheye->equirect)")
    g.add_argument("--ih_fov", type=float, default=200.0, help="input horizontal lens FOV (deg)")
    g.add_argument("--iv_fov", type=float, default=200.0, help="input vertical lens FOV (deg)")
    g.add_argument("--yaw", type=float, default=0.0, help="ERP yaw rotation (deg)")
    g.add_argument("--out_width", type=int, default=0, help="0 = 2x input height")
    g.add_argument("--out_height", type=int, default=0, help="0 = input height")
    g.add_argument("--jpeg_quality", type=int, default=95, help="output JPEG quality")

    g = p.add_argument_group("masks (255 = VALID, 0 = EXCLUDED)")
    g.add_argument("--nadir_deg", type=float, default=12.0,
                   help="bottom band excluded (stick / hand / feet)")
    g.add_argument("--zenith_deg", type=float, default=0.0,
                   help="top band excluded (overhead-stick captures)")
    g.add_argument("--seam_deg", type=float, default=None,
                   help="half-width of the +-90deg stitch-seam bands "
                        "(default 4 for dual fisheye, 0 for pre-stitched ERP)")
    g.add_argument("--operator_mask", choices=["auto", "sam", "box", "none"], default="auto",
                   help="auto = sam for dual fisheye, none for pre-stitched ERP")
    g.add_argument("--sam_checkpoint", default="sam2.1_hiera_large",
                   help="SAM 2.1 checkpoint name (resolved inside the repo's sam2 clone)")
    g.add_argument("--device", default="cuda")
    g.add_argument("--operator_sam_width", type=int, default=1024,
                   help="ERP width fed to SAM 2.1 (it resizes to 1024 internally anyway)")
    g.add_argument("--operator_pos",
                   default="0.478,0.50;0.478,0.74;0.44,0.84;0.51,0.84;0.385,0.86;0.605,0.84;0.475,0.93",
                   help="normalised positive prompt points 'x,y;x,y' on the FIRST frame - "
                        "head, torso, both shirt corners, BOTH FOREARMS, shorts. Tuned for the "
                        "standard chest-height-stick selfie pose; retune per rig if it misses.")
    g.add_argument("--operator_neg",
                   default="0.25,0.5;0.75,0.5;0.478,0.10;0.10,0.80;0.90,0.80;0.675,0.60",
                   help="normalised negative prompt points (seams, ceiling, far floor)")
    g.add_argument("--operator_box", type=_box, default=(0.37, 0.60, 0.72, 1.00),
                   help="normalised x0,x1,y0,y1 guaranteed-core operator body box - the "
                        "fallback used when SAM 2.1 is unavailable or its guards fire")
    g.add_argument("--operator_box_union", type=int, default=0,
                   help="1 = union the core box with the SAM mask on EVERY frame (belt-and-"
                        "braces; costs ~3%% of the frame in real scene content)")
    g.add_argument("--operator_dilate_px", type=int, default=12,
                   help="dilation of the operator mask, pixels")
    g.add_argument("--operator_max_yaw_deg", type=float, default=90.0,
                   help="operator clamp: rear hemisphere half-width")
    g.add_argument("--operator_max_pitch_deg", type=float, default=25.0,
                   help="operator clamp: highest elevation the operator may occupy")
    g.add_argument("--operator_max_area_frac", type=float, default=0.25,
                   help="above this frame fraction the SAM mask is discarded (runaway guard)")
    g.add_argument("--operator_min_area_frac", type=float, default=0.01,
                   help="below this the tracker is treated as lost (box carries the frame)")
    g.add_argument("--keep_sam_frames", action="store_true",
                   help="keep the downscaled SAM input frames for debugging")

    g = p.add_argument_group("QA / misc")
    g.add_argument("--contact_sheet", type=int, default=20,
                   help="N random frames in the mask-overlay QA sheet (0 = skip)")
    g.add_argument("--contact_cols", type=int, default=4)
    g.add_argument("--contact_tile", type=int, default=640)
    g.add_argument("--seed", type=int, default=0, help="contact-sheet sampling seed")
    g.add_argument("--progress", type=int, default=250, help="progress print interval")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.capture_type != "insta360":
        print("360 ingest is OFF (default). Pass --capture-type insta360 to enable it.")
        return 2
    if not args.input or not args.out_dir:
        print("--input and --out_dir are required with --capture-type insta360")
        return 2
    try:
        ingest(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
