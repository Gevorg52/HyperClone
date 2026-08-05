"""WP-O2 — 360 SfM lane: one WP-O1 pano dataset -> a COLMAP sparse model, two arms.

Takes the `images/` + `masks/` folder that `core.ingest_360` writes and reconstructs it with
COLMAP 4.1.0's panorama machinery. Two arms, same CLI, same report schema:

  ARM A  --arm crop_rig    PERSPECTIVE_OVERLAPPING: every pano is rendered into 12 virtual
                           pinhole cameras (4 yaw steps x pitch rows -35/0/+35 deg, 90x90 deg
                           FOV, half-step yaw stagger on the +35 row) that share the pano's
                           optical centre. The 12 crops of one pano are ONE RIG FRAME
                           (pycolmap.RigConfig -> apply_rig_config) with pure-rotation, exactly
                           known cam_from_rig, so BA holds the rig extrinsics AND the virtual
                           intrinsics FIXED and only pano poses + points move. Each pano pixel
                           is given to exactly ONE crop (nearest virtual camera centre) so a
                           feature cannot be extracted twice. Delivers sparse/0 in crop space
                           plus sparse_equirectangular/0 (one EQUIRECTANGULAR camera/image per
                           pano) via convert_to_equirectangular().
  ARM B  --arm spherical   EQUIRECTANGULAR single camera straight on the ERP frames (params are
                           (w,h) only: a fixed mapping with nothing to refine in BA). Faster,
                           "generally less accurate" per COLMAP's own source comment.

BOTH arms feed the WP-O1 masks (uint8 PNG, 255 = VALID, 0 = EXCLUDED: operator / nadir /
stitch seams) into feature extraction. COLMAP keeps a keypoint iff its mask pixel is non-zero
(feature_extraction.cc MaskFeatures), which is our convention already, and it resolves
`masks/pano_00000.png` for `images/pano_00000.jpg` on its own (image_reader.cc extension
fallback). In ARM A the WP-O1 pano mask is warped through the SAME remap as the image and
AND-ed into the exclusive crop mask, so a pano pixel masked by WP-O1 yields features in NO
crop. `--use_masks 0` is the honest no-priors A/B path.

DEFAULT-OFF: the module does nothing unless `--capture-type 360` is passed, and no production
module imports it (grep: nothing outside tests/ and _infusion/ references core.sfm_360), so
the production reconstruction path is untouched.

    <out_dir>/database.db                     COLMAP database
    <out_dir>/images/pano_cameraN/*.jpg       ARM A only: the 12 crops per pano (q95)
    <out_dir>/masks/pano_cameraN/*.jpg.png    ARM A only: exclusive crop mask AND WP-O1 mask
    <out_dir>/sparse/0/                       the reconstruction the mapper produced
    <out_dir>/sparse_equirectangular/0/       ARM A only: same model as one ERP cam/img per pano
    <out_dir>/sfm_report.json                 argv + seed + options + timings + metrics
    <out_dir>/qa/trajectory.png               top-down camera track + step histogram

PROVENANCE: the geometry (create_virtual_camera / get_virtual_camera_rays /
spherical_img_from_cam / get_virtual_rotations / create_pano_rig_config /
convert_to_equirectangular) is adapted from COLMAP's own panorama pipeline (BSD-3, snapshot in
tools/reference_sources/colmap_panorama/, ledger entry in THIRD_PARTY.md). Deliberate
deviations from upstream:
  1. WP-O1 mask compositing (upstream has no notion of an input mask).
  2. The 12 remap tables and the 12 exclusive masks are pano-independent, so they are built
     ONCE instead of per pano (upstream rebuilds them for every image).
  3. cv2 for IO instead of PIL (repo convention; JPEG quality is explicit at q95 instead of
     PIL's default 75). EXIF/GPS carry-over is dropped - our ERP frames carry none.
  4. Everything is wrapped in a stage heartbeat + hard `--timeout_s` because
     colmap/colmap#3599 reports panorama_sfm silently hanging on some setups. On timeout the
     process dumps every thread's Python stack and exits 3.
  5. Covisibility filtering (needs a depth-derived .npz we do not have) is not ported.

ENVIRONMENT: our pycolmap 4.1.0 wheel is built WITHOUT CUDA ("COLMAP 4.1.0 Commit fa8e3b3 on
2026-06-26 without CUDA"), so SIFT extraction and matching run on the CPU. `--use_gpu 1` is
accepted and recorded but has no effect in this build; the default is CPU.

EXIT CODES: 0 = ok · 1 = the run failed (bad input / COLMAP error) · 2 = usage (arm flag off,
missing args, out_dir already populated without --overwrite) · 3 = `--timeout_s` fired.

CLI:
    python -m core.sfm_360 --capture-type 360 --arm crop_rig \
        --dataset _dataset/360/flowershop_077 --out_dir _dataset/360/sfm/wpo2_crop_rig
    python -m core.sfm_360 --capture-type 360 --arm spherical \
        --dataset _dataset/360/flowershop_077 --out_dir _dataset/360/sfm/wpo2_spherical
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ARMS = ("crop_rig", "spherical")
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# 1. watchdog — colmap#3599 says panorama_sfm can hang silently; never run blind
# ---------------------------------------------------------------------------

class Watchdog:
    """Prints a heartbeat per stage and hard-exits the process on `timeout_s`.

    A pycolmap call is C++ holding the GIL-free thread; it cannot be interrupted from
    Python, so the only honest options are "keep printing so a hang is visible" and
    "die loudly with a stack dump". We do both.
    """

    def __init__(self, timeout_s: float, interval_s: float = 60.0):
        self.timeout_s = float(timeout_s)
        self.interval_s = float(interval_s)
        self.t0 = time.time()
        self.stage_name = "init"
        self.stage_t0 = self.t0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def stage(self, name: str) -> None:
        now = time.time()
        print(f"[stage] {self.stage_name} done in {now - self.stage_t0:.1f} s -> {name}",
              flush=True)
        self.stage_name = name
        self.stage_t0 = now

    def start(self) -> "Watchdog":
        def loop():
            while not self._stop.wait(self.interval_s):
                el = time.time() - self.t0
                print(f"[hb] stage={self.stage_name} stage_elapsed="
                      f"{time.time() - self.stage_t0:.0f}s total={el:.0f}s", flush=True)
                if self.timeout_s > 0 and el > self.timeout_s:
                    print(f"\nTIMEOUT after {el:.0f} s in stage '{self.stage_name}' "
                          f"(--timeout_s {self.timeout_s:.0f}). This is the colmap#3599 "
                          f"failure mode; dumping all Python stacks and exiting 3.",
                          flush=True)
                    faulthandler.dump_traceback()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(3)
        self._thread = threading.Thread(target=loop, daemon=True, name="wpo2-watchdog")
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        return time.time() - self.t0


# ---------------------------------------------------------------------------
# 2. pano -> virtual perspective rig  (geometry adapted from COLMAP, BSD-3)
# ---------------------------------------------------------------------------

@dataclass
class RenderOptions:
    num_steps_yaw: int
    pitches_deg: tuple
    hfov_deg: float
    vfov_deg: float


RENDER_PRESETS = {
    # COLMAP's PERSPECTIVE_OVERLAPPING default: 4 yaws x 3 pitch rows = 12 crops.
    "perspective_overlapping": RenderOptions(4, (-35.0, 0.0, 35.0), 90.0, 90.0),
    # COLMAP's PERSPECTIVE_NON_OVERLAPPING: cubemap sides only = 4 crops.
    "perspective_non_overlapping": RenderOptions(4, (0.0,), 90.0, 90.0),
}


def create_virtual_camera(pano_width: int, pano_height: int, hfov_deg: float, vfov_deg: float):
    """A SIMPLE_PINHOLE camera whose angular resolution matches the pano's equator."""
    import pycolmap

    image_width = int(pano_width * hfov_deg / 360)
    image_height = int(pano_height * vfov_deg / 180)
    focal = image_width / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    camera = pycolmap.Camera.create_from_model_id(
        camera_id=0, model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
        focal_length=focal, width=image_width, height=image_height)
    camera.has_prior_focal_length = True      # not set by create_from_model_id
    return camera


def get_virtual_camera_rays(camera) -> np.ndarray:
    """Unit rays through every pixel centre of the virtual camera, C-order over (x, y)."""
    x, y = np.indices((camera.width, camera.height)).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()]) + 0.5   # COLMAP pixel centres
    xy_norm = camera.cam_from_img(image_points=xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def spherical_img_from_cam(image_size: tuple, rays_in_cam: np.ndarray) -> np.ndarray:
    """Project camera rays into ERP pixel coordinates (COLMAP pixel convention)."""
    if image_size[0] != image_size[1] * 2:
        raise ValueError(f"Only 360 deg 2:1 panoramas are supported, got {image_size}.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"rays_in_cam.shape={rays_in_cam.shape} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size


def get_virtual_rotations(num_steps_yaw: int, pitches_deg) -> list:
    """cam_from_pano rotations of the virtual cameras (panos assumed approximately upright)."""
    out = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            pitch, yaw = np.deg2rad([-pitch_deg, -yaw_deg])
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
            rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
            out.append(rot_x @ rot_y)
    return out


def create_pano_rig_config(cams_from_pano_rotation, ref_idx: int = 0):
    """One rig, N virtual cameras, pure-rotation cam_from_rig (shared optical centre)."""
    import pycolmap

    zero = np.zeros((3, 1), dtype=np.float64)
    rig_cameras = []
    for idx, cam_from_pano_r in enumerate(cams_from_pano_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref = cam_from_pano_r @ cams_from_pano_rotation[ref_idx].T
            cam_from_rig = pycolmap.Rigid3d(pycolmap.Rotation3d(cam_from_ref), zero)
        rig_cameras.append(pycolmap.RigConfigCamera(
            ref_sensor=idx == ref_idx,
            image_prefix=f"pano_camera{idx}/",
            cam_from_rig=cam_from_rig))
    return pycolmap.RigConfig(cameras=rig_cameras)


class PanoProcessor:
    """Renders the pano rig and converts the crop reconstruction back to equirectangular.

    Deviation from upstream: the remap tables and the exclusive masks do not depend on the
    pano content, so they are built once in `prepare()` and reused for every frame; only the
    WP-O1 mask warp is per frame.
    """

    def __init__(self, pano_image_dir: Path, pano_mask_dir: Path | None,
                 out_image_dir: Path, out_mask_dir: Path,
                 render_options: RenderOptions, jpeg_quality: int = 95):
        self.pano_image_dir = Path(pano_image_dir)
        self.pano_mask_dir = Path(pano_mask_dir) if pano_mask_dir else None
        self.out_image_dir = Path(out_image_dir)
        self.out_mask_dir = Path(out_mask_dir)
        self.render_options = render_options
        self.jpeg_quality = int(jpeg_quality)

        self.cams_from_pano_rotation = get_virtual_rotations(
            render_options.num_steps_yaw, render_options.pitches_deg)
        self.rig_config = create_pano_rig_config(self.cams_from_pano_rotation)
        # Each pano pixel belongs to the virtual camera with the closest camera centre.
        self.cam_centers_in_pano = np.einsum(
            "nij,i->nj", self.cams_from_pano_rotation, [0, 0, 1])

        self.camera = None
        self.pano_size: tuple | None = None
        self.maps: list = []              # per crop: (map_x, map_y) float32 (H, W)
        self.excl_masks: list = []        # per crop: uint8 (H, W), 255 where this crop owns
        self.excl_frac: list = []
        self.sphere_coverage = 0.0        # fraction of 4*pi the exclusive masks actually cover

    # -- rendering ---------------------------------------------------------

    def prepare(self, pano_width: int, pano_height: int) -> None:
        self.camera = create_virtual_camera(
            pano_width, pano_height,
            self.render_options.hfov_deg, self.render_options.vfov_deg)
        for rig_camera in self.rig_config.cameras:
            rig_camera.camera = self.camera
        self.pano_size = (pano_width, pano_height)
        rays_in_cam = get_virtual_camera_rays(self.camera)
        w, h = self.camera.width, self.camera.height
        # Solid angle of one pixel of a pinhole camera: dA * cos^3(theta) / f^2, and
        # cos(theta) is the z component of the unit ray. Summed over the exclusive masks this
        # says how much of the sphere the rig actually sees (upstream's 12-crop layout leaves
        # polar caps uncovered - the ERP arm does not).
        pix_sr = rays_in_cam[:, 2] ** 3 / float(self.camera.params[0]) ** 2
        covered_sr = 0.0
        for cam_idx, cam_from_pano_r in enumerate(self.cams_from_pano_rotation):
            rays_in_pano = rays_in_cam @ cam_from_pano_r
            xy = spherical_img_from_cam(self.pano_size, rays_in_pano)
            xy = xy.reshape(w, h, 2).astype(np.float32) - 0.5   # COLMAP -> OpenCV origin
            map_x, map_y = np.moveaxis(xy, [0, 1, 2], [2, 1, 0])
            self.maps.append((np.ascontiguousarray(map_x), np.ascontiguousarray(map_y)))
            closest = np.argmax(rays_in_pano @ self.cam_centers_in_pano.T, -1)
            covered_sr += float(pix_sr[closest == cam_idx].sum())
            excl = ((closest == cam_idx) * 255).astype(np.uint8).reshape(w, h).transpose()
            self.excl_masks.append(np.ascontiguousarray(excl))
            self.excl_frac.append(float((excl > 0).mean()))
        self.sphere_coverage = covered_sr / (4 * np.pi)

    def process(self, pano_name: str) -> dict:
        pano_path = self.pano_image_dir / pano_name
        pano = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
        if pano is None:
            raise RuntimeError(f"cannot read panorama {pano_path}")
        ph, pw = pano.shape[:2]
        if pw != ph * 2:
            raise ValueError(f"{pano_name}: only 360 deg 2:1 panoramas are supported ({pw}x{ph})")
        if (pw, ph) != self.pano_size:
            raise ValueError(f"{pano_name}: {pw}x{ph} but the rig was built for {self.pano_size}")

        pano_mask = None
        if self.pano_mask_dir is not None:
            mp = self.pano_mask_dir / (Path(pano_name).stem + ".png")
            pano_mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if pano_mask is None:
                raise RuntimeError(f"cannot read WP-O1 mask {mp}")
            if pano_mask.shape[:2] != (ph, pw):
                raise ValueError(f"mask {mp} is {pano_mask.shape[:2]}, image is {(ph, pw)}")

        valid_fracs = []
        for cam_idx, (map_x, map_y) in enumerate(self.maps):
            image = cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_WRAP)
            mask = self.excl_masks[cam_idx]
            if pano_mask is not None:
                # Bilinear warp + "== 255" keeps ONLY crop pixels whose four ERP taps are all
                # VALID, i.e. the composite is conservative at the mask boundary.
                warped = cv2.remap(pano_mask, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_WRAP)
                mask = cv2.bitwise_and(mask, (warped >= 255).astype(np.uint8) * 255)
            valid_fracs.append(float((mask > 0).mean()))

            image_name = self.rig_config.cameras[cam_idx].image_prefix + pano_name
            image_path = self.out_image_dir / image_name
            mask_path = self.out_mask_dir / f"{image_name}.png"
            image_path.parent.mkdir(exist_ok=True, parents=True)
            mask_path.parent.mkdir(exist_ok=True, parents=True)
            if not cv2.imwrite(str(image_path), image,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
                raise RuntimeError(f"cannot write {image_path}")
            if not cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"cannot write {mask_path}")
        return {"pano": pano_name, "valid_frac": valid_fracs}

    # -- reconstruction bookkeeping ---------------------------------------

    def split_image_name(self, image_name: str) -> tuple:
        for cam_idx, rig_camera in enumerate(self.rig_config.cameras):
            prefix = rig_camera.image_prefix
            if image_name.startswith(prefix):
                return cam_idx, image_name[len(prefix):]
        raise ValueError(f"Unknown virtual camera for image {image_name!r}.")

    def convert_to_equirectangular(self, reconstruction):
        """Fold the virtual-camera reconstruction back onto one ERP camera/image per pano.

        Verbatim in spirit from COLMAP's panorama.py: frame poses, 3D points and ALL
        keypoints are carried over by re-projecting through the same spherical mapping, so
        the result is a valid, bundle-adjustable EQUIRECTANGULAR reconstruction.
        """
        import collections

        import pycolmap

        if self.camera is None or self.pano_size is None:
            raise RuntimeError("No panorama was rendered yet.")
        pano_width, pano_height = self.pano_size

        equirect = pycolmap.Reconstruction()
        equirect_camera = pycolmap.Camera.create_from_model_id(
            camera_id=1, model=pycolmap.CameraModelId.EQUIRECTANGULAR,
            focal_length=0.0, width=pano_width, height=pano_height)
        equirect.add_camera_with_trivial_rig(equirect_camera)

        # Rig reference sensor is virtual camera 0, so rig_from_world == cam0_from_world and
        # pano_from_world = pano_from_cam0 @ cam0_from_world (zero translation: the virtual
        # cameras share the pano centre).
        pano_from_ref = pycolmap.Rigid3d(
            pycolmap.Rotation3d(self.cams_from_pano_rotation[0]),
            np.zeros((3, 1), dtype=np.float64)).inverse()

        images_by_frame = collections.defaultdict(list)
        for image in reconstruction.images.values():
            if image.has_pose:
                images_by_frame[image.frame_id].append(image)

        old_to_new_point2D: dict = {}
        pano_to_image_id: dict = {}
        frame_images = sorted(images_by_frame.values(),
                              key=lambda images: self.split_image_name(images[0].name)[1])
        for image_id, images in enumerate(frame_images, start=1):
            pano_name = self.split_image_name(images[0].name)[1]
            pano_to_image_id[pano_name] = image_id
            frame = images[0].frame
            pano_from_world = pano_from_ref * frame.rig_from_world

            keypoints: list = []
            for image in images:
                cam_idx = self.split_image_name(image.name)[0]
                num_points2D = len(image.points2D)
                if num_points2D == 0:
                    old_to_new_point2D[image.image_id] = {}
                    continue
                xy = np.array([point2D.xy for point2D in image.points2D])
                rays_in_cam = np.asarray(self.camera.cam_ray_from_img(image_points=xy))
                rays_in_cam /= np.linalg.norm(rays_in_cam, axis=-1, keepdims=True)
                rays_in_pano = rays_in_cam @ self.cams_from_pano_rotation[cam_idx]
                xy_in_pano = spherical_img_from_cam(self.pano_size, rays_in_pano)
                base_idx = len(keypoints)
                keypoints.extend(xy_in_pano)
                old_to_new_point2D[image.image_id] = {
                    i: (base_idx + i, pano_name) for i in range(num_points2D)}

            equirect.add_image_with_trivial_frame(
                pycolmap.Image(name=pano_name, keypoints=keypoints,
                               camera_id=equirect_camera.camera_id, image_id=image_id),
                pano_from_world)

        for point3D_id, point3D in reconstruction.points3D.items():
            track = pycolmap.Track()
            for element in point3D.track.elements:
                new_idx, pano_name = old_to_new_point2D[element.image_id][element.point2D_idx]
                track.add_element(pano_to_image_id[pano_name], new_idx)
            equirect.add_point3D_with_id(
                point3D_id,
                pycolmap.Point3D(xyz=point3D.xyz, color=point3D.color, track=track))
        return equirect


def render_pano_rig(processor: PanoProcessor, pano_names, workers: int, progress: int) -> list:
    stats: list = []
    done = 0
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(processor.process, name) for name in pano_names]
        for future in as_completed(futures):
            s = future.result()
            with lock:
                stats.append(s)
                done += 1
                if progress and done % progress == 0:
                    print(f"[render] {done}/{len(pano_names)} panos", flush=True)
    return stats


# ---------------------------------------------------------------------------
# 3. metrics
# ---------------------------------------------------------------------------

def _f(x) -> float:
    return round(float(x), 6)


def reconstruction_metrics(rec, n_input_panos: int, pano_of_image=None) -> dict:
    """Registration / error / track statistics for one reconstruction."""
    reg_panos = set()
    for image_id in rec.reg_image_ids():
        name = rec.images[image_id].name
        reg_panos.add(pano_of_image(name) if pano_of_image else name)
    return {
        "num_input_panos": int(n_input_panos),
        "num_registered_panos": len(reg_panos),
        "registered_pct": _f(100.0 * len(reg_panos) / max(1, n_input_panos)),
        "num_images": int(rec.num_images()),
        "num_reg_images": int(rec.num_reg_images()),
        "num_frames": int(rec.num_frames()),
        "num_reg_frames": int(rec.num_reg_frames()),
        "num_points3D": int(rec.num_points3D()),
        "num_observations": int(rec.compute_num_observations()),
        "mean_reproj_error_px": _f(rec.compute_mean_reprojection_error()),
        "mean_track_length": _f(rec.compute_mean_track_length()),
        "mean_obs_per_reg_image": _f(rec.compute_mean_observations_per_reg_image()),
    }


def pano_centers(rec, pano_of_image=None) -> tuple:
    """(names, Nx3 centres) of the registered panos, ordered by pano name."""
    per_pano: dict = {}
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        name = pano_of_image(image.name) if pano_of_image else image.name
        per_pano.setdefault(name, []).append(np.asarray(image.projection_center()))
    names = sorted(per_pano)
    centers = np.array([np.mean(per_pano[n], axis=0) for n in names])
    return names, centers


def estimate_up(rec) -> np.ndarray:
    """World-space "up" from the panos themselves — needs an EQUIRECTANGULAR reconstruction.

    In our ERP convention the zenith is the top row, whose camera-frame ray is (0, -1, 0)
    (see spherical_img_from_cam). The operator carries the rig roughly upright, so averaging
    R_i^T @ (0,-1,0) over the registered panos recovers world up well enough to separate the
    floor plan from the height variation - which is what "corridor-shaped" actually means.
    """
    ups = []
    for image_id in rec.reg_image_ids():
        r = rec.images[image_id].cam_from_world().rotation.matrix()
        ups.append(r.T @ np.array([0.0, -1.0, 0.0]))
    if not ups:
        return np.array([0.0, 0.0, 1.0])
    up = np.mean(ups, axis=0)
    n = np.linalg.norm(up)
    return up / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def pca_frame(centers: np.ndarray) -> np.ndarray:
    """Right-handed basis whose columns are the principal axes of the camera track.

    COLMAP's world frame is arbitrary (it is fixed by the first image pair), so an
    axis-aligned bounding box says nothing about the SHAPE of the track. The principal
    axes do: a corridor walk has one dominant component.
    """
    centered = centers - centers.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def upright_stats(rec) -> dict:
    """How well the panos satisfy COLMAP's "approximately upright" precondition.

    COLMAP's panorama rig is built assuming the panos are roughly level (PLAN_360 §3, quoted
    from upstream's own source). We do not apply any horizon levelling in WP-O1, so this
    measures the violation: the angle of each pano's own zenith against the mean zenith.
    """
    ups = []
    for image_id in rec.reg_image_ids():
        r = rec.images[image_id].cam_from_world().rotation.matrix()
        ups.append(r.T @ np.array([0.0, -1.0, 0.0]))
    if len(ups) < 2:
        return {}
    ups = np.asarray(ups)
    mean_up = ups.mean(0)
    mean_up /= max(np.linalg.norm(mean_up), 1e-12)
    ang = np.degrees(np.arccos(np.clip(ups @ mean_up, -1.0, 1.0)))
    return {"tilt_deg_median": _f(np.median(ang)), "tilt_deg_p90": _f(np.percentile(ang, 90)),
            "tilt_deg_max": _f(ang.max()), "n": int(len(ang))}


def trajectory_stats(centers: np.ndarray, up: np.ndarray | None = None) -> dict:
    """Scale-free shape/smoothness descriptors of a camera track (SfM scale is arbitrary)."""
    if len(centers) < 2:
        return {"n": int(len(centers))}
    extent = centers.max(0) - centers.min(0)
    order = np.argsort(-extent)
    proj = (centers - centers.mean(0)) @ pca_frame(centers)
    pca_extent = proj.max(0) - proj.min(0)
    plan: dict = {}
    if up is not None:
        up = np.asarray(up, float)
        up = up / max(np.linalg.norm(up), 1e-12)
        rel = centers - centers.mean(0)
        height = rel @ up
        flat = rel - np.outer(height, up)
        fproj = flat @ pca_frame(flat)[:, :2]
        fext = fproj.max(0) - fproj.min(0)
        plan = {
            "up_world": [_f(v) for v in up],
            "height_extent": _f(height.max() - height.min()),
            "height_std": _f(height.std()),
            "floorplan_extent_sorted": [_f(v) for v in np.sort(fext)[::-1]],
            "floorplan_aspect": _f(min(fext) / max(max(fext), 1e-12)),
            "height_over_floorplan_extent": _f((height.max() - height.min())
                                               / max(max(fext), 1e-12)),
        }
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    med = float(np.median(steps)) or 1e-12
    path = float(steps.sum())
    disp = float(np.linalg.norm(centers[-1] - centers[0]))
    return {
        "n": int(len(centers)),
        "pca_extent_sorted": [_f(e) for e in pca_extent],
        "pca_extent_ratio": [_f(e / pca_extent[0]) for e in pca_extent],
        "bbox_extent_sorted": [_f(e) for e in extent[order]],
        "bbox_axes_sorted": [int(a) for a in order],
        "bbox_extent_ratio": [_f(e / max(extent)) for e in extent[order]],
        "step_median": _f(med),
        "step_mean": _f(steps.mean()),
        "step_p90": _f(np.percentile(steps, 90)),
        "step_max": _f(steps.max()),
        "step_p90_over_median": _f(np.percentile(steps, 90) / med),
        "step_max_over_median": _f(steps.max() / med),
        "num_steps_gt_5x_median": int((steps > 5 * med).sum()),
        "path_length": _f(path),
        "end_to_end_displacement": _f(disp),
        "path_over_displacement": _f(path / max(disp, 1e-12)),
        "extent_over_median_step": _f(pca_extent[0] / med),
        **plan,
    }


def plot_trajectory(centers: np.ndarray, out_png: Path, title: str,
                    up: np.ndarray | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # COLMAP's world axes are arbitrary. With an estimated up vector this is a true
    # top-down floor plan; without one it falls back to the track's principal plane.
    rel = centers - centers.mean(0)
    if up is not None:
        up = np.asarray(up, float)
        up = up / max(np.linalg.norm(up), 1e-12)
        flat = rel - np.outer(rel @ up, up)
        proj = flat @ pca_frame(flat)[:, :2]
        plane = "floor plane (up removed)"
    else:
        proj = rel @ pca_frame(rel)[:, :2]
        plane = "principal plane"
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) if len(centers) > 1 else np.zeros(1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [2, 1]})
    ax[0].plot(proj[:, 0], proj[:, 1], "-", lw=1.0, color="0.6", zorder=1)
    sc = ax[0].scatter(proj[:, 0], proj[:, 1], c=np.arange(len(centers)),
                       cmap="viridis", s=16, zorder=2)
    ax[0].scatter(proj[0, 0], proj[0, 1], marker="s", s=90,
                  facecolor="none", edgecolor="tab:green", lw=2, label="first", zorder=3)
    ax[0].scatter(proj[-1, 0], proj[-1, 1], marker="X", s=90,
                  color="tab:red", label="last", zorder=3)
    ax[0].set_aspect("equal", "datalim")
    ax[0].set_xlabel(f"{plane}: axis 1 (SfM units)")
    ax[0].set_ylabel(f"{plane}: axis 2 (SfM units)")
    ax[0].legend(loc="best", fontsize=8)
    ax[0].set_title(f"{title} — top-down camera track (n={len(centers)})", fontsize=10)
    fig.colorbar(sc, ax=ax[0], label="pano index", fraction=0.035)
    ax[1].hist(steps, bins=30, color="tab:blue")
    ax[1].axvline(np.median(steps), color="k", ls="--", lw=1,
                  label=f"median {np.median(steps):.3f}")
    ax[1].set_xlabel("inter-pano step (SfM units)")
    ax[1].set_ylabel("count")
    ax[1].legend(fontsize=8)
    ax[1].set_title("step-size distribution", fontsize=10)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. the two arms
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    dataset: Path
    images: Path
    masks: Path | None
    out_dir: Path
    database: Path
    sparse: Path
    report: Path
    qa: Path = field(init=False)

    def __post_init__(self):
        self.qa = self.out_dir / "qa"


def resolve_paths(args) -> Paths:
    dataset = Path(args.dataset) if args.dataset else Path(args.images).parent
    images = Path(args.images) if args.images else dataset / "images"
    masks = None
    if args.use_masks:
        masks = Path(args.masks) if args.masks else dataset / "masks"
        if not masks.is_dir():
            raise FileNotFoundError(f"mask dir {masks} does not exist (pass --use_masks 0 to "
                                    f"run the no-priors arm)")
    if not images.is_dir():
        raise FileNotFoundError(f"image dir {images} does not exist")
    out_dir = Path(args.out_dir)
    return Paths(dataset=dataset, images=images, masks=masks, out_dir=out_dir,
                 database=out_dir / "database.db", sparse=out_dir / "sparse",
                 report=out_dir / "sfm_report.json")


def list_panos(image_dir: Path, max_panos: int = 0) -> list:
    names = sorted(p.name for p in image_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not names:
        raise FileNotFoundError(f"no images in {image_dir}")
    return names[:max_panos] if max_panos else names


def matching_options(args, rig: bool):
    import pycolmap

    opts = pycolmap.FeatureMatchingOptions()
    opts.use_gpu = bool(args.use_gpu)
    opts.gpu_index = str(args.gpu_index)
    opts.num_threads = int(args.num_threads)
    if rig:
        # Perfect sensor_from_rig by construction (up to stitching artifacts) -> use rig
        # constraints in geometric verification; the exclusive masks mean crops of one pano
        # share no pixel, so intra-frame pairs are pure waste.
        opts.rig_verification = True
        opts.skip_image_pairs_in_same_frame = True
    return opts


def run_matcher(args, database_path: Path, opts) -> dict:
    import pycolmap

    t0 = time.time()
    info = {"matcher": args.matcher}
    if args.matcher == "sequential":
        pairing = pycolmap.SequentialPairingOptions(
            overlap=int(args.overlap),
            quadratic_overlap=bool(args.quadratic_overlap),
            loop_detection=bool(args.loop_detection),
            num_threads=int(args.num_threads))
        info.update(overlap=int(args.overlap),
                    quadratic_overlap=bool(args.quadratic_overlap),
                    loop_detection=bool(args.loop_detection),
                    expand_rig_images=bool(pairing.expand_rig_images))
        pycolmap.match_sequential(database_path, pairing_options=pairing,
                                  matching_options=opts)
    elif args.matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path, matching_options=opts)
    elif args.matcher == "vocabtree":
        pycolmap.match_vocabtree(database_path, matching_options=opts)
    else:
        raise ValueError(f"unknown matcher {args.matcher}")
    info["seconds"] = round(time.time() - t0, 2)
    return info


def mapper_options(args, fix_rig: bool):
    import pycolmap

    kw = dict(num_threads=int(args.num_threads), random_seed=int(args.seed))
    if fix_rig:
        # Rig extrinsics AND the virtual intrinsics are known by construction.
        kw.update(ba_refine_sensor_from_rig=False, ba_refine_focal_length=False,
                  ba_refine_principal_point=False, ba_refine_extra_params=False)
    return pycolmap.IncrementalPipelineOptions(**kw)


def run_crop_rig(args, paths: Paths, wd: Watchdog) -> dict:
    import pycolmap

    render_options = RENDER_PRESETS[args.render_type]
    pano_names = list_panos(paths.images, args.max_panos)
    crop_images = paths.out_dir / "images"
    crop_masks = paths.out_dir / "masks"
    crop_images.mkdir(parents=True, exist_ok=True)
    crop_masks.mkdir(parents=True, exist_ok=True)

    probe = cv2.imread(str(paths.images / pano_names[0]), cv2.IMREAD_COLOR)
    if probe is None:
        raise RuntimeError(f"cannot read {paths.images / pano_names[0]}")
    pano_h, pano_w = probe.shape[:2]
    processor = PanoProcessor(paths.images, paths.masks, crop_images, crop_masks,
                              render_options, args.jpeg_quality)
    processor.prepare(pano_w, pano_h)
    n_cams = len(processor.cams_from_pano_rotation)
    print(f"[crop_rig] {len(pano_names)} panos {pano_w}x{pano_h} -> {n_cams} virtual "
          f"{processor.camera.width}x{processor.camera.height} cameras "
          f"(f={processor.camera.params[0]:.1f}); {len(pano_names) * n_cams} crops",
          flush=True)

    wd.stage("render")
    t0 = time.time()
    stats = render_pano_rig(processor, pano_names, args.render_workers, args.progress)
    t_render = time.time() - t0
    valid = np.array([s["valid_frac"] for s in stats])          # (n_panos, n_cams)

    wd.stage("extract_features")
    t0 = time.time()
    reader = pycolmap.ImageReaderOptions(
        camera_model=processor.camera.model_name,
        camera_params=processor.camera.params_to_string())
    if paths.masks is not None:
        reader.mask_path = crop_masks
    pycolmap.extract_features(
        paths.database, crop_images, reader_options=reader,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
        extraction_options=pycolmap.FeatureExtractionOptions(
            use_gpu=bool(args.use_gpu), gpu_index=str(args.gpu_index),
            num_threads=int(args.num_threads)))
    t_extract = time.time() - t0

    wd.stage("apply_rig_config")
    with pycolmap.Database.open(paths.database) as db:
        pycolmap.apply_rig_config([processor.rig_config], db)

    wd.stage(f"match_{args.matcher}")
    match_info = run_matcher(args, paths.database, matching_options(args, rig=True))

    wd.stage("incremental_mapping")
    t0 = time.time()
    recs = pycolmap.incremental_mapping(paths.database, crop_images, paths.sparse,
                                        mapper_options(args, fix_rig=True))
    t_map = time.time() - t0
    if not recs:
        raise RuntimeError("incremental_mapping produced no reconstruction")
    for idx, rec in recs.items():
        print(f"[map] #{idx} {rec.summary()}", flush=True)

    wd.stage("convert_to_equirectangular")
    best_idx = max(recs, key=lambda i: recs[i].num_reg_images())
    rec = recs[best_idx]
    equirect_dir = paths.out_dir / "sparse_equirectangular"
    equirect_metrics = None
    equirect_traj = None
    for idx, r in recs.items():
        eq = processor.convert_to_equirectangular(r)
        eq.update_point_3d_errors()      # errors are not carried over by the conversion
        model_path = equirect_dir / str(idx)
        model_path.mkdir(parents=True, exist_ok=True)
        eq.write(model_path)
        print(f"[equirect] #{idx} {eq.summary()}", flush=True)
        if idx == best_idx:
            equirect_metrics = reconstruction_metrics(eq, len(pano_names))
            names, centers = pano_centers(eq)
            up = estimate_up(eq)
            equirect_traj = trajectory_stats(centers, up)
            np.save(paths.qa / "pano_centers.npy", centers)
            np.save(paths.qa / "pano_up.npy", up)
            plot_trajectory(centers, paths.qa / "trajectory.png", "ARM A crop-rig", up)
            (paths.qa / "pano_order.json").write_text(json.dumps(names, indent=1), "utf8")

    metrics = reconstruction_metrics(rec, len(pano_names),
                                     lambda n: processor.split_image_name(n)[1])
    return {
        "arm": "crop_rig",
        "render_type": args.render_type,
        "num_virtual_cameras": n_cams,
        "pano_size": [pano_w, pano_h],
        "crop_size": [processor.camera.width, processor.camera.height],
        "crop_focal": _f(processor.camera.params[0]),
        "num_crops": len(pano_names) * n_cams,
        "exclusive_mask_frac_per_camera": [_f(v) for v in processor.excl_frac],
        "sphere_coverage_frac": _f(processor.sphere_coverage),
        "valid_frac_mean": _f(valid.mean()),
        "valid_frac_min": _f(valid.min()),
        "valid_frac_vs_exclusive": _f(valid.mean() / float(np.mean(processor.excl_frac))),
        "best_reconstruction": int(best_idx),
        "num_reconstructions": len(recs),
        "metrics_crop": metrics,
        "metrics_equirect": equirect_metrics,
        "trajectory": equirect_traj,
        "matching": match_info,
        "timings_s": {"render": round(t_render, 2), "extract": round(t_extract, 2),
                      "match": match_info["seconds"], "mapping": round(t_map, 2)},
        "sparse_dir": str(paths.sparse / str(best_idx)),
        "equirect_dir": str(equirect_dir / str(best_idx)),
    }


def run_spherical(args, paths: Paths, wd: Watchdog) -> dict:
    import pycolmap

    all_names = list_panos(paths.images)
    pano_names = all_names[:args.max_panos] if args.max_panos else all_names
    image_dir = paths.images
    if len(pano_names) < len(all_names):
        # COLMAP walks the whole folder; stage a subset when --max_panos is used.
        image_dir = paths.out_dir / "images_subset"
        image_dir.mkdir(parents=True, exist_ok=True)
        for name in pano_names:
            shutil.copyfile(paths.images / name, image_dir / name)

    wd.stage("extract_features")
    t0 = time.time()
    reader = pycolmap.ImageReaderOptions(camera_model="EQUIRECTANGULAR")
    if paths.masks is not None:
        reader.mask_path = paths.masks
    pycolmap.extract_features(
        paths.database, image_dir, reader_options=reader,
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=pycolmap.FeatureExtractionOptions(
            use_gpu=bool(args.use_gpu), gpu_index=str(args.gpu_index),
            num_threads=int(args.num_threads)))
    t_extract = time.time() - t0

    wd.stage(f"match_{args.matcher}")
    match_info = run_matcher(args, paths.database, matching_options(args, rig=False))

    wd.stage("incremental_mapping")
    t0 = time.time()
    recs = pycolmap.incremental_mapping(paths.database, image_dir, paths.sparse,
                                        mapper_options(args, fix_rig=False))
    t_map = time.time() - t0
    if not recs:
        raise RuntimeError("incremental_mapping produced no reconstruction")
    for idx, rec in recs.items():
        print(f"[map] #{idx} {rec.summary()}", flush=True)

    best_idx = max(recs, key=lambda i: recs[i].num_reg_images())
    rec = recs[best_idx]
    metrics = reconstruction_metrics(rec, len(pano_names))
    names, centers = pano_centers(rec)
    up = estimate_up(rec)
    traj = trajectory_stats(centers, up)
    np.save(paths.qa / "pano_centers.npy", centers)
    np.save(paths.qa / "pano_up.npy", up)
    plot_trajectory(centers, paths.qa / "trajectory.png", "ARM B spherical", up)
    (paths.qa / "pano_order.json").write_text(json.dumps(names, indent=1), "utf8")
    return {
        "arm": "spherical",
        "camera_model": "EQUIRECTANGULAR",
        "pano_size": list(cv2.imread(str(image_dir / pano_names[0])).shape[1::-1]),
        "num_panos": len(pano_names),
        "best_reconstruction": int(best_idx),
        "num_reconstructions": len(recs),
        "metrics_crop": None,
        "metrics_equirect": metrics,
        "trajectory": traj,
        "matching": match_info,
        "timings_s": {"render": 0.0, "extract": round(t_extract, 2),
                      "match": match_info["seconds"], "mapping": round(t_map, 2)},
        "sparse_dir": str(paths.sparse / str(best_idx)),
        "equirect_dir": str(paths.sparse / str(best_idx)),
    }


# ---------------------------------------------------------------------------
# 5. driver
# ---------------------------------------------------------------------------

def prior_output(out_dir: Path) -> list:
    hits = []
    for rel in ("database.db", "sparse", "sfm_report.json", "images", "INCOMPLETE"):
        if (out_dir / rel).exists():
            hits.append(rel)
    return hits


def mark_incomplete(out_dir: Path, exc: BaseException) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "INCOMPLETE").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf8")
    except Exception:
        pass


def reconstruct(args) -> dict:
    import pycolmap

    out_dir = Path(args.out_dir)
    hits = prior_output(out_dir)
    if hits and not args.overwrite:
        raise FileExistsError(f"out_dir contains prior output ({', '.join(hits)}); "
                              f"pass --overwrite")
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    paths = resolve_paths(args)
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.qa.mkdir(parents=True, exist_ok=True)
    paths.sparse.mkdir(parents=True, exist_ok=True)

    pycolmap.set_random_seed(int(args.seed))
    wd = Watchdog(args.timeout_s, args.heartbeat_s).start()
    t_all = time.time()
    try:
        runner = run_crop_rig if args.arm == "crop_rig" else run_spherical
        result = runner(args, paths, wd)
    finally:
        wd.stop()

    report = {
        "wp": "WP-O2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": [sys.executable, "-m", "core.sfm_360"] + list(sys.argv[1:]),
        "dataset": str(paths.dataset),
        "image_dir": str(paths.images),
        "mask_dir": str(paths.masks) if paths.masks else None,
        "use_masks": bool(args.use_masks),
        "seed": int(args.seed),
        "colmap": {"version": pycolmap.COLMAP_version, "build": pycolmap.COLMAP_build,
                   "has_cuda": bool(pycolmap.has_cuda),
                   "use_gpu_requested": bool(args.use_gpu),
                   "num_threads": int(args.num_threads)},
        "options": {"arm": args.arm, "matcher": args.matcher,
                    "render_type": args.render_type if args.arm == "crop_rig" else None,
                    "max_panos": int(args.max_panos), "timeout_s": float(args.timeout_s)},
        **result,
    }
    report["timings_s"]["total"] = round(time.time() - t_all, 2)
    paths.report.write_text(json.dumps(report, indent=1), encoding="utf8")
    m = report["metrics_equirect"] or report["metrics_crop"]
    print(f"\n[done] arm={args.arm} registered {m['num_registered_panos']}/"
          f"{m['num_input_panos']} panos ({m['registered_pct']:.2f}%) · mean reproj "
          f"{m['mean_reproj_error_px']:.4f} px · {m['num_points3D']} points · track "
          f"{m['mean_track_length']:.2f} · {report['timings_s']['total']:.1f} s\n"
          f"       report: {paths.report}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m core.sfm_360",
        description="WP-O2 360 SfM lane: crop-rig (COLMAP panorama rig) or spherical "
                    "(EQUIRECTANGULAR direct) reconstruction of a WP-O1 pano dataset. "
                    "DEFAULT-OFF: does nothing without --capture-type 360.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--capture-type", dest="capture_type", default="off",
                   choices=["off", "360", "insta360"],
                   help="OFF by default. '360' opts into the 360 SfM lane.")
    p.add_argument("--arm", choices=list(ARMS), default="crop_rig",
                   help="crop_rig = 12 virtual pinholes per pano as one rig frame; "
                        "spherical = EQUIRECTANGULAR camera straight on the ERP frames")
    p.add_argument("--dataset", help="WP-O1 out_dir (expects images/ and masks/)")
    p.add_argument("--images", help="override <dataset>/images")
    p.add_argument("--masks", help="override <dataset>/masks")
    p.add_argument("--out_dir", help="output dir (database.db, sparse/, sfm_report.json)")
    p.add_argument("--overwrite", action="store_true",
                   help="delete out_dir first; REQUIRED to re-use a dir that already holds a "
                        "database/sparse/report (otherwise exit 2 and nothing is written)")

    g = p.add_argument_group("what to feed COLMAP")
    g.add_argument("--use_masks", type=int, default=1, choices=[0, 1],
                   help="1 = apply the WP-O1 masks at feature extraction (operator/nadir/"
                        "seams). 0 = the no-priors A/B path")
    g.add_argument("--max_panos", type=int, default=0, help="cap panos (0 = all)")
    g.add_argument("--render_type", default="perspective_overlapping",
                   choices=sorted(RENDER_PRESETS),
                   help="crop_rig only: COLMAP's virtual-camera layout")
    g.add_argument("--jpeg_quality", type=int, default=95, help="crop_rig only: crop JPEG q")
    g.add_argument("--render_workers", type=int, default=8,
                   help="crop_rig only: threads used to render crops")

    g = p.add_argument_group("features / matching / mapping")
    g.add_argument("--matcher", default="sequential",
                   choices=["sequential", "exhaustive", "vocabtree"],
                   help="sequential is rig-aware (expand_rig_images) and is what COLMAP's own "
                        "panorama pipeline defaults to; exhaustive is fine for ~150 ERP frames")
    g.add_argument("--overlap", type=int, default=10, help="sequential: neighbours per image")
    g.add_argument("--quadratic_overlap", type=int, default=1, choices=[0, 1],
                   help="sequential: neighbour offsets 1,2,4,8,... instead of 1,2,3,...")
    g.add_argument("--loop_detection", type=int, default=0, choices=[0, 1],
                   help="sequential: vocab-tree loop closure (COLMAP downloads and caches the "
                        "vocab tree on first use)")
    g.add_argument("--use_gpu", type=int, default=0, choices=[0, 1],
                   help="NO-OP in this wheel: our pycolmap 4.1.0 is built WITHOUT CUDA, so "
                        "SIFT extraction and matching are CPU either way (recorded in the "
                        "report as colmap.has_cuda)")
    g.add_argument("--gpu_index", default="-1")
    g.add_argument("--num_threads", type=int, default=-1, help="-1 = all cores")
    g.add_argument("--seed", type=int, default=0, help="COLMAP random seed")

    g = p.add_argument_group("hang protection (colmap#3599) / misc")
    g.add_argument("--timeout_s", type=float, default=21600.0,
                   help="hard wall-clock limit; on expiry all Python stacks are dumped and "
                        "the process exits 3 (0 = no limit)")
    g.add_argument("--heartbeat_s", type=float, default=60.0, help="heartbeat print interval")
    g.add_argument("--progress", type=int, default=25, help="render progress interval (panos)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.capture_type == "off":
        print("360 SfM is OFF (default). Pass --capture-type 360 to enable it.")
        return 2
    if not args.dataset and not args.images:
        print("--dataset (or --images) is required with --capture-type 360")
        return 2
    if not args.out_dir:
        print("--out_dir is required with --capture-type 360")
        return 2
    try:
        reconstruct(args)
    except FileExistsError as e:              # re-run guard: nothing was written
        print(f"ERROR: {e}")
        return 2
    except BaseException as e:
        mark_incomplete(Path(args.out_dir), e)
        if isinstance(e, (FileNotFoundError, ValueError, RuntimeError)):
            print(f"ERROR: {e}")
            return 1
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
