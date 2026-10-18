"""WP-SfM2 — Master Evaluation Driver for COLMAP / pycolmap 4.2.1 vs 4.1.0.

Executes:
  1. API & Binary Capability Audit (pycolmap 4.1.0 vs pycolmap 4.2.1 wheel vs standalone
     COLMAP 4.2.1 Windows CUDA binary).
  2. Benchmark 1 — WP-O2 ERP-Direct SfM Probe (40-frame FlowerShop equirectangular @ 1 fps,
     unmasked & masked, incremental & global, guided matching A/B, plus full 151-frame
     WP-O2 spherical DB).
  3. Benchmark 2 — WP-O5 360 Walkthrough Crop-Rig / Pano SfM (full 94-pano / 1128-crop 8K
     SIFT DB + 20-pano / 240-crop SIFT & HLoc ALIKED+LightGlue DBs + COLMAP 4.2.1 native
     ONNX LOMA_B feature/matcher evaluation).

Persists all structured metrics, argv, timings, database statistics, and Sim3 pose trajectory
comparisons to `_infusion/wpsfm2/gate.json`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "_infusion" / "wpsfm2"
SCRATCH_DIR = OUT_DIR / "_scratch"
WORKER = REPO_ROOT / "_infusion" / "wpsfm2_worker.py"

PY410 = Path(r"C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe")
PY421 = Path(r"D:\REPOS\HYPERclone\venvs\colmap421\Scripts\python.exe")
COLMAP421_EXE = Path(r"D:\REPOS\HYPERclone\venvs\colmap421\colmap_bin\bin\colmap.exe")
TORCH_LIB = r"C:\Users\gevor\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"


def _f(val, nd: int = 6) -> float:
    return round(float(val), nd)


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (scale, R, t) mapping src (Nx3) onto dst (Nx3) (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    rot = u @ s @ vt
    var_s = float((s0 ** 2).sum() / src.shape[0])
    scale = float(np.trace(np.diag(d) @ s) / var_s) if var_s > 1e-12 else 1.0
    trans = mu_d - scale * (rot @ mu_s)
    return scale, rot, trans


def kabsch_rotation(R_src_list: np.ndarray, R_dst_list: np.ndarray) -> np.ndarray:
    """Optimal global rotation R_global minimizing sum_i ||R_global @ R_src_i - R_dst_i||_F^2."""
    M = np.einsum("nij,nkj->ik", R_dst_list, R_src_list)
    u, _, vt = np.linalg.svd(M)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    return u @ s @ vt


def rotation_angles_deg(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Geodesic angles in degrees between pairs of 3x3 rotation matrices (N,3,3)."""
    dR = np.einsum("nji,njk->nik", R_a, R_b)
    tr = np.trace(dR, axis1=1, axis2=2)
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def compare_poses(
    ref_poses: dict[str, dict],
    other_poses: dict[str, dict],
    metres_per_step: float = 0.65,
    ordered_step_Reference: bool = True,
) -> dict:
    ref_names = sorted(ref_poses.keys())
    if len(ref_names) >= 2 and ordered_step_Reference:
        ref_all_c = np.array([ref_poses[n]["center"] for n in ref_names], dtype=np.float64)
        ref_step = float(np.median(np.linalg.norm(np.diff(ref_all_c, axis=0), axis=1)))
    else:
        ref_step = 1.0
    if ref_step < 1e-9:
        # If crop poses (12 per pano share center), compute step on unique pano centers
        uniq = []
        for n in ref_names:
            c = np.array(ref_poses[n]["center"], dtype=np.float64)
            if not uniq or np.linalg.norm(c - uniq[-1]) > 1e-6:
                uniq.append(c)
        if len(uniq) >= 2:
            ref_step = float(np.median(np.linalg.norm(np.diff(np.stack(uniq), axis=0), axis=1)))
        else:
            ref_step = 1.0

    common = [n for n in ref_names if n in other_poses]
    if len(common) < 3:
        return {"num_common": len(common), "status": "insufficient_common_views"}

    c_ref = np.array([ref_poses[n]["center"] for n in common], dtype=np.float64)
    c_oth = np.array([other_poses[n]["center"] for n in common], dtype=np.float64)
    R_ref = np.array([ref_poses[n]["R_c2w"] for n in common], dtype=np.float64)
    R_oth = np.array([other_poses[n]["R_c2w"] for n in common], dtype=np.float64)

    scale, rot_sim3, trans = umeyama_sim3(c_oth, c_ref)
    c_aligned = (scale * (rot_sim3 @ c_oth.T)).T + trans
    t_err = np.linalg.norm(c_aligned - c_ref, axis=1)

    # Rotation residual under Sim3 translation-derived global rotation
    R_oth_sim3 = np.einsum("ij,njk->nik", rot_sim3, R_oth)
    ang_sim3 = rotation_angles_deg(R_oth_sim3, R_ref)

    # Rotation residual under direct Kabsch global rotation alignment
    rot_kabsch = kabsch_rotation(R_oth, R_ref)
    R_oth_kabsch = np.einsum("ij,njk->nik", rot_kabsch, R_oth)
    ang_kabsch = rotation_angles_deg(R_oth_kabsch, R_ref)

    rmse_u = float(np.sqrt((t_err ** 2).mean()))
    med_u = float(np.median(t_err))
    p90_u = float(np.percentile(t_err, 90))
    max_u = float(t_err.max())
    m_per_u = metres_per_step / ref_step

    return {
        "num_common": len(common),
        "ref_step_median_units": _f(ref_step, 6),
        "metres_per_step_assumed": _f(metres_per_step, 4),
        "metres_per_ref_unit": _f(m_per_u, 6),
        "sim3_scale": _f(scale, 6),
        "trans_rmse_units": _f(rmse_u, 6),
        "trans_median_units": _f(med_u, 6),
        "trans_p90_units": _f(p90_u, 6),
        "trans_max_units": _f(max_u, 6),
        "trans_rmse_steps": _f(rmse_u / ref_step, 5),
        "trans_median_steps": _f(med_u / ref_step, 5),
        "trans_p90_steps": _f(p90_u / ref_step, 5),
        "trans_max_steps": _f(max_u / ref_step, 5),
        "trans_rmse_m": _f(rmse_u * m_per_u, 5),
        "trans_median_m": _f(med_u * m_per_u, 5),
        "trans_p90_m": _f(p90_u * m_per_u, 5),
        "trans_max_m": _f(max_u * m_per_u, 5),
        "rot_sim3_mean_deg": _f(ang_sim3.mean(), 4),
        "rot_sim3_median_deg": _f(np.median(ang_sim3), 4),
        "rot_sim3_p90_deg": _f(np.percentile(ang_sim3, 90), 4),
        "rot_sim3_max_deg": _f(ang_sim3.max(), 4),
        "rot_direct_mean_deg": _f(ang_kabsch.mean(), 4),
        "rot_direct_median_deg": _f(np.median(ang_kabsch), 4),
        "rot_direct_p90_deg": _f(np.percentile(ang_kabsch, 90), 4),
        "rot_direct_max_deg": _f(ang_kabsch.max(), 4),
    }


def scan_log_warnings(log_path: Path) -> dict:
    if not log_path.is_file():
        return {"warning_lines": 0, "ceres_cholmod_not_positive_definite": 0}
    lines = log_path.read_text(encoding="utf8", errors="replace").splitlines()
    warns = [ln for ln in lines if ln.startswith("W")]
    chol = [ln for ln in warns if "CHOLMOD warning: Matrix not positive definite" in ln]
    return {
        "warning_lines": len(warns),
        "ceres_cholmod_not_positive_definite": len(chol),
    }


def run_cmd(cmd: list[str], log_path: Path, env: dict | None = None) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"+ {' '.join(str(x) for x in cmd)}", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf8", errors="replace") as fh:
        fh.write("+ " + " ".join(str(x) for x in cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return proc.returncode, time.time() - t0


def stage_erp40_dataset(src_dataset: Path, dst_dir: Path, n_frames: int = 40, step: int = 2) -> None:
    """Stage 40 ERP frames at 1 fps (every 2nd frame of the 2 fps WP-O1 dataset) + their masks."""
    img_out = dst_dir / "images"
    msk_out = dst_dir / "masks"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted((src_dataset / "images").glob("*.jpg"))
    chosen = all_imgs[::step][:n_frames]
    if len(chosen) != n_frames:
        raise RuntimeError(f"Expected {n_frames} frames, got {len(chosen)}")
    for p in chosen:
        shutil.copyfile(p, img_out / p.name)
        mp = src_dataset / "masks" / f"{p.stem}.png"
        if mp.is_file():
            shutil.copyfile(mp, msk_out / mp.name)


def strip_poses(arm_dict: dict) -> dict:
    """Remove bulky per-crop pose dicts before writing top-level summary if needed,
    keeping poses_equirect for full reproducibility."""
    out = dict(arm_dict)
    out.pop("poses_crop", None)
    return out


def main() -> int:
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SCRATCH_DIR.exists():
        shutil.rmtree(SCRATCH_DIR)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. API & Binary Capability Audit
    # -----------------------------------------------------------------------
    print("\n=== [1/3] Inspecting pycolmap 4.1.0 & 4.2.1 APIs ===", flush=True)
    api410_json = SCRATCH_DIR / "api_410.json"
    api421_json = SCRATCH_DIR / "api_421.json"
    run_cmd([str(PY410), str(WORKER), "inspect_api", "--out", str(api410_json)], SCRATCH_DIR / "api_410.log")
    run_cmd([str(PY421), str(WORKER), "inspect_api", "--out", str(api421_json)], SCRATCH_DIR / "api_421.log")

    api410 = json.loads(api410_json.read_text(encoding="utf8"))
    api421 = json.loads(api421_json.read_text(encoding="utf8"))
    added_symbols = sorted(set(api421["all_symbols"]) - set(api410["all_symbols"]))
    removed_symbols = sorted(set(api410["all_symbols"]) - set(api421["all_symbols"]))
    api410.pop("all_symbols", None)
    api421.pop("all_symbols", None)

    api_audit = {
        "system_pycolmap_410": api410,
        "venv_pycolmap_421": api421,
        "added_top_level_symbols_in_421": added_symbols,
        "removed_top_level_symbols_in_421": removed_symbols,
        "standalone_colmap_421_cuda_binary": {
            "executable": str(COLMAP421_EXE),
            "version_banner": "COLMAP 4.2.1 (Commit bd1fcf6 on 2026-09-29 with CUDA, boost hash maps)",
            "source_archive": "https://github.com/colmap/colmap/releases/download/4.2.1/colmap-x64-windows-cuda.zip",
            "sift_gpu_supported": True,
            "onnx_cuda_provider_bundled": True,
            "onnx_requires_torch_lib_cudnn64_9_on_path": True,
            "loma_supported_via_cli": True,
            "aliked_supported_via_cli": True,
            "caspar_ba_compiled_in": False,
            "caspar_ba_note": "Both pycolmap-4.2.1 win_amd64 wheel and colmap-x64-windows-cuda.zip are compiled with CASPAR_ENABLED=OFF.",
            "ceres_gpu_ba_compiled_in": False,
            "ceres_gpu_ba_note": "Ceres in colmap-x64-windows-cuda.zip is compiled without CUDA/cuDSS solvers; falls back to CPU sparse/dense solvers.",
        },
    }

    # -----------------------------------------------------------------------
    # 2. Benchmark 1 — WP-O2 ERP-Direct SfM Probe (40 frames + 151 frames)
    # -----------------------------------------------------------------------
    print("\n=== [2/3] Running Benchmark 1: WP-O2 ERP-Direct SfM Probe ===", flush=True)
    erp40_stage = SCRATCH_DIR / "erp40_dataset"
    stage_erp40_dataset(REPO_ROOT / "_dataset/360/flowershop_077_ihfov198", erp40_stage, n_frames=40, step=2)

    b1_specs = [
        # (tag, py_exe, mapper, use_masks, guided, source_db_tag)
        ("erp40_unmasked_410_inc", PY410, "incremental", False, 0, None),
        ("erp40_unmasked_410_glo", PY410, "global", False, 0, "erp40_unmasked_410_inc"),
        ("erp40_unmasked_421_inc", PY421, "incremental", False, 0, None),
        ("erp40_unmasked_421_inc_on_410db", PY421, "incremental", False, 0, "erp40_unmasked_410_inc"),
        ("erp40_unmasked_421_glo", PY421, "global", False, 0, "erp40_unmasked_421_inc"),
        ("erp40_unmasked_421_glo_on_410db", PY421, "global", False, 0, "erp40_unmasked_410_inc"),
        ("erp40_unmasked_410_inc_guided", PY410, "incremental", False, 1, None),
        ("erp40_unmasked_421_inc_guided", PY421, "incremental", False, 1, None),
        ("erp40_masked_410_inc", PY410, "incremental", True, 0, None),
        ("erp40_masked_421_inc", PY421, "incremental", True, 0, None),
        ("erp40_masked_421_glo", PY421, "global", True, 0, "erp40_masked_421_inc"),
    ]

    b1_arms: dict[str, dict] = {}
    for tag, py_exe, mapper, use_masks, guided, src_db_tag in b1_specs:
        arm_dir = SCRATCH_DIR / tag
        log_path = SCRATCH_DIR / f"{tag}.log"
        cmd = [
            str(py_exe), str(WORKER), "run_erp",
            "--image_dir", str(erp40_stage / "images"),
            "--out_dir", str(arm_dir),
            "--mapper", mapper,
            "--guided_matching", str(guided),
            "--seed", "0",
        ]
        if use_masks:
            cmd += ["--mask_dir", str(erp40_stage / "masks")]
        if src_db_tag:
            cmd += ["--source_db", str(SCRATCH_DIR / src_db_tag / "database.db")]
        rc, wall = run_cmd(cmd, log_path)
        rfile = arm_dir / "result.json"
        res = json.loads(rfile.read_text(encoding="utf8")) if rfile.is_file() else {"status": f"crashed_rc_{rc}"}
        res["tag"] = tag
        res["returncode"] = rc
        res["wall_clock_s"] = _f(wall, 2)
        res["argv"] = cmd
        res["warnings"] = scan_log_warnings(log_path)
        b1_arms[tag] = res

    # Compare all unmasked 40-frame ERP arms against erp40_unmasked_410_inc (1 fps -> ~1.30 m/step)
    ref_erp40_unmasked = b1_arms["erp40_unmasked_410_inc"].get("poses_equirect", {})
    ref_erp40_masked = b1_arms["erp40_masked_410_inc"].get("poses_equirect", {})
    for tag, arm in b1_arms.items():
        if arm.get("status") == "ok":
            ref_p = ref_erp40_masked if arm.get("used_masks") else ref_erp40_unmasked
            arm["sim3_vs_410_inc_baseline"] = compare_poses(ref_p, arm["poses_equirect"], metres_per_step=1.30)

    gate_path = OUT_DIR / "gate.json"
    def save_partial():
        gate_path.write_text(json.dumps({
            "wp": "WP-SfM2",
            "title": "COLMAP / pycolmap 4.2.1 vs 4.1.0 Evaluation on 360 ERP & Walkthrough Rig SfM",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_eval_wall_s": _f(time.time() - t_start, 2),
            "api_and_binary_audit": api_audit,
            "benchmark1_erp40_probe": b1_arms,
            "benchmark1b_erp151_wpo2_spherical": b1_151_arms,
            "benchmark2a_wpo5_full_94pano_8k_rig": b2_full_arms,
            "benchmark2b_wpo5_20pano_parity_and_loma": b2_20p_arms,
        }, indent=1), encoding="utf8")

    b1_151_arms: dict[str, dict] = {}
    b2_full_arms: dict[str, dict] = {}
    b2_20p_arms: dict[str, dict] = {}
    save_partial()

    # Also evaluate on the full 151-frame WP-O2 spherical ERP database (_dataset/360/sfm/wpo2_spherical/database.db)
    print("\n=== [2b/3] Running Full 151-Frame WP-O2 Spherical ERP DB Comparison ===", flush=True)
    wpo2_sph_dir = REPO_ROOT / "_dataset/360/sfm/wpo2_spherical"
    erp151_banked_json = SCRATCH_DIR / "erp151_banked_410_inc.json"
    run_cmd(
        [str(PY410), str(WORKER), "inspect_existing_erp", "--run_dir", str(wpo2_sph_dir), "--out", str(erp151_banked_json)],
        SCRATCH_DIR / "erp151_banked_410_inc.log",
    )
    b1_151_arms["erp151_banked_410_inc"] = json.loads(erp151_banked_json.read_text(encoding="utf8"))
    b1_151_arms["erp151_banked_410_inc"]["tag"] = "erp151_banked_410_inc"
    b1_151_arms["erp151_banked_410_inc"]["warnings"] = scan_log_warnings(REPO_ROOT / "_dataset/360/log_wpo2_spherical.log")

    for tag, py_exe, mapper in [
        ("erp151_410_glo", PY410, "global"),
        ("erp151_421_glo", PY421, "global"),
        ("erp151_421_inc", PY421, "incremental"),
    ]:
        arm_dir = SCRATCH_DIR / tag
        log_path = SCRATCH_DIR / f"{tag}.log"
        cmd = [
            str(py_exe), str(WORKER), "run_erp",
            "--image_dir", str(REPO_ROOT / "_dataset/360/flowershop_077_ihfov198/images"),
            "--source_db", str(wpo2_sph_dir / "database.db"),
            "--out_dir", str(arm_dir),
            "--mapper", mapper,
            "--seed", "0",
        ]
        rc, wall = run_cmd(cmd, log_path)
        rfile = arm_dir / "result.json"
        res = json.loads(rfile.read_text(encoding="utf8")) if rfile.is_file() else {"status": f"crashed_rc_{rc}"}
        res["tag"] = tag
        res["returncode"] = rc
        res["wall_clock_s"] = _f(wall, 2)
        res["argv"] = cmd
        res["warnings"] = scan_log_warnings(log_path)
        if res.get("status") == "ok":
            res["sim3_vs_wpo2_410_baseline"] = compare_poses(
                b1_151_arms["erp151_banked_410_inc"]["poses_equirect"],
                res["poses_equirect"],
                metres_per_step=0.65,
            )
        b1_151_arms[tag] = res
    save_partial()

    # -----------------------------------------------------------------------
    # 3. Benchmark 2 — WP-O5 360 Walkthrough Crop-Rig / Pano SfM
    # -----------------------------------------------------------------------
    print("\n=== [3/3] Running Benchmark 2: WP-O5 360 Walkthrough Rig / Pano SfM ===", flush=True)
    wpo5_full_dir = REPO_ROOT / "_dataset/360/sfm/wpo5_theater_027_8k"
    wpo5_full_banked_json = SCRATCH_DIR / "wpo5_full_banked_410_inc.json"
    run_cmd(
        [str(PY410), str(WORKER), "inspect_existing_rig", "--run_dir", str(wpo5_full_dir), "--out", str(wpo5_full_banked_json)],
        SCRATCH_DIR / "wpo5_full_banked_410_inc.log",
    )
    b2_full_arms["wpo5_full_banked_410_inc"] = json.loads(wpo5_full_banked_json.read_text(encoding="utf8"))
    b2_full_arms["wpo5_full_banked_410_inc"]["tag"] = "wpo5_full_banked_410_inc"
    b2_full_arms["wpo5_full_banked_410_inc"]["warnings"] = scan_log_warnings(REPO_ROOT / "_dataset/360/wpo5_sfm_full.log")

    for tag, py_exe, mapper in [
        ("wpo5_full_rerun_410_inc", PY410, "incremental"),
        ("wpo5_full_421_inc", PY421, "incremental"),
        ("wpo5_full_410_glo", PY410, "global"),
        ("wpo5_full_421_glo", PY421, "global"),
    ]:
        arm_dir = SCRATCH_DIR / tag
        log_path = SCRATCH_DIR / f"{tag}.log"
        cmd = [
            str(py_exe), str(WORKER), "run_rig_map",
            "--source_db", str(wpo5_full_dir / "database.db"),
            "--crop_image_dir", str(wpo5_full_dir / "images"),
            "--pano_size", "7680x3840",
            "--n_input_panos", "94",
            "--out_dir", str(arm_dir),
            "--mapper", mapper,
            "--seed", "0",
        ]
        rc, wall = run_cmd(cmd, log_path)
        rfile = arm_dir / "result.json"
        res = json.loads(rfile.read_text(encoding="utf8")) if rfile.is_file() else {"status": f"crashed_rc_{rc}"}
        res["tag"] = tag
        res["returncode"] = rc
        res["wall_clock_s"] = _f(wall, 2)
        res["argv"] = cmd
        res["warnings"] = scan_log_warnings(log_path)
        if res.get("status") == "ok":
            res["sim3_equirect_vs_wpo5_baseline"] = compare_poses(
                b2_full_arms["wpo5_full_banked_410_inc"]["poses_equirect"],
                res["poses_equirect"],
                metres_per_step=0.65,
            )
            res["sim3_crop_vs_wpo5_baseline"] = compare_poses(
                b2_full_arms["wpo5_full_banked_410_inc"]["poses_crop"],
                res["poses_crop"],
                metres_per_step=0.65,
                ordered_step_Reference=False,
            )
        b2_full_arms[tag] = strip_poses(res)
    b2_full_arms["wpo5_full_banked_410_inc"] = strip_poses(b2_full_arms["wpo5_full_banked_410_inc"])
    save_partial()

    # 3b. WP-O5 20-pano parity guard (SIFT vs HLoc ALIKED+LightGlue vs COLMAP 4.2.1 Native LOMA_B)
    print("\n=== [3b/3] Running WP-O5 20-Pano Parity Guard (SIFT / ALIKED / LOMA_B) ===", flush=True)
    wpo5_sift_dir = REPO_ROOT / "_dataset/360/sfm/wpo5_ab_sift"
    wpo5_aliked_dir = REPO_ROOT / "_dataset/360/sfm/wpo5_ab_aliked"

    sift20_banked_json = SCRATCH_DIR / "wpo5_20p_sift_banked_410_inc.json"
    aliked20_banked_json = SCRATCH_DIR / "wpo5_20p_aliked_banked_410_inc.json"
    run_cmd(
        [str(PY410), str(WORKER), "inspect_existing_rig", "--run_dir", str(wpo5_sift_dir), "--out", str(sift20_banked_json)],
        SCRATCH_DIR / "wpo5_20p_sift_banked.log",
    )
    run_cmd(
        [str(PY410), str(WORKER), "inspect_existing_rig", "--run_dir", str(wpo5_aliked_dir), "--out", str(aliked20_banked_json)],
        SCRATCH_DIR / "wpo5_20p_aliked_banked.log",
    )

    sift20_ref = json.loads(sift20_banked_json.read_text(encoding="utf8"))
    aliked20_ref = json.loads(aliked20_banked_json.read_text(encoding="utf8"))

    b2_20p_arms["wpo5_20p_sift_banked_410_inc"] = dict(sift20_ref, tag="wpo5_20p_sift_banked_410_inc")
    b2_20p_arms["wpo5_20p_aliked_banked_410_inc"] = dict(
        aliked20_ref,
        tag="wpo5_20p_aliked_banked_410_inc",
        sim3_equirect_vs_sift_410=compare_poses(sift20_ref["poses_equirect"], aliked20_ref["poses_equirect"], 0.65),
    )

    # Build COLMAP 4.2.1 native LOMA_B database on the exact 240 crops of wpo5_ab_sift
    loma_db_dir = SCRATCH_DIR / "loma_b_20p_db"
    loma_db_dir.mkdir(parents=True, exist_ok=True)
    loma_db_path = loma_db_dir / "database.db"
    env_loma = os.environ.copy()
    env_loma["PATH"] = f"{TORCH_LIB};{env_loma.get('PATH', '')}"

    rc_ext, t_loma_ext = run_cmd(
        [
            str(COLMAP421_EXE), "feature_extractor",
            "--database_path", str(loma_db_path),
            "--image_path", str(wpo5_sift_dir / "images"),
            "--ImageReader.mask_path", str(wpo5_sift_dir / "masks"),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--ImageReader.camera_params", "960,960,960",
            "--ImageReader.single_camera_per_folder", "1",
            "--FeatureExtraction.type", "LOMA_B",
            "--FeatureExtraction.use_gpu", "1",
            "--LomaExtraction.max_num_features", "8192",
            "--default_random_seed", "0",
        ],
        SCRATCH_DIR / "loma_b_20p_extract.log",
        env=env_loma,
    )
    run_cmd(
        [
            str(PY421), str(WORKER), "prep_loma_rig",
            "--db_path", str(loma_db_path),
            "--crop_image_dir", str(wpo5_sift_dir / "images"),
            "--pano_size", "7680x3840",
        ],
        SCRATCH_DIR / "loma_b_20p_prep_rig.log",
    )
    rc_mat, t_loma_mat = run_cmd(
        [
            str(COLMAP421_EXE), "sequential_matcher",
            "--database_path", str(loma_db_path),
            "--FeatureMatching.type", "LOMA_B",
            "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.rig_verification", "1",
            "--FeatureMatching.skip_image_pairs_in_same_frame", "1",
            "--SequentialMatching.overlap", "10",
            "--SequentialMatching.quadratic_overlap", "1",
            "--SequentialMatching.expand_rig_images", "1",
            "--SequentialMatching.loop_detection", "0",
            "--default_random_seed", "0",
        ],
        SCRATCH_DIR / "loma_b_20p_match.log",
        env=env_loma,
    )

    for tag, py_exe, src_db, img_dir, mapper in [
        ("wpo5_20p_sift_421_inc", PY421, wpo5_sift_dir / "database.db", wpo5_sift_dir / "images", "incremental"),
        ("wpo5_20p_sift_421_glo", PY421, wpo5_sift_dir / "database.db", wpo5_sift_dir / "images", "global"),
        ("wpo5_20p_aliked_421_inc", PY421, wpo5_aliked_dir / "database.db", wpo5_aliked_dir / "images", "incremental"),
        ("wpo5_20p_aliked_421_glo", PY421, wpo5_aliked_dir / "database.db", wpo5_aliked_dir / "images", "global"),
        ("wpo5_20p_loma_b_421_inc", PY421, loma_db_path, wpo5_sift_dir / "images", "incremental"),
        ("wpo5_20p_loma_b_421_glo", PY421, loma_db_path, wpo5_sift_dir / "images", "global"),
    ]:
        arm_dir = SCRATCH_DIR / tag
        log_path = SCRATCH_DIR / f"{tag}.log"
        cmd = [
            str(py_exe), str(WORKER), "run_rig_map",
            "--source_db", str(src_db),
            "--crop_image_dir", str(img_dir),
            "--pano_size", "7680x3840",
            "--n_input_panos", "20",
            "--out_dir", str(arm_dir),
            "--mapper", mapper,
            "--seed", "0",
        ]
        rc, wall = run_cmd(cmd, log_path)
        rfile = arm_dir / "result.json"
        res = json.loads(rfile.read_text(encoding="utf8")) if rfile.is_file() else {"status": f"crashed_rc_{rc}"}
        res["tag"] = tag
        res["returncode"] = rc
        res["wall_clock_s"] = _f(wall, 2)
        res["argv"] = cmd
        res["warnings"] = scan_log_warnings(log_path)
        if "loma_b" in tag:
            ts = res.setdefault("timings_s", {})
            ts["extract_loma_b_gpu"] = _f(t_loma_ext, 2)
            ts["match_loma_b_gpu"] = _f(t_loma_mat, 2)
            ts["total_end_to_end"] = _f(t_loma_ext + t_loma_mat + ts.get("mapping", 0.0), 2)
        if res.get("status") == "ok":
            res["sim3_equirect_vs_sift_410"] = compare_poses(
                sift20_ref["poses_equirect"], res["poses_equirect"], metres_per_step=0.65
            )
            if "aliked" in tag or "loma_b" in tag:
                res["sim3_equirect_vs_aliked_410"] = compare_poses(
                    aliked20_ref["poses_equirect"], res["poses_equirect"], metres_per_step=0.65
                )
        b2_20p_arms[tag] = strip_poses(res)

    b2_20p_arms["wpo5_20p_sift_banked_410_inc"] = strip_poses(b2_20p_arms["wpo5_20p_sift_banked_410_inc"])
    b2_20p_arms["wpo5_20p_aliked_banked_410_inc"] = strip_poses(b2_20p_arms["wpo5_20p_aliked_banked_410_inc"])

    # Clean up bulky scratch databases/models
    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
    save_partial()
    print(f"\n=== ALL BENCHMARKS COMPLETE in {_f(time.time() - t_start, 2)}s -> {gate_path} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
