"""WP-SH4 phase A + WP-D2 gate — measures and PERSISTS every number the record quotes.

  python _infusion/wpsh4_phaseA_gate.py            ->  _infusion/wpsh4/phaseA_gate.json

Sections:
  kernel     gsplat CUDA spherical_harmonics vs torch oracle, degrees 0..4, fwd + bwd, fp32
  rotation   tools/align_ply.py SH rotation: f'(Rd) vs f(d) via the CUDA kernel, degrees 1..4
  splat_transform   what splat-transform v2.7.1 does to a degree-4 PLY (the corruption finding)
  smoke      the --sh_degree 4 smoke train (and its sh3 twin for time/VRAM/size only)
  spz        real-model PLY -> SPZ v4 -> decode: per-attribute error, clip incidence, size
             ratio, and held-out-camera render PSNR original vs decoded (gsplat)
Seeded and deterministic apart from wall-clock/VRAM fields.
"""

import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.training import gsplat_jit_shim  # noqa: E402,F401
import gsplat  # noqa: E402
from gsplat.cuda._torch_impl import _spherical_harmonics  # noqa: E402

from core.sh_utils import ply_sh_degree, read_ply_header, truncate_ply_file  # noqa: E402
from core.spz_io import read_spz, spz_header, write_spz  # noqa: E402

SMOKE = REPO / "_dataset/Laptop/apex_images/training/wpsh4_smoke"
DATA = REPO / "_dataset/Laptop/apex_images/training/gsplat_prod/data"
OUT = REPO / "_infusion/wpsh4"
WORK = SMOKE / "_gate_work"                    # big binaries stay in the gitignored tree
DEV = torch.device("cuda")


def kernel_section():
    rows = []
    for degree in range(5):
        for unit in (True, False):
            K = (degree + 1) ** 2
            g = torch.Generator(device="cuda").manual_seed(100 + degree)
            d = torch.randn(4096, 3, device="cuda", generator=g)
            d = d / d.norm(dim=-1, keepdim=True) if unit else \
                d * (0.1 + 3.0 * torch.rand(4096, 1, device="cuda", generator=g))
            c = torch.randn(4096, K, 3, device="cuda", generator=g)
            v = torch.randn(4096, 3, device="cuda", generator=g)
            res = []
            for fn in (gsplat.spherical_harmonics, _spherical_harmonics):
                dd, cc = d.clone().requires_grad_(True), c.clone().requires_grad_(True)
                o = fn(degree, dd, cc)
                (o * v).sum().backward()
                gd = dd.grad if dd.grad is not None else torch.zeros_like(d)
                res.append((o.detach(), gd, cc.grad))
            (o1, gd1, gc1), (o2, gd2, gc2) = res
            rows.append({"degree": degree, "dirs": "unit" if unit else "raw",
                         "fwd_max_abs": (o1 - o2).abs().max().item(),
                         "fwd_ref_max": o2.abs().max().item(),
                         "dcoeffs_max_abs": (gc1 - gc2).abs().max().item(),
                         "ddirs_max_abs": (gd1 - gd2).abs().max().item(),
                         "ddirs_rel": ((gd1 - gd2).abs().max()
                                       / gd2.abs().max().clamp(min=1e-30)).item()})
    return rows


def rotation_section():
    from tools.align_ply import fibonacci_sphere, rotate_sh, sh_rotation_matrix
    rows = []
    for degree in range(1, 5):
        q, _ = np.linalg.qr(np.random.default_rng(30 + degree).normal(size=(3, 3)))
        R = q if np.linalg.det(q) > 0 else -q
        D = sh_rotation_matrix(R, degree)
        k = (degree + 1) ** 2
        c = torch.randn(1, k, 3, generator=torch.Generator().manual_seed(5)).double()
        cr = rotate_sh(c, D)
        u = fibonacci_sphere(4000, "cpu", torch.float64)
        Ru = u @ torch.as_tensor(R.T)
        f = gsplat.spherical_harmonics(degree, u.float().cuda(),
                                       c.float().expand(4000, k, 3).contiguous().cuda())
        fr = gsplat.spherical_harmonics(degree, Ru.float().cuda(),
                                        cr.float().expand(4000, k, 3).contiguous().cuda())
        off = max((D[a * a:(a + 1) ** 2, b * b:(b + 1) ** 2].abs().max().item()
                   for a in range(degree + 1) for b in range(degree + 1) if a != b),
                  default=0.0)
        orth = max(((D[a * a:(a + 1) ** 2, a * a:(a + 1) ** 2]
                     @ D[a * a:(a + 1) ** 2, a * a:(a + 1) ** 2].T
                     - torch.eye(2 * a + 1, dtype=torch.float64)).abs().max().item())
                   for a in range(degree + 1))
        rows.append({"degree": degree, "radiance_max_abs_cuda_fp32": (f - fr).abs().max().item(),
                     "offblock_max": off, "band_orthogonality_max": orth})
    return rows


def splat_transform_version(exe: str) -> str:
    """`splat-transform --version` prints to STDERR (v2.7.1), so read both streams. The first
    gate run read stdout only and recorded "" — see patch_splat_transform_version()."""
    r = subprocess.run([exe, "--version"], capture_output=True, text=True)
    return (r.stdout.strip() or r.stderr.strip())


def patch_splat_transform_version():
    """Back-fill splat_transform.version into an existing phaseA_gate.json without re-running
    the gate (review fix, 2026-10-17): `python _infusion/wpsh4_phaseA_gate.py --patch_version`."""
    path = OUT / "phaseA_gate.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    exe = shutil.which("splat-transform")
    if exe is None:
        raise SystemExit("splat-transform not on PATH")
    st = rec["splat_transform"]
    if st.get("version"):
        print(f"version already set: {st['version']!r}")
        return
    st["version"] = splat_transform_version(exe)
    st["version_note"] = ("back-filled " + time.strftime("%Y-%m-%dT%H:%M:%S") + " by "
                          "--patch_version (original run captured stdout only; the version "
                          "string is on stderr). Same installed binary; gate not re-run.")
    path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"patched splat_transform.version = {st['version']!r} -> {path}")


def splat_transform_section(src_ply: Path):
    exe = shutil.which("splat-transform")
    if exe is None:
        return {"skipped": "splat-transform not on PATH"}
    ver = splat_transform_version(exe)
    st = WORK / "splat_transform"
    st.mkdir(parents=True, exist_ok=True)
    rows = {"version": ver, "input": str(src_ply.name), "input_f_rest":
            sum(p.startswith("f_rest_") for p in read_ply_header(src_ply)[1])}
    for tag, acts in (("passthrough", []), ("filter_h3", ["--filter-harmonics", "3"]),
                      ("filter_h2", ["--filter-harmonics", "2"]),
                      ("filter_h0", ["--filter-harmonics", "0"])):
        out = st / f"{tag}.ply"
        subprocess.run([exe, "-w", "-g", "cpu", str(src_ply)] + acts + [str(out)],
                       capture_output=True, text=True, check=True)
        names = [p for p in read_ply_header(out)[1] if p.startswith("f_rest_")]
        idx = sorted(int(n[7:]) for n in names)
        try:
            deg = ply_sh_degree(out)
        except ValueError as e:
            deg = f"INVALID: {e}"
        rows[tag] = {"f_rest": len(names), "f_rest_index_min": idx[0] if idx else None,
                     "f_rest_index_max": idx[-1] if idx else None, "parsed_degree": deg}
    sog = st / "sh4.sog"
    # GPU k-means (CPU took >10 min at 500k); 2 iterations is enough to read the band count
    subprocess.run([exe, "-w", "-g", "0", "-i", "2", str(src_ply), str(sog)],
                   capture_output=True, text=True, check=True)
    import zipfile
    meta = json.loads(zipfile.ZipFile(sog).read("meta.json"))
    rows["sog_from_sh4"] = {"shN_bands": meta.get("shN", {}).get("bands")}
    # ground truth of what a correct band<=3 subset looks like
    a = _load(src_ply)
    b = _load(st / "filter_h3.ply")
    rows["filter_h3_changed_any_value"] = bool(
        b["shN"].shape != a["shN"].shape or (b["shN"] - a["shN"]).abs().max().item() > 0)
    return rows


def _load(p):
    import qa_report
    return qa_report.load_splats(str(p), "cpu")


def smoke_section():
    out = {}
    for arm in ("sh4", "sh3"):
        d = SMOKE / arm
        log = (SMOKE / f"{arm}.log").read_text(encoding="utf8", errors="replace")
        steps = re.findall(r"step=(\d+) .*?mem=(\d+)MB \| ([\d.]+) it/s", log)
        ply = d / "ply" / "point_cloud_4999.ply"
        out[arm] = {"argv": (SMOKE / f"{arm}.argv.txt").read_text().strip(),
                    "exit": (SMOKE / f"{arm}.exit.txt").read_text().strip(),
                    "results_json": json.loads((d / "stats" / "results.json").read_text()),
                    "train_seconds": float(re.search(r"Training complete in ([\d.]+)s",
                                                     log).group(1)),
                    "peak_mem_mb_logged": max(int(m) for _, m, _ in steps),
                    "final_it_per_s": float(steps[-1][2]),
                    "ply_bytes": ply.stat().st_size,
                    "ply_f_rest": sum(p.startswith("f_rest_")
                                      for p in read_ply_header(ply)[1]),
                    "ply_sh_degree": ply_sh_degree(ply)}
    out["trainer_sha256_at_run"] = (SMOKE / "trainer_sha256.txt").read_text().split()[0]
    out["note"] = ("time/VRAM comparison is indicative only: the GPU was shared with another "
                   "agent's job (nvidia-smi showed ~78 GB / 88% util from it at launch). NOT a "
                   "quality A/B — the held-out PSNR in results_json is reported, not judged.")
    return out


def render_psnr(a: dict, b: dict, data: dict, degree: int, idxs):
    from core.training.gsplat_trainer import load_image, render_splats
    A = {k: v.to(DEV) for k, v in a.items()}
    B = {k: v.to(DEV) for k, v in b.items()}
    ps = []
    for i in idxs:
        bt = load_image(data, int(i))
        H, W = int(bt["image"].shape[0]), int(bt["image"].shape[1])
        K = bt["K"].to(DEV)
        vm = torch.linalg.inv(bt["c2w"].to(DEV).float())
        bg = torch.zeros(3, device=DEV)
        with torch.no_grad():
            ra = render_splats(A, vm, K, W, H, degree, bg)[0][0][..., :3].clamp(0, 1)
            rb = render_splats(B, vm, K, W, H, degree, bg)[0][0][..., :3].clamp(0, 1)
        mse = ((ra - rb) ** 2).mean().item()
        ps.append(10 * math.log10(1.0 / max(mse, 1e-12)))
    return {"views": len(ps), "psnr_mean": float(np.mean(ps)), "psnr_min": float(np.min(ps))}


def spz_section():
    from core.training.gsplat_trainer import load_colmap_data
    data = load_colmap_data(str(DATA), factor=4, test_every=8, normalize=True)
    idxs = list(data["test_indices"])[:12]
    rows = []
    for arm in ("sh4", "sh3"):
        ply = SMOKE / arm / "ply" / "point_cloud_4999.ply"
        s = _load(ply)
        deg = ply_sh_degree(ply)
        n = s["means"].shape[0]
        clip = {"shN_abs_gt_1_frac": float((s["shN"].abs() > 1.0).float().mean()),
                "sh0_outside_pm3.33_frac": float((s["sh0"].abs() > 1 / 0.3).float().mean()),
                "scale_outside_range_frac": float(((s["scales"] < -10)
                                                   | (s["scales"] > 5.9375)).float().mean()),
                "pos_outside_pm2048_frac": float((s["means"].abs() > 2048).float().mean()),
                "shN_abs_p99": float(s["shN"].abs().flatten().quantile(0.99)
                                     if s["shN"].numel() < 16_000_000 else
                                     s["shN"].abs().flatten()[::7].quantile(0.99))}
        for bits in ((5, 4), (8, 8)):          # spz library default vs our default
            spz_path = WORK / f"{arm}_b{bits[0]}{bits[1]}.spz"
            t0 = time.time()
            rec = write_spz(s, spz_path, sh1_bits=bits[0], sh_rest_bits=bits[1])
            t_write = time.time() - t0
            d = read_spz(spz_path)
            h = spz_header(spz_path)
            pos_ok = s["means"].abs().amax(dim=1) < 2047.5
            s_cmp = {k: v[pos_ok] for k, v in s.items()}       # writer drops the rest
            assert rec["dropped_out_of_range"] == int((~pos_ok).sum())
            sc_in = (s_cmp["scales"] >= -10) & (s_cmp["scales"] <= 5.9375)
            err_in = {"means_max": (d["means"] - s_cmp["means"]).abs().max().item(),
                      "scales_max_in_range": (d["scales"] - s_cmp["scales"])[sc_in]
                      .abs().max().item(),
                      "scales_max_incl_clipped": (d["scales"] - s_cmp["scales"])
                      .abs().max().item(),
                      "quat_max_deg": math.degrees(float(2 * torch.acos(
                          (torch.nn.functional.normalize(d["quats"], dim=-1)
                           * torch.nn.functional.normalize(s_cmp["quats"], dim=-1))
                          .sum(-1).abs().clamp(max=1).min())))}
            s, s_full = s_cmp, s
            inr = s["shN"].abs() <= 0.99
            err = {**err_in,
                   "alpha_sigmoid_max": (torch.sigmoid(d["opacities"])
                                         - torch.sigmoid(s["opacities"])).abs().max().item(),
                   "sh0_max_in_range": (d["sh0"] - s["sh0"])[s["sh0"].abs() < 3.3]
                   .abs().max().item(),
                   "shN_max_in_range": (d["shN"] - s["shN"])[inr].abs().max().item(),
                   "shN_band1_max_in_range": (d["shN"][:, :3] - s["shN"][:, :3])
                   [inr[:, :3]].abs().max().item(),
                   "shN_rms": (d["shN"] - s["shN"]).pow(2).mean().sqrt().item()}
            # render: ORIGINAL (incl. the far out-of-range Gaussians) vs decoded
            r = render_psnr(s_full, d, data, deg, idxs)
            s = s_full
            rows.append({"model": arm, "sh_degree": deg, "n": n, "bits": list(bits),
                         "header": h, "ply_bytes": ply.stat().st_size,
                         "dropped_out_of_range": rec["dropped_out_of_range"],
                         "spz_bytes": rec["bytes"],
                         "ply_over_spz_ratio": ply.stat().st_size / rec["bytes"],
                         "write_seconds": round(t_write, 2), "errors": err, "clip": clip,
                         "render_psnr_orig_vs_decoded": r})
            print(f"[spz] {arm} bits {bits}: ratio {rows[-1]['ply_over_spz_ratio']:.1f}x  "
                  f"render PSNR mean {r['psnr_mean']:.2f} min {r['psnr_min']:.2f}", flush=True)
        # degree-4 master -> degree-3 twin and its SPZ (the two-consumer recipe)
        if deg == 4:
            twin = WORK / "sh4_twin_sh3.ply"
            truncate_ply_file(ply, twin, 3)
            rows.append({"model": "sh4 -> exact deg-3 twin (PLY)", "ply_bytes": twin.stat().st_size,
                         "master_ply_bytes": ply.stat().st_size,
                         "twin_over_master": twin.stat().st_size / ply.stat().st_size})
    return {"render_views": [int(i) for i in idxs], "factor": 4, "rows": rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    import spz
    rec = {"script": "_infusion/wpsh4_phaseA_gate.py", "argv": sys.argv,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "versions": {"torch": torch.__version__, "gsplat": gsplat.__version__,
                        "spz_bindings": getattr(spz, "__version__", "1.1.0 (nianticlabs/spz "
                                                "@ affd0ec, local build)"),
                        "gpu": torch.cuda.get_device_name(0)}}
    rec["kernel"] = kernel_section()
    rec["rotation"] = rotation_section()
    rec["smoke"] = smoke_section()
    rec["splat_transform"] = splat_transform_section(SMOKE / "sh4" / "ply" /
                                                     "point_cloud_4999.ply")
    rec["spz"] = spz_section()
    path = OUT / "phaseA_gate.json"
    path.write_text(json.dumps(rec, indent=1), encoding="utf8")
    print(f"wrote {path}")


if __name__ == "__main__":
    if "--patch_version" in sys.argv[1:]:
        patch_splat_transform_version()
    else:
        main()
