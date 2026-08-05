"""Tests for tools/prune_ply.py — WP-O5 opacity/outlier pruning of a 3DGS PLY.

The load-bearing claim is that the PLY reader is the exact inverse of
`gsplat.export_splats(format="ply")` — including the shN channel interleaving, which is easy
to transpose silently and would recolour every Gaussian's view-dependent term. Everything is
round-tripped through the real gsplat writer, on CPU, at tiny sizes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.prune_ply import main, prune, read_ckpt, read_gsplat_ply, splat_stats  # noqa: E402

gsplat = pytest.importorskip("gsplat")


def make_splats(n: int = 64, sh_bands: int = 15, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"means": torch.randn(n, 3, generator=g),
            "scales": torch.randn(n, 3, generator=g) * 0.3 - 2.0,
            "quats": torch.randn(n, 4, generator=g),
            "opacities": torch.randn(n, generator=g) * 2.0,
            "sh0": torch.randn(n, 1, 3, generator=g),
            "shN": torch.randn(n, sh_bands, 3, generator=g)}


def export(splats: dict, path: Path) -> Path:
    gsplat.export_splats(means=splats["means"], scales=splats["scales"],
                         quats=splats["quats"], opacities=splats["opacities"],
                         sh0=splats["sh0"], shN=splats["shN"],
                         format="ply", save_to=str(path))
    return path


def test_ply_reader_is_the_exact_inverse_of_the_gsplat_writer(tmp_path):
    src = make_splats()
    back = read_gsplat_ply(export(src, tmp_path / "a.ply"))
    for k, v in src.items():
        assert back[k].shape == v.shape, k
        assert torch.allclose(back[k], v, atol=0, rtol=0), k


def test_ply_reader_handles_sh_degree_zero(tmp_path):
    src = make_splats(n=8, sh_bands=0)
    back = read_gsplat_ply(export(src, tmp_path / "a.ply"))
    assert back["shN"].shape == (8, 0, 3)
    assert torch.allclose(back["means"], src["means"])


def test_ckpt_reader_matches_the_ply_reader(tmp_path):
    src = make_splats()
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"step": 41, "splats": {k: v.clone() for k, v in src.items()}}, ckpt)
    a, b = read_ckpt(ckpt), read_gsplat_ply(export(src, tmp_path / "a.ply"))
    for k in a:
        assert torch.allclose(a[k], b[k]), k


def test_opacity_prune_keeps_exactly_the_gaussians_above_the_threshold():
    src = make_splats(n=200)
    kept, reasons = prune(src, opacity_min=0.05, bbox_scale=0.0)
    expected = int((torch.sigmoid(src["opacities"]) >= 0.05).sum())
    assert reasons["kept"] == expected == kept["means"].shape[0]
    assert reasons["dropped_low_opacity"] == 200 - expected
    assert torch.sigmoid(kept["opacities"]).min() >= 0.05
    for k in src:
        assert kept[k].shape[0] == expected, k


def test_opacity_zero_threshold_is_a_noop():
    src = make_splats(n=32)
    kept, reasons = prune(src, opacity_min=0.0, bbox_scale=0.0)
    assert reasons["kept"] == 32 and reasons["dropped_total"] == 0
    assert torch.allclose(kept["means"], src["means"])


def test_bbox_prune_removes_a_planted_runaway():
    src = make_splats(n=100)
    src["means"][0] = torch.tensor([1e5, 1e5, 1e5])
    kept, reasons = prune(src, opacity_min=0.0, bbox_scale=3.0)
    assert reasons["dropped_spatial_outlier"] >= 1
    assert kept["means"].abs().max() < 1e4


def test_splat_stats_flags_the_wpo3_collapse_signature():
    healthy = make_splats(n=500)
    healthy["opacities"] = torch.full((500,), 0.5)          # sigmoid(0.5) = 0.62
    st = splat_stats(healthy)
    assert st["num_gaussians"] == 500
    assert st["opacity_mean"] == pytest.approx(0.6225, abs=1e-3)
    assert st["opacity_frac_below"]["0.05"] == 0.0

    dead = make_splats(n=500)
    dead["opacities"] = torch.full((500,), -12.0)           # sigmoid(-12) = 6e-6
    dead["means"] = dead["means"] * 1e5
    ds = splat_stats(dead)
    assert ds["opacity_frac_below"]["0.05"] == 1.0
    assert ds["mean_norm_p99"] > 1e4


def test_cli_writes_a_readable_pruned_ply_and_stats(tmp_path):
    src = make_splats(n=300)
    ply = export(src, tmp_path / "in.ply")
    out, stats = tmp_path / "out.ply", tmp_path / "stats.json"
    rc = main(["--ply", str(ply), "--opacity_min", "0.05", "--out", str(out),
               "--stats_json", str(stats)])
    assert rc == 0 and out.is_file()
    back = read_gsplat_ply(out)
    expected = int((torch.sigmoid(src["opacities"]) >= 0.05).sum())
    assert back["means"].shape[0] == expected
    payload = json.loads(stats.read_text())
    assert payload["before"]["num_gaussians"] == 300
    assert payload["after"]["num_gaussians"] == expected
    assert payload["pruned"]["kept"] == expected
    assert payload["opacity_min"] == 0.05
    assert Path(payload["out"]).name == "out.ply"


def test_cli_stats_only_writes_no_ply(tmp_path):
    ply = export(make_splats(n=16), tmp_path / "in.ply")
    stats = tmp_path / "stats.json"
    assert main(["--ply", str(ply), "--stats_only", "--stats_json", str(stats)]) == 0
    assert "after" not in json.loads(stats.read_text())
    assert not (tmp_path / "out.ply").exists()


def test_cli_requires_an_output_unless_stats_only(tmp_path):
    ply = export(make_splats(n=8), tmp_path / "in.ply")
    with pytest.raises(SystemExit):
        main(["--ply", str(ply)])


def test_effective_rank_endpoints():
    from tools.prune_ply import effective_rank

    log = torch.log(torch.tensor([[1.0, 1.0, 1.0],        # isotropic  -> 3
                                  [1.0, 1e-6, 1e-6],      # needle     -> 1
                                  [1.0, 1.0, 1e-6]]))     # disc       -> 2
    er = effective_rank(log)
    assert er[0] == pytest.approx(3.0, abs=1e-6)
    assert er[1] == pytest.approx(1.0, abs=1e-3)
    assert er[2] == pytest.approx(2.0, abs=1e-3)


def test_splat_stats_reports_the_needle_fraction():
    s = make_splats(n=100)
    s["scales"] = torch.log(torch.tensor([[1.0, 1e-6, 1e-6]])).repeat(100, 1)
    st = splat_stats(s)
    assert st["needle_frac_erank_lt_2"] == 1.0
    assert st["erank_mean"] == pytest.approx(1.0, abs=1e-3)

    s["scales"] = torch.zeros(100, 3)                      # exp(0) = 1 on every axis
    st = splat_stats(s)
    assert st["needle_frac_erank_lt_2"] == 0.0
    assert st["erank_mean"] == pytest.approx(3.0, abs=1e-6)
    assert st["aspect_ratio_median"] == pytest.approx(1.0)


def test_erank_prune_removes_needles_and_keeps_blobs():
    s = make_splats(n=10)
    s["scales"] = torch.log(torch.tensor(
        [[1.0, 1e-6, 1e-6]] * 4 +          # 4 needles  (erank ~ 1)
        [[1.0, 1.0, 1.0]] * 6))            # 6 isotropic (erank = 3)
    kept, reasons = prune(s, opacity_min=0.0, bbox_scale=0.0, erank_min=2.0)
    assert reasons["dropped_needle"] == 4
    assert reasons["kept"] == 6 == kept["means"].shape[0]
    assert splat_stats(kept)["needle_frac_erank_lt_2"] == 0.0


def test_erank_prune_is_off_by_default_in_the_cli(tmp_path):
    src = make_splats(n=50)
    src["scales"] = torch.log(torch.full((50, 3), 1.0)) + torch.tensor([0.0, -8.0, -8.0])
    src["opacities"] = torch.full((50,), 5.0)          # nothing dies to the opacity filter
    ply = export(src, tmp_path / "in.ply")
    out, stats = tmp_path / "out.ply", tmp_path / "stats.json"
    assert main(["--ply", str(ply), "--out", str(out), "--stats_json", str(stats)]) == 0
    payload = json.loads(stats.read_text())
    assert payload["erank_min"] == 0.0
    assert payload["after"]["num_gaussians"] == 50      # all needles, none dropped
    assert payload["before"]["needle_frac_erank_lt_2"] == 1.0
