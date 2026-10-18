"""WP-SH4 review fixes — splat-transform never sees a degree-4 (72 f_rest) PLY, and the SPZ
writer reports what it drops/clips (WP-D2).

`run_splat_transform` is stubbed (no splat-transform binary, no GPU): the stub records the
f_rest count + values of every PLY it is handed, then copies it through. Covers
  * core.delivery.main on a degree-4 master -> every splat-transform call gets 45 f_rest
    (SPLAT_TRANSFORM_MAX_SH pre-truncation, exact values), and `<name>_full_sh4.ply` keeps 72;
  * gui2.preview.export_viewer_sog on a degree-4 source -> one call with a degree-2 PLY
    (24 f_rest, exact), no `--filter-harmonics`, temp PLY cleaned up;
  * write_spz drop / clip reporting (record fields, warnings, delivery manifest + stdout).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import core.delivery as delivery  # noqa: E402
from core.sh_utils import num_rest  # noqa: E402


def synth(n=200, degree=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = lambda *s, lo, hi: lo + (hi - lo) * torch.rand(*s, generator=g)  # noqa: E731
    return {"means": u(n, 3, lo=-1.0, hi=1.0),
            "scales": u(n, 3, lo=-6.0, hi=-2.0),
            "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
            "opacities": u(n, lo=1.0, hi=4.0),
            "sh0": u(n, 1, 3, lo=-1.0, hi=1.0),
            "shN": u(n, num_rest(degree), 3, lo=-0.5, hi=0.5)}


def f_rest_count(ply) -> int:
    from core.sh_utils import read_ply_header
    return sum(p.startswith("f_rest_") for p in read_ply_header(ply)[1])


class FakeSplatTransform:
    """Stand-in for core.delivery.run_splat_transform: records + copies the input through."""

    def __init__(self):
        self.calls = []

    def __call__(self, in_ply, out_path, actions, gpu="0", iterations=None, summary=False):
        import qa_report
        self.calls.append({"in": str(in_ply), "out": str(out_path), "actions": list(actions),
                           "f_rest": f_rest_count(in_ply),
                           "shN": qa_report.load_splats(str(in_ply), "cpu")["shN"].clone()})
        if str(out_path) == "null":
            return "summary"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if str(out_path).endswith(".ply"):
            shutil.copyfile(in_ply, out_path)
        else:
            Path(out_path).write_bytes(b"fake-sog")
        return ""


def fake_data():
    c2w = np.stack([np.eye(4) for _ in range(4)])
    for i in range(4):
        c2w[i, :3, 3] = [i * 0.5, 0.0, -3.0]
    return {"camtoworlds": c2w, "train_indices": [0, 1, 2, 3], "test_indices": []}


def run_delivery(monkeypatch, tmp_path, src_ply, extra=()):
    fake = FakeSplatTransform()
    monkeypatch.setattr(delivery, "run_splat_transform", fake)
    monkeypatch.setattr(delivery, "load_colmap_data", lambda *a, **k: fake_data())
    monkeypatch.setattr(delivery, "object_center", lambda d: np.zeros(3))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "core.delivery", "--ckpt", str(src_ply), "--data_dir", str(tmp_path), "--name", "t",
        "--out", str(out), "--device", "cpu", "--gpu", "cpu", "--no_qa", "--no_decimate",
        "--opacity_min", "0", "--sor_k", "0", *extra])
    delivery.main()
    rec = json.loads((out / "t_full_delivery.json").read_text(encoding="utf-8"))
    return fake, out, rec


def write_ply(splats, path):
    delivery.export_ply(splats, str(path))
    return path


def test_delivery_pretruncates_degree4_for_splat_transform(monkeypatch, tmp_path):
    s = synth(200, 4)
    src = write_ply(s, tmp_path / "master_sh4.ply")
    assert f_rest_count(src) == 72
    fake, out, rec = run_delivery(monkeypatch, tmp_path, src)
    assert delivery.SPLAT_TRANSFORM_MAX_SH == 3
    assert fake.calls, "splat-transform was never called"
    # every splat-transform invocation sees the degree-3 layout, never 72 f_rest
    assert [c["f_rest"] for c in fake.calls] == [45] * len(fake.calls)
    first = fake.calls[0]
    assert Path(first["in"]).name == "pruned.ply"
    # exact truncation (no pruning in this run, so order is preserved)
    assert torch.equal(first["shN"], s["shN"][:, :num_rest(3)])
    assert rec["stages"]["sh_truncate_for_splat_transform"] == {"from": 4, "to": 3}
    hi = out / "t_full_sh4.ply"
    assert hi.is_file() and f_rest_count(hi) == 72
    assert rec["deliverable"]["ply_full_sh"]["sh_degree"] == 4


def test_delivery_degree3_master_unchanged(monkeypatch, tmp_path):
    src = write_ply(synth(200, 3), tmp_path / "master_sh3.ply")
    fake, out, rec = run_delivery(monkeypatch, tmp_path, src)
    assert [c["f_rest"] for c in fake.calls] == [45] * len(fake.calls)
    assert "sh_truncate_for_splat_transform" not in rec["stages"]
    assert not (out / "t_full_sh4.ply").exists()


def test_export_viewer_sog_truncates_degree4_source(monkeypatch, tmp_path):
    from gui2.preview import export_viewer_sog
    s = synth(200, 4, seed=1)
    ply_dir = tmp_path / "ply"
    src = write_ply(s, ply_dir / "point_cloud_4999.ply")
    src_bytes = src.read_bytes()
    fake = FakeSplatTransform()
    monkeypatch.setattr(delivery, "run_splat_transform", fake)
    out_sog = tmp_path / "view" / "viewer.sog"
    export_viewer_sog(str(tmp_path / "ckpt_4999_rank0.pt"), str(ply_dir), str(out_sog))
    assert len(fake.calls) == 1
    c = fake.calls[0]
    assert c["f_rest"] == 3 * num_rest(2) == 24
    assert torch.equal(c["shN"], s["shN"][:, :num_rest(2)])
    assert "--filter-harmonics" not in c["actions"] and c["actions"] == ["--morton-order"]
    assert c["out"] == str(out_sog)
    assert not Path(c["in"]).exists()                    # temp .sh2.ply removed
    assert src.read_bytes() == src_bytes                 # trainer's PLY untouched


def test_export_viewer_sog_degree3_uses_filter_harmonics(monkeypatch, tmp_path):
    from gui2.preview import export_viewer_sog
    ply_dir = tmp_path / "ply"
    write_ply(synth(100, 3), ply_dir / "point_cloud_10.ply")
    fake = FakeSplatTransform()
    monkeypatch.setattr(delivery, "run_splat_transform", fake)
    export_viewer_sog(str(tmp_path / "ckpt_10_rank0.pt"), str(ply_dir),
                      str(tmp_path / "v.sog"))
    assert fake.calls[0]["f_rest"] == 45
    assert fake.calls[0]["actions"] == ["--filter-harmonics", "2", "--morton-order"]


# ---- SPZ drop / clip reporting ---------------------------------------------------------

def test_write_spz_reports_drops_and_clip_fractions(tmp_path):
    pytest.importorskip("spz")
    from core.spz_io import SCALE_CLIP_WARN_FRAC, write_spz
    s = synth(100, 4, seed=2)
    s["means"][50] = torch.tensor([3000.0, 0.0, 0.0])          # dropped
    s["scales"][:10, 0] = -12.0                                # 10 of 300 kept values below -10
    s["scales"][10, 1] = 7.0                                   # 1 above 5.9375
    s["sh0"][:4, 0, 0] = 4.0                                   # 4 of 300 beyond 3.333
    s["shN"][:2, 0, 0] = 1.5                                   # 2 of 99*72
    rec = write_spz(s, tmp_path / "c.spz")
    assert rec["dropped_out_of_range"] == 1 and rec["n"] == 99
    cf = rec["clip_frac"]
    assert cf["scales"] == pytest.approx(11 / (99 * 3))
    assert cf["sh0"] == pytest.approx(4 / (99 * 3))
    assert cf["shN"] == pytest.approx(2 / (99 * 72))
    assert cf["scales"] > SCALE_CLIP_WARN_FRAC
    w = " | ".join(rec["warnings"])
    assert len(rec["warnings"]) == 2
    assert "dropped 1 of 100" in w and "1.0000%" in w
    assert "log-scale" in w and "3.70%" in w


def test_write_spz_clean_model_has_no_warnings(tmp_path):
    pytest.importorskip("spz")
    from core.spz_io import write_spz
    rec = write_spz(synth(100, 3, seed=3), tmp_path / "ok.spz")
    assert rec["warnings"] == [] and rec["dropped_out_of_range"] == 0
    assert rec["clip_frac"] == {"scales": 0.0, "sh0": 0.0, "shN": 0.0}


def test_delivery_surfaces_spz_drop_warning(monkeypatch, tmp_path, capsys):
    pytest.importorskip("spz")
    s = synth(200, 4, seed=4)
    s["means"][7] = torch.tensor([0.0, -5000.0, 0.0])
    src = write_ply(s, tmp_path / "far_sh4.ply")
    _, out, rec = run_delivery(monkeypatch, tmp_path, src, extra=("--spz",))
    srec = rec["deliverable"]["spz"]
    assert srec["dropped_out_of_range"] == 1 and srec["n"] == 199 and srec["sh_degree"] == 4
    assert set(srec["clip_frac"]) == {"scales", "sh0", "shN"}
    assert any("dropped 1 of 200" in w and "0.5000%" in w for w in rec["warnings"])
    txt = capsys.readouterr().out
    assert "[deliver] WARNING: SPZ dropped 1 of 200" in txt
    assert "[delivery] WARNING: SPZ dropped 1 of 200" in txt
    assert "SPZ 8-bit clipping" in txt
