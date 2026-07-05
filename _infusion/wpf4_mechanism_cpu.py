"""WP-F4 CPU mechanism check: prove the ported planar-loss math on synthetic
depth (positive, in-front-of-camera) — nonzero terms, finite gradients that
flow, no NaN, and correct geometry. GPU-free (runs on any machine). Persisted
as gate evidence per the §0 GATE EVIDENCE RULE.

  python _infusion/wpf4_mechanism_cpu.py   # writes _infusion/wpf4_mechanism_cpu.txt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import co_planar_depth, normal_from_depth  # noqa

OUT = Path(__file__).resolve().parent / "wpf4_mechanism_cpu.txt"
lines = []


def log(s):
    print(s); lines.append(s)


def main():
    torch.manual_seed(0)
    H, W = 64, 96
    fx = fy = 80.0
    cx, cy = W / 2, H / 2
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float64)
    invK = torch.linalg.inv(K)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    rays = invK @ torch.stack([xx.reshape(-1), yy.reshape(-1),
                               torch.ones(H * W, dtype=torch.float64)], 0)
    # tilted plane facing the camera: n·X = c, z = c/(n·ray); n·ray≈-1 so c<0 -> z>0
    n = torch.tensor([0.2, -0.3, -1.0], dtype=torch.float64); n = n / n.norm()
    c = -5.0
    z = (c / (n @ rays)).reshape(H, W)
    assert (z > 0).all(), "synthetic depth must be positive (in front of camera)"
    pm = torch.ones(H, W, dtype=torch.long)

    log("== WP-F4 planar-loss CPU mechanism check ==")
    log(f"synthetic depth: {H}x{W}, range [{z.min():.3f}, {z.max():.3f}] (positive)")

    # T1: co_planar on a perfect plane -> ~0 residual
    pd = co_planar_depth(z, pm, invK)
    r1 = ((pd - z).abs() / z).mean().item()
    log(f"T1 perfect-plane L_p rel-resid = {r1:.2e}  (PASS if <1e-4)  "
        f"-> {'PASS' if abs(r1) < 1e-4 else 'FAIL'}")

    # T2: bump -> L_p>0, grads flow
    zb = z.clone(); zb[20:40, 30:60] += 0.4 * z[20:40, 30:60]
    zb = zb.detach().requires_grad_(True)
    pd2 = co_planar_depth(zb, pm, invK)
    mask = (pm > 0).float(); sc = zb.detach()[pm > 0].median() + 1e-6
    Lp = (((zb - pd2).abs() * mask).sum() / mask.sum()) / sc
    Lp.backward()
    ok2 = Lp.item() > 0 and torch.isfinite(zb.grad).all().item() and zb.grad.abs().sum() > 0
    log(f"T2 bumped   L_p = {Lp.item():.4f} (>0); grad finite+nonzero -> "
        f"{'PASS' if ok2 else 'FAIL'}")

    # T3: depth-normal matches the true plane normal; points toward camera (z<0)
    nn = normal_from_depth(z, invK)
    core = nn[8:-8, 8:-8].reshape(-1, 3)
    cos = (core @ n).abs()
    zneg = (nn[..., 2] < 0).float().mean().item()
    ok3 = cos.min().item() > 0.999 and zneg > 0.99
    log(f"T3 depth-normal vs true |cos| mean={cos.mean():.4f} min={cos.min():.4f}; "
        f"toward-cam frac(z<0)={zneg:.2f} -> {'PASS' if ok3 else 'FAIL'}")

    # T4: L_rn-style term (L1 + 1-cos), grads flow, no NaN
    zb2 = z.clone(); zb2[20:40, 30:60] += 0.2 * z[20:40, 30:60]
    zb2 = zb2.detach().requires_grad_(True)
    dn = normal_from_depth(zb2, invK)
    pnorm = nn.detach(); m = torch.ones(H, W, dtype=torch.float64)
    l1n = ((pnorm - dn).abs().sum(-1) * m).sum() / m.sum()
    l2n = ((1 - (pnorm * dn).sum(-1)) * m).sum() / m.sum()
    Lrn = l1n + l2n; Lrn.backward()
    ok4 = (Lrn.item() > 0 and torch.isfinite(Lrn).item()
           and zb2.grad.abs().sum() > 0 and torch.isfinite(zb2.grad).all().item())
    log(f"T4 L_rn = {Lrn.item():.4f} (>0); finite+grad_nonzero+no_nan -> "
        f"{'PASS' if ok4 else 'FAIL'}")

    allfin = all(torch.isfinite(x).all().item() for x in [pd, pd2, nn, dn])
    passed = (abs(r1) < 1e-4) and ok2 and ok3 and ok4 and allfin
    log(f"ALL OUTPUTS FINITE: {allfin}")
    log(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
