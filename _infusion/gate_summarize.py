"""Collect a per-arm gate table (WP-Q1 scorecard + region + floaters + prune + off-path fog) into
one metrics JSON + a markdown table. Used by STEP 0 (opacity_reg), WP-B2 and WP-F5 records.

  python _infusion/gate_summarize.py --prefix qa/opreg0_laptop_ --arms or0 or01 wps1 \
      --offpath qa/opreg0_laptop_offpath_or0_or01_wps1/offpath.json --out _infusion/opreg0_laptop_gate.json
"""
import argparse
import json
import re
from pathlib import Path


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf8")) if p.is_file() else None


def floaters(p):
    p = Path(p)
    if not p.is_file():
        return None
    t = p.read_text(encoding="utf8", errors="replace")
    m = re.search(r"needle\s+([\d.]+)%\s+SOR-outlier\s+([\d.]+)%\s+giant\s+([\d.]+)%\s+low-opa\s+"
                  r"([\d.]+)%\s+med-opa\s+([\d.]+)\s+mean-erank\s+([\d.]+)", t)
    if not m:
        return None
    k = ("needle_pct", "sor_pct", "giant_pct", "low_opa_pct", "med_opa", "mean_erank")
    return dict(zip(k, map(float, m.groups())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="qa dir prefix, arm name appended")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--offpath", default=None)
    ap.add_argument("--region", default="region_deck.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    off = load(a.offpath) if a.offpath else None
    rows = {}
    for arm in a.arms:
        d = Path(a.prefix + arm)
        sc = load(d / "scorecard.json") or {}
        s = sc.get("scorers", {})
        reg = load(d / a.region)
        reg_arm = next(iter(reg["arms"].values())) if reg else {}
        pr = load(d / "prune_stats.json") or {}
        row = {
            "masked_psnr": sc.get("masked_psnr"), "masked_ssim": sc.get("masked_ssim"),
            "masked_lpips": sc.get("masked_lpips"), "n_views": sc.get("n_views"),
            "crossscore": s.get("crossscore", {}).get("mean"),
            "nova": s.get("nova", {}).get("mean"), "tsed": s.get("tsed", {}).get("ratio"),
            "raft_warp": s.get("raft_warp", {}).get("mean"),
            "nvs_sqa_jod": s.get("nvs_sqa", {}).get("jod"),
            "flicker": s.get("flicker", {}).get("score"),
            "region_name": reg.get("region_name") if reg else None,
            "region_lpips": reg_arm.get("region_lpips"), "region_psnr": reg_arm.get("region_psnr"),
            "floaters": floaters(d / "floaters.txt"),
            "prune": ({"before_n": pr["before"]["num_gaussians"],
                       "after_n": pr["after"]["num_gaussians"], "ply": pr.get("out")}
                      if pr.get("before") and pr.get("after") else None),
        }
        if off and arm in off["arms"]:
            oa = off["arms"][arm]
            st = oa["ckpt_stats"]
            row["offpath_fog"] = oa["fog_proxy_mean"]
            row["offpath_fog_by_kind"] = oa["fog_proxy_by_kind"]
            row["ckpt_opacity_mean"] = st.get("opacity_mean")
            row["ckpt_opacity_frac_below_0.05"] = st.get("opacity_frac_below", {}).get("0.05")
            row["ckpt_outliers_beyond_3x_bbox"] = st.get("outliers_beyond_3x_bbox")
            row["ckpt_needle_frac_erank_lt_2"] = st.get("needle_frac_erank_lt_2")
        rows[arm] = row
    out = {"arms": rows, "sources": {"prefix": a.prefix, "offpath": a.offpath}}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf8")

    def f(x, n=4):
        return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
    cols = [("mLPIPS", lambda r: f(r["masked_lpips"])), ("mPSNR", lambda r: f(r["masked_psnr"], 3)),
            ("mSSIM", lambda r: f(r["masked_ssim"])),
            ("regLPIPS", lambda r: f(r["region_lpips"])),
            ("xscore", lambda r: f(r["crossscore"])), ("nova", lambda r: f(r["nova"])),
            ("tsed", lambda r: f(r["tsed"], 3)), ("raft", lambda r: f(r["raft_warp"])),
            ("jod", lambda r: f(r["nvs_sqa_jod"], 3)),
            ("op<0.05", lambda r: f(r.get("ckpt_opacity_frac_below_0.05"))),
            ("op_mean", lambda r: f(r.get("ckpt_opacity_mean"))),
            ("foglayerA", lambda r: f((r.get("offpath_fog") or {}).get("fog_layer_alpha_mean"))),
            ("prune_dMAE", lambda r: f((r.get("offpath_fog") or {}).get("prune_delta_mae"), 3)),
            ("needle%", lambda r: f((r["floaters"] or {}).get("needle_pct"), 2)),
            ("giant%", lambda r: f((r["floaters"] or {}).get("giant_pct"), 2))]
    print("| arm | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for arm, r in rows.items():
        print(f"| {arm} | " + " | ".join(fn(r) for _, fn in cols) + " |")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
