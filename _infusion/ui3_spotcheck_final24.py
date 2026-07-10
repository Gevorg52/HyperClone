"""UI-3 spot-check (closeout finding #4): the truthful 24-view collapse number for the
FINAL per-seed merge propagation at the [294 primary, 107 radius-45 corrective]
iteration — the configuration the RECORD credits with "23/24 = 95.8%".

Re-runs the SHIPPED gui2.sam_service.propagate (per-seed independent tracks + nearest-
anchor merge) on that 2-anchor set and scores 24 evenly-spread orbit views with the same
collapse detector the sweep used: sam_frac > 0.02 AND (sam area / bref area) >= 0.4. The
BiRefNet denominator is masks_2_birefnet_bak (the canonical mattes; per-stem fractions
match the original sweep to 3 dp).

The stale _infusion/ui3_spotcheck_metrics.json (19/24 = 79.2%) came from the REJECTED
global-reseed approach and is renamed ui3_spotcheck_REJECTED_reseed.json; this driver
banks the honest number under ui3_spotcheck_final24.json — whatever it is.

Re-runnable:  python _infusion/ui3_spotcheck_final24.py   (~2 min on the RTX PRO 6000)
"""
import json
import os
import sys

import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from gui2 import sam_service as SAM  # noqa: E402

DS = os.path.join(REPO, "_dataset", "SONY_TEST")
FRAMES = os.path.join(DS, "sam_work", "frames")
BREF = os.path.join(DS, "training", "gsplat", "data", "masks_2_birefnet_bak")
OUT_MASKS = os.path.join(DS, "sam_work", "masks_spot24")
PROMPT_JSON = os.path.join(DS, "sam_work", "prompt.json")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ui3_spotcheck_final24.json")
K = 24
SAM_MIN_FRAC = 0.02
RATIO_MIN = 0.4


def spread(n: int, k: int) -> list[int]:
    """Evenly spread k orbit indices over [0, n) — matches review.spread_indices."""
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def frac(path: str):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if m is None else float((m > 127).mean())


def main():
    prompts_all = json.load(open(PROMPT_JSON))["prompts"]
    anchors = prompts_all[:2]   # [294 primary, 107 radius=45 corrective] = the 95.8% iter
    stems = SAM.load_index_map(FRAMES)
    n = len(stems)

    stats = SAM.propagate(FRAMES, anchors, OUT_MASKS, device="cuda")
    print("[spot] propagate:", json.dumps(stats))

    idxs = spread(n, K)
    views, npass = [], 0
    for i in idxs:
        st = stems[i]
        sf = frac(os.path.join(OUT_MASKS, st + ".png"))
        bf = frac(os.path.join(BREF, st + ".png"))
        ratio = (sf / bf) if (sf is not None and bf and bf > 0) else 0.0
        ok = bool(sf is not None and sf > SAM_MIN_FRAC and ratio >= RATIO_MIN)
        npass += int(ok)
        views.append({"idx": i, "stem": st,
                      "sam_frac": round(sf, 4) if sf is not None else None,
                      "bref_frac": round(bf, 4) if bf is not None else None,
                      "ratio": round(ratio, 3), "ok": ok})

    result = {
        "config": "per-seed merge; anchors [294 primary, 107 radius=45 corrective]",
        "iteration": "seed 294 + sole corrective 107 (the RECORD's 24-view spot-check)",
        "criterion": "sam_frac>0.02 and sam/bref area ratio>=0.4 (collapse detector)",
        "bref_source": "masks_2_birefnet_bak (canonical BiRefNet mattes)",
        "propagate_seconds": stats.get("seconds"),
        "peak_vram_gb": stats.get("peak_vram_gb"),
        "seed_idxs": stats.get("seed_idxs"),
        "n": K, "pass": npass, "pct": round(100.0 * npass / K, 1),
        "views": views,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[spot] {npass}/{K} = {result['pct']}%  ->  {OUT_JSON}")


if __name__ == "__main__":
    main()
