"""Truncate a degree-N 3DGS PLY to SH degree M exactly (WP-SH4 'Postshot twin').

Keeps the first (M+1)^2-1 rest coeffs per channel (channel-major INRIA layout), copies every
other property bit-for-bit, preserves row order. Rendering the output equals rendering the
input at degree M (bands are orthogonal) — `tests/test_sh_ply_io.py` checks it numerically.

Use this, NOT splat-transform `--filter-harmonics`, on a degree-4 PLY: splat-transform v2.7.1
assumes the degree-3 layout: on a 72-f_rest file `3` is a no-op (keeps 72) and `2`/`0` give
invalid property counts (51/27) (measured, WP-SH4).

  python tools/truncate_sh_ply.py master_sh4.ply postshot_sh3.ply --degree 3
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.sh_utils import truncate_ply_file  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python tools/truncate_sh_ply.py")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--degree", type=int, required=True, choices=[0, 1, 2, 3, 4])
    args = p.parse_args(argv)
    if Path(args.src).resolve() == Path(args.dst).resolve():
        p.error("dst must differ from src")
    print(json.dumps(truncate_ply_file(args.src, args.dst, args.degree)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
