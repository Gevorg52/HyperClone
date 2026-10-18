"""Spherical-harmonic degree bookkeeping shared by every reader/writer (WP-SH4).

One place that knows the relation  rest coeffs per channel = (degree + 1)^2 - 1, so no tool
silently assumes degree 3 again. Degrees 0..4 are what gsplat's kernel supports (its torch
oracle returns garbage above 25 bases) and what our Unreal plugin reads; Postshot is assumed to
cap at 3.

PLY convention (INRIA / gsplat `export_splats`): `f_rest_*` is CHANNEL-MAJOR — all rest coeffs
of R, then all of G, then all of B, i.e. `shN.permute(0, 2, 1).reshape(N, -1)` for an
(N, K-1, 3) tensor. At degree 4 that is 3 x 24 = 72 f_rest properties.
"""

from __future__ import annotations

MAX_SH_DEGREE = 4


def num_rest(degree: int) -> int:
    """Rest (non-DC) coeffs per colour channel for an SH degree."""
    if not 0 <= int(degree) <= MAX_SH_DEGREE:
        raise ValueError(f"SH degree {degree} outside supported 0..{MAX_SH_DEGREE}")
    return (int(degree) + 1) ** 2 - 1


def sh_degree_from_rest(n_rest_per_channel: int) -> int:
    """Exact inverse of `num_rest`. Raises on a count that is not (d+1)^2 - 1."""
    n = int(n_rest_per_channel)
    for d in range(MAX_SH_DEGREE + 1):
        if num_rest(d) == n:
            return d
    raise ValueError(f"{n} rest SH coeffs per channel is not (d+1)^2-1 for any d in "
                     f"0..{MAX_SH_DEGREE}")


def sh_degree_from_f_rest(n_f_rest: int) -> int:
    """Degree from the total number of `f_rest_*` PLY properties (3 channels)."""
    if int(n_f_rest) % 3:
        raise ValueError(f"{n_f_rest} f_rest properties is not a multiple of 3")
    return sh_degree_from_rest(int(n_f_rest) // 3)


def sh_degree_of(splats: dict) -> int:
    """Degree implied by a splats dict's `shN` (N, K-1, 3)."""
    return sh_degree_from_rest(int(splats["shN"].shape[1]))


def read_ply_header(path) -> tuple[int, list[str], int]:
    """(n_vertex, float property names in order, header byte length) of a binary-LE,
    all-float32 vertex PLY (what gsplat / splat-transform / INRIA write). Raises otherwise."""
    import re

    with open(path, "rb") as fh:
        head = b""
        while not head.endswith(b"end_header\n"):
            ch = fh.read(1)
            if not ch:
                raise ValueError(f"{path}: no end_header")
            head += ch
            if len(head) > 1 << 20:
                raise ValueError(f"{path}: header > 1 MB")
    text = head.decode("ascii", "replace")
    if "format binary_little_endian" not in text:
        raise ValueError(f"{path}: only binary_little_endian PLYs are supported")
    elements = re.findall(r"^element (\S+) (\d+)", text, flags=re.M)
    if [e[0] for e in elements] != ["vertex"]:
        raise ValueError(f"{path}: expected a single 'vertex' element, got {elements}")
    props = re.findall(r"^property (\S+) (\S+)", text, flags=re.M)
    bad = [p for p in props if p[0] not in ("float", "float32")]
    if bad:
        raise ValueError(f"{path}: non-float properties not supported: {bad[:3]}")
    return int(elements[0][1]), [p[1] for p in props], len(head)


def ply_sh_degree(path) -> int:
    """SH degree from a PLY header's f_rest count (exact; raises on a non-(d+1)^2-1 count)."""
    _, props, _ = read_ply_header(path)
    return sh_degree_from_f_rest(sum(1 for p in props if p.startswith("f_rest_")))


def truncate_ply_file(src, dst, degree: int) -> dict:
    """Write `dst` = `src` with SH bands above `degree` removed — EXACTLY.

    Byte-level: every other property (and its float32 bits, and the row order) is copied
    unchanged; of the channel-major f_rest block it keeps, per channel c, the first
    (degree+1)^2-1 coeffs, renumbered f_rest_0.. in the same channel-major layout. Because SH
    bands are orthogonal, the result renders identically to `src` rendered at `degree` (the
    'Postshot twin' of a degree-4 master). This exists because splat-transform v2.7.1
    `--filter-harmonics` assumes a degree-3 layout: on a degree-4 PLY `3` is a no-op and `2`/`0`
    give invalid f_rest counts (WP-SH4)."""
    import numpy as np

    n, props, hlen = read_ply_header(src)
    rest = sorted((p for p in props if p.startswith("f_rest_")), key=lambda s: int(s[7:]))
    have = sh_degree_from_f_rest(len(rest))
    if degree > have:
        raise ValueError(f"{src}: degree {have} cannot be truncated up to {degree}")
    k_old, k_new = num_rest(have), num_rest(degree)
    raw = np.fromfile(src, dtype="<f4", offset=hlen, count=n * len(props))
    if raw.size != n * len(props):
        raise ValueError(f"{src}: truncated body ({raw.size} of {n * len(props)} floats)")
    raw = raw.reshape(n, len(props))
    col = {p: i for i, p in enumerate(props)}
    keep_rest = [f"f_rest_{c * k_old + k}" for c in range(3) for k in range(k_new)]
    out_cols, out_names, j = [], [], 0
    for p in props:
        if p.startswith("f_rest_"):
            if p == rest[0]:                       # emit the whole new block at the old spot
                for name in keep_rest:
                    out_cols.append(col[name])
                    out_names.append(f"f_rest_{j}")
                    j += 1
            continue
        out_cols.append(col[p])
        out_names.append(p)
    body = np.ascontiguousarray(raw[:, out_cols]).astype("<f4", copy=False)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in out_names) + "end_header\n")
    with open(dst, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(body.tobytes())
    return {"src": str(src), "dst": str(dst), "n": n, "degree_in": have, "degree_out": degree,
            "f_rest_in": len(rest), "f_rest_out": 3 * k_new}


def truncate_shN(shN, degree: int):
    """Keep the first (degree+1)^2-1 rest coeffs (bands 1..degree). EXACT: SH bands are
    orthogonal, so rendering the truncated model equals rendering the original at `degree`."""
    have = sh_degree_from_rest(int(shN.shape[1]))
    if degree > have:
        raise ValueError(f"cannot truncate degree {have} up to {degree}")
    return shN[:, :num_rest(degree)].contiguous()


def resize_shN(shN, degree: int):
    """Resize `shN` (N, K-1, 3) to `degree` (0..4): truncate if `degree < have`, zero-pad higher
    bands if `degree > have`, or return as-is when equal. Used by `--init_ckpt` warm-start."""
    have = sh_degree_from_rest(int(shN.shape[1]))
    k_new = num_rest(degree)
    if k_new == shN.shape[1]:
        return shN
    if k_new < shN.shape[1]:
        return shN[:, :k_new].contiguous()
    import torch
    pad = torch.zeros(
        (shN.shape[0], k_new - shN.shape[1], shN.shape[2]),
        dtype=shN.dtype,
        device=shN.device,
    )
    return torch.cat([shN, pad], dim=1)
