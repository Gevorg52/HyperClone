"""TSED — Thresholded Symmetric Epipolar Distance (vendored ~100 lines).

Reference: Watson et al., "Novel View Synthesis with Diffusion Models" / the
Photoconsistent-NVS repo (github.com/yorkucvil/Photoconsistent-NVS), MIT. We
re-implement the metric from the method (SIFT + Lowe ratio + median symmetric
epipolar distance) rather than vendoring their code, because their pose plumbing
assumes c2w OpenGL and hardcoded 256^2 intrinsics — ours are w2c OpenCV with real
per-view K.

A consecutive rendered pair (I1,pose1,K1) -> (I2,pose2,K2) is "consistent" iff
it has >= min_matches SIFT correspondences AND their MEDIAN symmetric epipolar
distance (in query-frame pixels) is below `thresh_px * (H/256)`. The fundamental
matrix comes ENTIRELY from the known poses (no RANSAC F estimation) — this is the
whole point: we test whether the render's *appearance* correspondences obey the
render's *claimed* geometry.

CONVENTION (critical): our poses are w2c OpenCV [R|t] (X_cam = R X_world + t,
+Z forward). Given two cameras (R1,t1),(R2,t2):
    X_cam2 = R2 R1^T X_cam1 + (t2 - R2 R1^T t1)
so the relative rotation/translation cam1->cam2 is
    R_rel = R2 R1^T,   t_rel = t2 - R_rel t1
    E = [t_rel]_x R_rel,   F = K2^-T E K1^-1
and for a correspondence x1 (in I1) <-> x2 (in I2): x2^T F x1 = 0.
NO OpenGL axis flip is applied (we start from OpenCV w2c directly).

Deps: numpy + opencv-contrib (SIFT). Deterministic given identical images.
"""
import cv2
import numpy as np

_EPS = 1e-8


def _skew(t):
    return np.array([[0, -t[2], t[1]],
                     [t[2], 0, -t[0]],
                     [-t[1], t[0], 0]], dtype=np.float64)


def fundamental_from_w2c(R1, t1, K1, R2, t2, K2):
    """F such that x2^T F x1 = 0, from w2c OpenCV poses + real intrinsics."""
    R1 = np.asarray(R1, np.float64); t1 = np.asarray(t1, np.float64).reshape(3)
    R2 = np.asarray(R2, np.float64); t2 = np.asarray(t2, np.float64).reshape(3)
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    E = _skew(t_rel) @ R_rel
    K1i = np.linalg.inv(np.asarray(K1, np.float64))
    K2i = np.linalg.inv(np.asarray(K2, np.float64))
    F = K2i.T @ E @ K1i
    n = np.linalg.norm(F)
    return F / n if n > _EPS else F


_sift = None


def _get_sift():
    global _sift
    if _sift is None:
        # SIFT lives in opencv-contrib; present in the main env's opencv build.
        _sift = cv2.SIFT_create()
    return _sift


def sift_matches(img1_gray, img2_gray, ratio=0.75):
    """SIFT + BF knn(2) + Lowe ratio test. Returns (pts1, pts2) Nx2 float arrays
    of matched pixel coordinates (may be empty)."""
    sift = _get_sift()
    k1, d1 = sift.detectAndCompute(img1_gray, None)
    k2, d2 = sift.detectAndCompute(img2_gray, None)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(d1, d2, k=2)
    p1, p2 = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            p1.append(k1[m.queryIdx].pt)
            p2.append(k2[m.trainIdx].pt)
    return np.asarray(p1, np.float64), np.asarray(p2, np.float64)


def symmetric_epipolar_distances(pts1, pts2, F):
    """Per-correspondence symmetric epipolar distance (pixels).

    l2 = F x1 (epiline in I2), l1 = F^T x2 (epiline in I1). The Sampson-style
    symmetric point-line distance is
        d_i^2 = (x2^T F x1)^2 * (1/(l2x^2+l2y^2) + 1/(l1x^2+l1y^2))
    We return sqrt(d^2) per match. eps-guards the line-normal division (epipole /
    degenerate lines)."""
    if len(pts1) == 0:
        return np.zeros((0,))
    x1 = np.concatenate([pts1, np.ones((len(pts1), 1))], axis=1)   # (N,3)
    x2 = np.concatenate([pts2, np.ones((len(pts2), 1))], axis=1)
    Fx1 = (F @ x1.T).T           # (N,3) epiline in I2
    Ftx2 = (F.T @ x2.T).T        # (N,3) epiline in I1
    num = np.einsum("ij,ij->i", x2, Fx1) ** 2
    d1 = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2
    d2 = Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    denom_inv = 1.0 / np.maximum(d1, _EPS) + 1.0 / np.maximum(d2, _EPS)
    return np.sqrt(num * denom_inv)


def tsed_pair(img1_gray, img2_gray, R1, t1, K1, R2, t2, K2, H,
              min_matches=10, thresh_px=2.0, ratio=0.75):
    """One consecutive pair. Returns dict with match count, median symmetric
    epipolar distance, the H-scaled threshold, and a `consistent` bool.

    consistent := (n_matches >= min_matches) AND (median_epi < thresh_px*(H/256)).
    0-match / <min_matches pairs are INCONSISTENT but reported with their true
    match count so 'textureless' can be told apart from 'geometry failure'."""
    p1, p2 = sift_matches(img1_gray, img2_gray, ratio=ratio)
    n = int(len(p1))
    thr = thresh_px * (H / 256.0)
    if n < min_matches:
        return {"n_matches": n, "median_epi": float("nan"),
                "thresh": thr, "consistent": False, "reason": "few_matches"}
    F = fundamental_from_w2c(R1, t1, K1, R2, t2, K2)
    d = symmetric_epipolar_distances(p1, p2, F)
    med = float(np.median(d))
    return {"n_matches": n, "median_epi": med, "thresh": thr,
            "consistent": bool(med < thr), "reason": "ok"}
