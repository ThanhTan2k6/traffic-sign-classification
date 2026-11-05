"""Hand-crafted features for traffic-sign crops.

Feature chain (each level adds one idea on top of the previous one):

  L1  H-only            hue histogram (+1 'achromatic' bin for white/grey/black pixels)
  L2  H+S 2D            joint hue/saturation histogram
  L3  H+S grid (g x g)  one H+S histogram per cell -> adds *where* each colour is
  L4  + aspect ratio    log(w/h) of the bounding box (round/triangular vs rectangular signs)
  L5  + HOG             edge orientations (arrows, digits, shape)

Design choices:
  * V (brightness) is dropped on purpose: it changes with day/night/weather, hue/saturation much less.
  * Hue is rotated by HUE_SHIFT so red (near 0 and 179 in OpenCV) becomes one contiguous range.
  * Everything is computed inside an inscribed ellipse, so the 4 background corners of the
    rectangular bbox do not pollute the histogram.
  * Histograms are L1-normalised then square-rooted (Hellinger kernel): euclidean distance on the
    result == Hellinger distance between histograms.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog as sk_hog

HUE_SHIFT = 10      # rotate hue so red is contiguous
S_MIN = 40          # saturation below this = achromatic (white/grey/black)
H_BINS = 18         # L1: 10 degrees (OpenCV units) per bin
HS_H, HS_S = 18, 8  # L2: joint histogram bins
GRID_H, GRID_S = 12, 4  # L3: per-cell joint histogram bins
HOG_SIZE = 32


def level_specs(best_grid: int = 3) -> dict:
    g = best_grid
    return {
        "L1_H": dict(hist="h"),
        "L2_HS": dict(hist="hs"),
        "L3a_HS_grid2": dict(hist="grid", grid=2),
        "L3b_HS_grid3": dict(hist="grid", grid=3),
        f"L4_grid{g}+AR": dict(hist="grid", grid=g, ar=True),
        f"L5_grid{g}+AR+HOG": dict(hist="grid", grid=g, ar=True, hog=True),
    }


@lru_cache(maxsize=8)
def _ellipse_mask(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    r = size / 2.0
    return (((xx - c) / r) ** 2 + ((yy - c) / r) ** 2) <= 1.0


def _hs(crop_bgr: np.ndarray):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = (hsv[..., 0].astype(np.int32) + HUE_SHIFT) % 180
    s = hsv[..., 1].astype(np.int32)
    return h, s


def hist_h(crop_bgr: np.ndarray) -> np.ndarray:
    h, s = _hs(crop_bgr)
    m = _ellipse_mask(crop_bgr.shape[0])
    chroma = m & (s >= S_MIN)
    achro = m & (s < S_MIN)
    hb = np.minimum(h * H_BINS // 180, H_BINS - 1)
    v = np.bincount(hb[chroma], minlength=H_BINS).astype(np.float64)
    v = np.concatenate([v, [achro.sum()]])
    v /= max(m.sum(), 1)
    return np.sqrt(v)


def hist_hs(crop_bgr: np.ndarray) -> np.ndarray:
    h, s = _hs(crop_bgr)
    m = _ellipse_mask(crop_bgr.shape[0])
    hb = np.minimum(h * HS_H // 180, HS_H - 1)
    sb = np.minimum(s * HS_S // 256, HS_S - 1)
    idx = hb * HS_S + sb
    v = np.bincount(idx[m], minlength=HS_H * HS_S).astype(np.float64)
    v /= max(m.sum(), 1)
    return np.sqrt(v)


def hist_grid(crop_bgr: np.ndarray, g: int) -> np.ndarray:
    h, s = _hs(crop_bgr)
    size = crop_bgr.shape[0]
    m = _ellipse_mask(size)
    hb = np.minimum(h * GRID_H // 180, GRID_H - 1)
    sb = np.minimum(s * GRID_S // 256, GRID_S - 1)
    idx = hb * GRID_S + sb
    cells = []
    edges = [round(i * size / g) for i in range(g + 1)]
    for r in range(g):
        for c in range(g):
            sl = (slice(edges[r], edges[r + 1]), slice(edges[c], edges[c + 1]))
            mm = m[sl]
            v = np.bincount(idx[sl][mm], minlength=GRID_H * GRID_S).astype(np.float64)
            n = mm.sum()
            if n > 0:
                v /= n
            cells.append(v / (g * g))  # whole vector sums to ~1
    return np.sqrt(np.concatenate(cells))


def hog_feat(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA)
    return sk_hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", feature_vector=True).astype(np.float64)


def extract(crop_bgr: np.ndarray, w: float, h: float, spec: dict) -> np.ndarray:
    """Feature vector for one crop. (w, h) = size of the *original* bbox in pixels."""
    kind = spec["hist"]
    if kind == "h":
        parts = [hist_h(crop_bgr)]
    elif kind == "hs":
        parts = [hist_hs(crop_bgr)]
    elif kind == "grid":
        parts = [hist_grid(crop_bgr, int(spec["grid"]))]
    else:
        raise ValueError(f"unknown hist kind {kind}")
    if spec.get("ar"):
        parts.append(np.array([np.log(max(w, 1.0) / max(h, 1.0))]))
    if spec.get("hog"):
        parts.append(hog_feat(crop_bgr))
    return np.concatenate(parts).astype(np.float32)


def _chunk(crops, whs, spec):
    return np.stack([extract(c, w, h, spec) for c, (w, h) in zip(crops, whs)])


def extract_batch(crops: np.ndarray, whs: np.ndarray, spec: dict, n_jobs: int = -1, chunk: int = 2000) -> np.ndarray:
    """crops: (N,S,S,3) uint8 (can be a memmap); whs: (N,2) original bbox width/height."""
    n = len(crops)
    if n == 0:
        return np.zeros((0, 1), dtype=np.float32)
    ranges = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    out = Parallel(n_jobs=n_jobs)(
        delayed(_chunk)(np.asarray(crops[a:b]), np.asarray(whs[a:b]), spec) for a, b in ranges
    )
    return np.concatenate(out, axis=0)
