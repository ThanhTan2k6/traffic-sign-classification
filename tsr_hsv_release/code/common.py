"""Shared helpers for the traffic-sign pipeline (YOLO detector -> crop -> HSV classifier)."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------- groups
CCTSDB_NAMES = {0: "mandatory", 1: "prohibitory", 2: "warning"}  # order from the EDA report
COMMON3 = {"prohibitory", "mandatory", "warning"}


def vn_code(name: str) -> str:
    """'P.127*60 - Gioi han toc do' -> 'P.127*60'."""
    name = name.strip()
    if " - " in name:
        return name.split(" - ")[0].strip()
    return name.split()[0] if name else name


def vn_group(name: str) -> str:
    """Map a Vietnamese QCVN 41 sign name/code to a coarse group by its prefix."""
    code = vn_code(name)
    low = code.lower()
    if low.startswith("camera"):
        return "camera"
    m = re.match(r"[A-Za-z]+", code)
    if not m:
        return "unknown"
    p = m.group(0).upper()
    if p in ("P", "B"):  # B.8a is the old code of P.106b (prohibitory)
        return "prohibitory"
    return {"W": "warning", "R": "mandatory", "I": "information", "S": "supplementary"}.get(p, "unknown")


def to_common3(group: str) -> str:
    return group if group in COMMON3 else "other"


# ---------------------------------------------------------------- IO (unicode-safe paths on Windows)
def imread(path) -> np.ndarray | None:
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite(path, img, quality: int = 95) -> bool:
    ext = Path(path).suffix.lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink when possible (same drive), otherwise copy. Skips if dst already exists."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)


def read_yolo_label(path: Path) -> np.ndarray:
    """Return (n,5) array [cls, xc, yc, w, h]; empty (0,5) if file missing/empty."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            rows.append([float(x) for x in parts[:5]])
        except ValueError:
            continue
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def yolo_to_xyxy(row, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    _, xc, yc, w, h = row
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    x1, x2 = np.clip([x1, x2], 0, img_w)
    y1, y2 = np.clip([y1, y2], 0, img_h)
    return float(x1), float(y1), float(x2), float(y2)


def xyxy_to_yolo_line(box, img_w: int, img_h: int, cls: int = 0) -> str:
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2 / img_w
    yc = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


# ---------------------------------------------------------------- clean split (no frame leakage)
def numeric_key(stem: str):
    m = re.findall(r"\d+", stem)
    return (int(m[-1]), stem) if m else (10**12, stem)


def block_split(stems, ratios: dict, block: int = 100, gap: int = 5, seed: int = 42) -> dict:
    """Assign each frame to a split by *contiguous blocks* of frames.

    Frames are sorted by their numeric index (video order). Whole blocks go to one split, and
    frames within `gap` positions of a block boundary that separates two different splits are
    dropped (label 'gap'), so near-identical neighbouring frames never straddle train/val/test.
    """
    stems = sorted(stems, key=numeric_key)
    n = len(stems)
    n_blocks = (n + block - 1) // block
    names = list(ratios)
    probs = np.array([ratios[k] for k in names], dtype=float)
    probs /= probs.sum()
    rng = np.random.default_rng(seed)
    # deterministic proportional assignment of blocks, then shuffled
    counts = np.floor(probs * n_blocks).astype(int)
    counts[0] += n_blocks - counts.sum()
    assign = np.concatenate([[names[i]] * counts[i] for i in range(len(names))])
    rng.shuffle(assign)
    out = {}
    for i, s in enumerate(stems):
        b, off = divmod(i, block)
        sp = assign[b]
        drop = False
        if off < gap and b > 0 and assign[b - 1] != sp:
            drop = True
        if off >= block - gap and b + 1 < n_blocks and assign[b + 1] != sp:
            drop = True
        out[s] = "gap" if drop else str(sp)
    return out


# ---------------------------------------------------------------- crops
def crop_resize(img: np.ndarray, box, pad: float = 0.05, size: int = 64) -> np.ndarray:
    """Crop a (x1,y1,x2,y2) box with relative padding and resize to size x size.

    Used identically for training crops (GT boxes) and at inference time (YOLO boxes).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad
    xi1, yi1 = int(max(0, np.floor(x1))), int(max(0, np.floor(y1)))
    xi2, yi2 = int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))
    if xi2 - xi1 < 2 or yi2 - yi1 < 2:
        return np.zeros((size, size, 3), dtype=np.uint8)
    patch = img[yi1:yi2, xi1:xi2]
    interp = cv2.INTER_AREA if min(patch.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(patch, (size, size), interpolation=interp)


def jitter_box(box, rng: np.random.Generator, amount: float = 0.10):
    """Randomly perturb each box edge by up to `amount` of the box size (simulates detector noise)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    d = rng.uniform(-amount, amount, size=4)
    return (x1 + d[0] * bw, y1 + d[1] * bh, x2 + d[2] * bw, y2 + d[3] * bh)


# ---------------------------------------------------------------- matching / metrics
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (n,4) and (m,4) xyxy arrays."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def greedy_match(det_boxes, det_conf, gt_boxes, thr: float = 0.5):
    """Greedy one-to-one matching (highest confidence first). Returns {det_idx: gt_idx}."""
    det_boxes = np.asarray(det_boxes, dtype=float).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return {}
    ious = iou_matrix(det_boxes, gt_boxes)
    taken = set()
    pairs = {}
    for di in np.argsort(-np.asarray(det_conf)):
        order = np.argsort(-ious[di])
        for gi in order:
            if ious[di, gi] < thr:
                break
            if gi not in taken:
                taken.add(int(gi))
                pairs[int(di)] = int(gi)
                break
    return pairs


# ---------------------------------------------------------------- crop storage (raw uint8, low RAM)
def load_crops(data_dir, n: int, size: int = 64) -> np.ndarray:
    """Memory-mapped (N,size,size,3) uint8 array written by prepare_data.py (crops.u8)."""
    return np.memmap(Path(data_dir) / "crops.u8", dtype=np.uint8, mode="r", shape=(n, size, size, 3))
