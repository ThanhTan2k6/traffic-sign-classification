"""
================================================================================
TRÍCH XUẤT ĐẶC TRƯNG MÀU SẮC HSV + GRID + HÌNH HỌC + HOG (STAGE 2 S3)
Thành viên 1: Phương pháp Thị giác máy tính truyền thống (QCVN 41:2019)
================================================================================
Chuỗi đặc trưng:
- S0: Biểu đồ Histogram Hue (36 bins dịch +30°) + Saturation (8 bins) -> 44 chiều
- S1: S0 + Tỷ lệ khung hình (Aspect Ratio w/h) -> 45 chiều
- S2 (Lưới không gian 2x2): 4 ô cục bộ (18 Hue + 4 Sat) -> 88 chiều + S1 = 133 chiều
- S3 (+HOG): Đặc trưng gradient định hướng HOG (64x64, 9 hướng) -> 324 chiều
Tổng cộng: 457 chiều
================================================================================
"""

from typing import Tuple
import cv2
import numpy as np
from skimage.feature import hog


def extract_s3_features(
    crop_bgr: np.ndarray,
    w_orig: float = None,
    h_orig: float = None,
    grid: str = "2x2",
) -> np.ndarray:
    """
    Trích xuất vector đặc trưng S3 (457 chiều) cho 1 ảnh ROI biển báo.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((457,), dtype=np.float64)

    h_img, w_img = crop_bgr.shape[:2]
    w_orig = float(w_orig if w_orig is not None else w_img)
    h_orig = float(h_orig if h_orig is not None else h_img)

    # 1. S0: HSV Global Histogram
    c32 = cv2.resize(crop_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    hsv32 = cv2.cvtColor(c32, cv2.COLOR_BGR2HSV)
    h32 = ((hsv32[:, :, 0].astype(np.int32) + 30) % 180).astype(np.uint8)
    s32 = hsv32[:, :, 1]

    hist_h = cv2.calcHist([h32], [0], None, [36], [0, 180]).flatten()
    hist_s = cv2.calcHist([s32], [0], None, [8], [0, 256]).flatten()
    s0 = np.concatenate([hist_h, hist_s])
    s0 = s0 / (np.sum(s0) + 1e-7)  # 44 dims

    # 2. S1: Aspect ratio w/h
    ar = float(w_orig) / float(max(h_orig, 1.0))
    s1 = np.append(s0, ar)  # 45 dims

    # 3. S2: Lưới không gian 2x2 (4 cells of 16x16)
    cells_2x2 = []
    for r in range(2):
        for c in range(2):
            cell_h = h32[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16]
            cell_s = s32[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16]
            hh = cv2.calcHist([cell_h], [0], None, [18], [0, 180]).flatten()
            ss = cv2.calcHist([cell_s], [0], None, [4], [0, 256]).flatten()
            c_feat = np.concatenate([hh, ss])
            c_feat = c_feat / (np.sum(c_feat) + 1e-7)
            cells_2x2.append(c_feat)
    s2_2x2 = np.concatenate([s1, np.concatenate(cells_2x2)])  # 133 dims

    # 4. HOG (64x64, 9 orientations, 16x16 pixels per cell, 2x2 cells per block) -> 324 dims
    c64 = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray64 = cv2.cvtColor(c64, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray64,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )  # 324 dims

    s3 = np.concatenate([s2_2x2, hog_feat])  # 457 dims
    return s3
