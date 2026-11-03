"""
Module trích xuất đặc trưng Color + HOG cho Phương pháp 1 (Thành viên 1).
- HOG (Histogram of Oriented Gradients) trên ảnh mức xám (64x64, 1764 features)
- Histogram màu trong không gian HSV (32 features: H: 16, S: 8, V: 8)
- Tổng hợp vector đặc trưng: 1796 chiều
"""

import cv2
import numpy as np

# Cấu hình HOG chuẩn cho ảnh 64x64
HOG_WIN_SIZE = (64, 64)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9

_hog_descriptor = None


def get_hog_descriptor() -> cv2.HOGDescriptor:
    global _hog_descriptor
    if _hog_descriptor is None:
        _hog_descriptor = cv2.HOGDescriptor(
            HOG_WIN_SIZE,
            HOG_BLOCK_SIZE,
            HOG_BLOCK_STRIDE,
            HOG_CELL_SIZE,
            HOG_NBINS,
        )
    return _hog_descriptor


def extract_color_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Trích xuất vector đặc trưng kết hợp (Color HSV Hist + HOG) từ ảnh ROI BGR.
    Trả về numpy array 1D độ dài 1796.
    """
    if image is None or image.size == 0:
        return np.zeros(1796, dtype=np.float32)

    # 1. Resize về kích thước chuẩn 64x64
    resized = cv2.resize(image, HOG_WIN_SIZE, interpolation=cv2.INTER_AREA)

    # 2. Trích xuất đặc trưng HOG trên ảnh xám
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hog = get_hog_descriptor()
    hog_feat = hog.compute(gray).flatten()

    # 3. Trích xuất biểu đồ màu HSV (Color Histogram)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()

    color_feat = np.concatenate([hist_h, hist_s, hist_v])
    color_norm = np.linalg.norm(color_feat)
    if color_norm > 1e-7:
        color_feat = color_feat / color_norm

    # 4. Ghép vector đặc trưng: HOG (1764) + Color (32) = 1796
    features = np.hstack([hog_feat, color_feat]).astype(np.float32)
    return features
