"""
File cấu hình chung cho toàn bộ dự án Nhận dạng & Phân loại Biển báo Giao thông.
Được chia sẻ giữa 3 thành viên để đảm bảo tính nhất quán của hệ thống.
"""

from pathlib import Path
import numpy as np

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC (PROJECT PATHS)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / "dataset_cropped_for_lenet"  # Hoặc data/raw
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"
ROI_OUTPUT_DIR = DATA_DIR / "rois"

MODEL_DIR = BASE_DIR / "models_saved"
MODEL_PATH = MODEL_DIR / "traffic_sign_svm.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

# ==============================================================================
# 2. CẤU HÌNH TIỀN XỬ LÝ & KÍCH THƯỚC ROI (THÀNH VIÊN 1 & 2)
# ==============================================================================
# Kích thước chuẩn hóa của vùng ROI sau khi cắt (cho HOG và Model)
ROI_SIZE = (64, 64)  # (width, height)

# Thư mục / nhãn cần loại bỏ trong quá trình làm sạch
IGNORED_CLASSES = ["Camera", ".ipynb_checkpoints"]

# ==============================================================================
# 3. CẤU HÌNH KHÔNG GIAN MÀU HSV (GIAI ĐOẠN 1 - COLOR MASKING)
# ==============================================================================
# Lưu ý trong OpenCV: H: [0, 180], S: [0, 255], V: [0, 255]
HSV_COLOR_RANGES = {
    # Màu đỏ bị ngắt ở 180 và 0 nên cần 2 dải
    "red": [
        {
            "lower": np.array([0, 50, 40], dtype=np.uint8),
            "upper": np.array([10, 255, 255], dtype=np.uint8),
        },
        {
            "lower": np.array([170, 50, 40], dtype=np.uint8),
            "upper": np.array([180, 255, 255], dtype=np.uint8),
        },
    ],
    # Màu xanh dương (Biển hiệu lệnh R, chỉ dẫn I)
    "blue": [
        {
            "lower": np.array([95, 50, 40], dtype=np.uint8),
            "upper": np.array([135, 255, 255], dtype=np.uint8),
        }
    ],
    # Màu vàng (Biển cảnh báo nguy hiểm W)
    "yellow": [
        {
            "lower": np.array([15, 50, 50], dtype=np.uint8),
            "upper": np.array([35, 255, 255], dtype=np.uint8),
        }
    ],
}

# ==============================================================================
# 4. CẤU HÌNH PHÂN ĐOẠN & TÌM ROI (GIAI ĐOẠN 2 - MORPHOLOGY & CONTOURS)
# ==============================================================================
# Kích thước kernel hình thái học
MORPH_KERNEL_SIZE = (5, 5)
MORPH_OPEN_ITERS = 1
MORPH_CLOSE_ITERS = 2

# Điều kiện lọc Contours phát hiện biển báo
MIN_CONTOUR_AREA = 100       # Diện tích tối thiểu để loại bỏ nhiễu hạt
MAX_CONTOUR_AREA = 500000    # Diện tích tối đa (tránh nhầm khung cảnh lớn)
MIN_ASPECT_RATIO = 0.5       # Tỷ lệ chiều rộng / chiều cao (width / height)
MAX_ASPECT_RATIO = 2.0       # Biển báo thường gần vuông, tròn hoặc tam giác đều

# ==============================================================================
# 5. CẤU HÌNH TRÍCH XUẤT ĐẶC TRƯNG HOG (GIAI ĐOẠN 3 - THÀNH VIÊN 2)
# ==============================================================================
HOG_CONFIG = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "visualize": False,
    "transform_sqrt": True,
}

# ==============================================================================
# 6. CẤU HÌNH HUẤN LUYỆN MÔ HÌNH (GIAI ĐOẠN 4 - THÀNH VIÊN 2)
# ==============================================================================
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
SVM_KERNEL = "rbf"
SVM_C = 10.0
SVM_GAMMA = "scale"
