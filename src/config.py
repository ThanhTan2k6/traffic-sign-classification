"""
================================================================================
CẤU HÌNH TẬP TRUNG CHO TOÀN BỘ HỆ THỐNG NHẬN DẠNG BIỂN BÁO GIAO THÔNG
Thống nhất giữa tất cả các thành viên trong nhóm (Benchmarking Framework).
================================================================================
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# UTF-8 stdout/stderr fix cho Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ==============================================================================
# 1. ĐƯỜNG DẪN DỰ ÁN (PROJECT PATHS)
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset gốc (YOLO Format: 3216 ảnh & nhãn)
DATASET_DIR = PROJECT_ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
CLASSES_FILE = DATASET_DIR / "classes.txt"
CLASSES_VIE_FILE = DATASET_DIR / "classes_vie.txt"
CLASSES_EN_FILE = DATASET_DIR / "classes_en.txt"
OFFICIAL_SPLIT_DIR = DATASET_DIR / "split_dataset"

# Thư mục dữ liệu xử lý chung
DATA_DIR = PROJECT_ROOT / "data"
CROPPED_DATA_DIR = DATA_DIR / "cropped"  # Ảnh ROI cắt cho phân loại (SVM/LeNet/CNN)
RAW_DATA_DIR = CROPPED_DATA_DIR          # Tương thích ngược với các module đánh giá
GLOBAL_SPLITS_DIR = DATA_DIR / "global_splits"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "res" / "outputs"

# Danh sách split toàn cục (Classification & Detection)
TRAIN_SPLIT_FILE = GLOBAL_SPLITS_DIR / "train_ids.txt"
VAL_SPLIT_FILE = GLOBAL_SPLITS_DIR / "val_ids.txt"
TEST_SPLIT_FILE = GLOBAL_SPLITS_DIR / "test_ids.txt"

YOLO_TRAIN_FILE = GLOBAL_SPLITS_DIR / "yolo_train.txt"
YOLO_VAL_FILE = GLOBAL_SPLITS_DIR / "yolo_val.txt"
YOLO_TEST_FILE = GLOBAL_SPLITS_DIR / "yolo_test.txt"
YOLO_DATA_YAML = DATA_DIR / "traffic_signs_4cls.yaml"

# ==============================================================================
# 2. CẤU HÌNH 4 NHÓM BIỂN BÁO MỤC TIÊU (QCVN 41:2019/BGTVT)
# ==============================================================================
PREFIX_TO_CLASS = {
    "P": "Bien_Cam",         # Biển cấm (Prohibitory) - Viền đỏ, nền trắng/xanh, hình tròn
    "W": "Bien_Nguy_Hiem",    # Biển cảnh báo (Warning) - Viền đỏ, nền vàng, tam giác đều
    "R": "Bien_Hieu_Lenh",    # Biển hiệu lệnh (Mandatory) - Nền xanh dương, hình tròn
    "I": "Bien_Chi_Dan",      # Biển chỉ dẫn (Information) - Nền xanh dương, chữ nhật/vuông
}

TARGET_CLASSES = ["Bien_Cam", "Bien_Nguy_Hiem", "Bien_Hieu_Lenh", "Bien_Chi_Dan"]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(TARGET_CLASSES)}

# Lớp loại trừ (nhiễu, biển phụ, ngoại cảnh)
IGNORED_PREFIXES = ["S", "CAMERA"]
IGNORED_CLASSES = ["Camera", ".ipynb_checkpoints"]


def get_target_category(class_name: str) -> Optional[str]:
    """
    Ánh xạ tên mã biển báo 52 lớp về 1 trong 4 nhóm mục tiêu:
    - P.*, B.8a -> Bien_Cam
    - W.* -> Bien_Nguy_Hiem
    - R.* -> Bien_Hieu_Lenh
    - I.* -> Bien_Chi_Dan
    - Còn lại (S.*, Camera) -> None (Ignored)
    """
    cleaned = class_name.strip()
    if cleaned.upper().startswith("CAMERA"):
        return None
    if cleaned.upper().startswith("B.8A") or cleaned.upper() == "B.8A":
        return "Bien_Cam"
    prefix = cleaned.split(".")[0].upper()
    return PREFIX_TO_CLASS.get(prefix, None)


def load_52_classes() -> List[str]:
    """Đọc danh sách 52 mã biển báo từ classes.txt."""
    if not CLASSES_FILE.exists():
        return []
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_52_classes_vie() -> Dict[str, str]:
    """Đọc từ điển mô tả tiếng Việt của 52 mã biển báo."""
    classes = load_52_classes()
    if not CLASSES_VIE_FILE.exists():
        return {c: c for c in classes}
    with open(CLASSES_VIE_FILE, "r", encoding="utf-8") as f:
        vie_names = [line.strip() for line in f if line.strip()]
    return {c: vie for c, vie in zip(classes, vie_names)}


# ==============================================================================
# 3. TỈ LỆ GLOBAL DATA SPLIT (70% TRAIN / 10% VAL / 20% TEST - BLOCK SPLIT)
# ==============================================================================
SPLIT_RATIOS = {
    "train": 0.70,   # 70% huấn luyện trọng số
    "val": 0.10,     # 10% tinh chỉnh siêu tham số (Tuning)
    "test": 0.20,    # 20% tập đánh giá mù (Final Benchmarking)
}
RANDOM_SEED = 42

# ==============================================================================
# 4. DANH SÁCH MÔ HÌNH BENCHMARK (2 THÀNH VIÊN)
# ==============================================================================
SUPPORTED_MODELS = [
    "color_hog_svm",  # Phương pháp 1 (Thành viên 1): HSV + Geometry + HOG + SVM
    "deep_learning",  # Phương pháp 2 (Thành viên 2): Enhanced LeNet-5 (PyTorch CNN)
]

# Kích thước ảnh chuẩn hóa cho việc trích xuất đặc trưng phân loại
IMAGE_SIZE = (64, 64)
