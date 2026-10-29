"""
Cấu hình tập trung cho toàn bộ Dự án Benchmarking Nhận dạng Biển báo Giao thông.
Được chia sẻ thống nhất giữa tất cả các thành viên trong nhóm.
"""

from pathlib import Path

# ==============================================================================
# 1. ĐƯỜNG DẪN DỰ ÁN (PROJECT PATHS)
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "dataset_cropped_for_lenet"  # Hoặc data/raw
GLOBAL_SPLITS_DIR = DATA_DIR / "global_splits"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "res" / "outputs"

# File phân chia dữ liệu toàn cục (Global Data Splits)
TRAIN_SPLIT_FILE = GLOBAL_SPLITS_DIR / "train_ids.txt"
VAL_SPLIT_FILE = GLOBAL_SPLITS_DIR / "val_ids.txt"
TEST_SPLIT_FILE = GLOBAL_SPLITS_DIR / "test_ids.txt"

# ==============================================================================
# 2. CẤU HÌNH NHÓM BIỂN BÁO (4 CLASSES CHUẨN QCVN 41:2019/BGTVT)
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

# Các thư mục hoặc lớp cần loại bỏ (nhiễu/ngoại cảnh/quá ít mẫu)
IGNORED_CLASSES = ["Camera", "B.8a", ".ipynb_checkpoints"]

# ==============================================================================
# 3. TỈ LỆ GLOBAL DATA SPLIT (3 TẬP ĐỘC LẬP: TRAIN / VAL / TEST)
# ==============================================================================
SPLIT_RATIOS = {
    "train": 0.70,   # 70% dùng để huấn luyện mô hình
    "val": 0.15,     # 15% dùng để tinh chỉnh siêu tham số (Tuning)
    "test": 0.15,    # 15% tập đánh giá mù (Final Benchmarking)
}
RANDOM_SEED = 42

# ==============================================================================
# 4. DANH SÁCH MÔ HÌNH HỖ TRỢ TRONG HỆ THỐNG BENCHMARK
# ==============================================================================
SUPPORTED_MODELS = [
    "color_hog_svm",  # Phương pháp 1 (Thành viên 1): Lọc màu HSV + Morphology + HOG + SVM
    "model_2",        # Phương pháp 2 (Thành viên 2): Template tự chọn (VD: Custom CNN / LeNet)
    "model_3",        # Phương pháp 3 (Thành viên 3): Template tự chọn (VD: YOLO / MobileNet / KNN)
]

# Kích thước ảnh chuẩn hóa cho việc trích xuất đặc trưng
IMAGE_SIZE = (64, 64)
