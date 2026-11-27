"""Shared metadata, presets, and benchmark tables for the Streamlit demo."""

from pathlib import Path
from src.config import PROJECT_ROOT, TARGET_CLASSES, RESULTS_DIR, CHECKPOINTS_DIR, DATASET_DIR

CLASS_METADATA = {
    "Bien_Cam": {
        "vn_name": "Biển Báo Cấm",
        "code_prefix": "P.*",
        "color_hex": "#EF4444",
        "color_rgb": (239, 68, 68),
        "color_bgr": (68, 68, 239),
        "badge_color": "red",
        "shape": "Hình tròn, viền đỏ, nền trắng hoặc xanh, hình vẽ đen/trắng",
        "description": "Báo hiệu các điều cấm mà người tham gia giao thông không được vi phạm.",
    },
    "Bien_Nguy_Hiem": {
        "vn_name": "Biển Nguy Hiểm",
        "code_prefix": "W.*",
        "color_hex": "#F59E0B",
        "color_rgb": (245, 158, 11),
        "color_bgr": (11, 158, 245),
        "badge_color": "orange",
        "shape": "Hình tam giác đều, viền đỏ, nền vàng, hình vẽ màu đen",
        "description": "Cảnh báo trước các tình huống nguy hiểm có thể xảy ra trên đoạn đường.",
    },
    "Bien_Hieu_Lenh": {
        "vn_name": "Biển Hiệu Lệnh",
        "code_prefix": "R.*",
        "color_hex": "#3B82F6",
        "color_rgb": (59, 130, 246),
        "color_bgr": (246, 130, 59),
        "badge_color": "blue",
        "shape": "Hình tròn, nền xanh lam, hình vẽ màu trắng",
        "description": "Báo hiệu các hiệu lệnh người tham gia giao thông bắt buộc phải thi hành.",
    },
    "Bien_Chi_Dan": {
        "vn_name": "Biển Chỉ Dẫn",
        "code_prefix": "I.*",
        "color_hex": "#10B981",
        "color_rgb": (16, 185, 129),
        "color_bgr": (129, 185, 16),
        "badge_color": "green",
        "shape": "Hình chữ nhật hoặc hình vuông, nền xanh lam/xanh lá, hình vẽ trắng",
        "description": "Hướng dẫn thông tin cần thiết về đường sá, địa danh, tiện ích giao thông.",
    },
    "Unknown": {
        "vn_name": "Không Xác Định",
        "code_prefix": "?",
        "color_hex": "#94A3B8",
        "color_rgb": (148, 163, 184),
        "color_bgr": (184, 163, 148),
        "badge_color": "gray",
        "shape": "Không xác định",
        "description": "Mô hình chưa đủ độ tin cậy để kết luận nhóm biển báo.",
    },
}

MODEL_OPTIONS = {
    "lenet": "Enhanced LeNet-5",
    "svm": "Color + HOG + SVM",
    "compare": "So sánh 2 mô hình",
}

PRESET_SCENES = {
    "scene_0001": {
        "label": "Cảnh 0001 (1 biển)",
        "caption": "0001.jpg — 1 biển báo đơn lẻ",
        "path": DATASET_DIR / "images" / "0001.jpg",
    },
    "scene_0756": {
        "label": "Cảnh 0756 (nhiều biển)",
        "caption": "0756.jpg — Giao lộ nhiều biển báo",
        "path": DATASET_DIR / "images" / "0756.jpg",
    },
    "scene_0544": {
        "label": "Cảnh 0544 (0 biển)",
        "caption": "0544.jpg — Đường phố không có biển",
        "path": DATASET_DIR / "images" / "0544.jpg",
    },
}

PRESET_ROIS = {
    "cam": {
        "label": "Biển Cấm (P.123a)",
        "path": PROJECT_ROOT / "data" / "cropped" / "test" / "Bien_Cam" / "0306_00.jpg",
        "true_class": "Bien_Cam",
    },
    "nguy_hiem": {
        "label": "Biển Nguy Hiểm (W.221b)",
        "path": PROJECT_ROOT / "data" / "cropped" / "test" / "Bien_Nguy_Hiem" / "0357_00.jpg",
        "true_class": "Bien_Nguy_Hiem",
    },
    "hieu_lenh": {
        "label": "Biển Hiệu Lệnh (R.302a)",
        "path": PROJECT_ROOT / "data" / "cropped" / "test" / "Bien_Hieu_Lenh" / "0357_01.jpg",
        "true_class": "Bien_Hieu_Lenh",
    },
    "chi_dan": {
        "label": "Biển Chỉ Dẫn (I.434a)",
        "path": PROJECT_ROOT / "data" / "cropped" / "test" / "Bien_Chi_Dan" / "0307_01.jpg",
        "true_class": "Bien_Chi_Dan",
    },
}

# Preserve original benchmark figures verbatim (audit reported separately)
BENCHMARK_ROI_ROWS = [
    {
        "Mô Hình Phân Loại": "Enhanced LeNet-5 (Đề xuất)",
        "Loại Mô Hình": "Học Sâu (Deep Learning)",
        "Người Phụ Trách": "Thành viên 2",
        "Độ Chính Xác (%)": 98.73,
        "Macro F1 (%)": 95.66,
        "Độ Trễ (ms)": 1.10,
        "Tốc Độ (FPS)": 909.1,
        "Số Tham Số": "0.56M",
        "Dung Lượng (MB)": "2.2 MB",
    },
    {
        "Mô Hình Phân Loại": "Color + HOG + SVM (S3)",
        "Loại Mô Hình": "Học Máy Cổ Điển",
        "Người Phụ Trách": "Thành viên 1",
        "Độ Chính Xác (%)": 98.57,
        "Macro F1 (%)": 95.51,
        "Độ Trễ (ms)": 2.63,
        "Tốc Độ (FPS)": 380.2,
        "Số Tham Số": "457 chiều",
        "Dung Lượng (MB)": "14.8 MB",
    },
    {
        "Mô Hình Phân Loại": "Classical LeNet-5 (Baseline)",
        "Loại Mô Hình": "Học Sâu Cổ Điển",
        "Người Phụ Trách": "Baseline",
        "Độ Chính Xác (%)": 96.25,
        "Macro F1 (%)": 91.80,
        "Độ Trễ (ms)": 0.85,
        "Tốc Độ (FPS)": 1176.5,
        "Số Tham Số": "0.06M",
        "Dung Lượng (MB)": "0.3 MB",
    },
    {
        "Mô Hình Phân Loại": "MobileNetV3-Small",
        "Loại Mô Hình": "Học Sâu Hiện Đại",
        "Người Phụ Trách": "Tham Chiếu",
        "Độ Chính Xác (%)": 98.62,
        "Macro F1 (%)": 95.40,
        "Độ Trễ (ms)": 2.80,
        "Tốc Độ (FPS)": 357.1,
        "Số Tham Số": "1.52M",
        "Dung Lượng (MB)": "6.1 MB",
    },
    {
        "Mô Hình Phân Loại": "ResNet-18",
        "Loại Mô Hình": "Học Sâu Hiện Đại",
        "Người Phụ Trách": "Tham Chiếu",
        "Độ Trễ (ms)": 4.25,
        "Độ Chính Xác (%)": 98.84,
        "Macro F1 (%)": 95.80,
        "Tốc Độ (FPS)": 235.3,
        "Số Tham Số": "11.17M",
        "Dung Lượng (MB)": "44.7 MB",
    },
]

BENCHMARK_SCENE_ROWS = [
    {
        "Pipeline Toàn Trình": "Two-Stage LeNet (YOLO11s + Enhanced LeNet-5)",
        "Tỷ Lệ Định Vị Đúng (IoU >= 0.5)": "85.61%",
        "Đoán Đúng Khi Đã Định Vị": "99.40%",
        "Recall Toàn Trình (E2E)": "85.10%",
        "Độ Trễ Toàn Cảnh": "91.45 ms",
        "Tốc Độ Xử Lý": "10.9 FPS",
    },
    {
        "Pipeline Toàn Trình": "Two-Stage SVM (YOLO11s + Color HOG SVM)",
        "Tỷ Lệ Định Vị Đúng (IoU >= 0.5)": "85.61%",
        "Đoán Đúng Khi Đã Định Vị": "99.17%",
        "Recall Toàn Trình (E2E)": "84.90%",
        "Độ Trễ Toàn Cảnh": "105.20 ms",
        "Tốc Độ Xử Lý": "9.5 FPS",
    },
]
