"""
Giai đoạn 4: Huấn luyện & Phân loại Biển báo (SVM / k-NN) - Khung sườn cho Thành viên 2 phụ trách.
Nhiệm vụ:
- Đọc dữ liệu ROI từ 'data/rois/'.
- Trích xuất đặc trưng HOG cho từng ảnh.
- Chia tập Train/Test theo tỷ lệ 80/20.
- Huấn luyện mô hình SVM / k-NN bằng scikit-learn.
- Lưu mô hình sang file .pkl để Thành viên 3 tích hợp.
"""

from pathlib import Path
from src.config import ROI_OUTPUT_DIR, MODEL_PATH, LABEL_ENCODER_PATH


def load_dataset_and_extract_hog(data_dir: Path):
    """Đọc tất cả ảnh ROI và trích xuất đặc trưng HOG."""
    # TODO: [Thành viên 2] Đọc ROI và trích xuất vector đặc trưng HOG
    raise NotImplementedError("TODO: Thành viên 2 phụ trách cài đặt load_dataset_and_extract_hog")


def train_svm_pipeline(
    data_dir: Path = ROI_OUTPUT_DIR,
    model_save_path: Path = MODEL_PATH,
    encoder_save_path: Path = LABEL_ENCODER_PATH,
    use_knn: bool = False,
):
    """Quy trình huấn luyện mô hình SVM / k-NN."""
    # TODO: [Thành viên 2] Huấn luyện và xuất model.pkl
    raise NotImplementedError("TODO: Thành viên 2 phụ trách cài đặt train_svm_pipeline")


if __name__ == "__main__":
    train_svm_pipeline()
