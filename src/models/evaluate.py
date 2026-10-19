"""
Module Đánh giá Mô hình - Khung sườn cho Thành viên 2 & 3.
Nhiệm vụ:
- Tính toán Accuracy, Precision, Recall, F1-Score.
- Trực quan hóa ma trận nhầm lẫn (Confusion Matrix).
"""

from pathlib import Path
from src.config import MODEL_PATH, LABEL_ENCODER_PATH, ROI_OUTPUT_DIR


def evaluate_model(
    model_path: Path = MODEL_PATH,
    encoder_path: Path = LABEL_ENCODER_PATH,
    data_dir: Path = ROI_OUTPUT_DIR,
    save_fig_path: Path = None,
):
    """Đánh giá chi tiết mô hình đã huấn luyện và vẽ ma trận nhầm lẫn."""
    # TODO: [Thành viên 2] Đo đạc các chỉ số đánh giá và vẽ đồ thị Confusion Matrix
    raise NotImplementedError("TODO: Thành viên 2 & 3 phụ trách cài đặt evaluate_model")


if __name__ == "__main__":
    evaluate_model()
