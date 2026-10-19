"""
Pipeline Tổng thể Tích hợp - Khung sườn cho Thành viên 3 phụ trách.
Luồng xử lý hoàn chỉnh 4 Giai đoạn:
Ảnh ➔ Lọc màu HSV ➔ Tìm ROI ➔ Trích xuất HOG ➔ Dự đoán SVM ➔ Hiển thị nhãn
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np

from src.config import MODEL_PATH, LABEL_ENCODER_PATH


class TrafficSignPipeline:
    """Hệ thống tích hợp toàn bộ các giai đoạn nhận diện biển báo."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        encoder_path: Path = LABEL_ENCODER_PATH,
    ):
        self.model_path = model_path
        self.encoder_path = encoder_path

    def predict(self, bgr_image: np.ndarray) -> Dict[str, Any]:
        """
        Nhận ảnh đầu vào BGR và trả về kết quả phát hiện, phân loại.
        """
        # TODO: [Thành viên 3] Lắp ráp code của TV1 và model của TV2 thành pipeline tổng thể
        raise NotImplementedError("TODO: Thành viên 3 phụ trách cài đặt pipeline tổng thể")
