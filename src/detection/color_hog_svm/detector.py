"""
Phương pháp 1 (Thành viên 1 phụ trách): Mô hình Truyền thống (HSV Filter + Morphology + HOG + SVM).
Kế thừa BaseDetector để tương thích với hệ thống Benchmark và App.
"""

from typing import Dict, Any, Optional
import numpy as np
from src.detection.base_detector import BaseDetector


class ColorHogSvmDetector(BaseDetector):
    """Lớp suy luận cho Phương pháp 1: HOG + SVM (Thành viên 1)."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(model_name="color_hog_svm", checkpoint_path=checkpoint_path)
        self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: Optional[str] = None):
        """TODO: [Thành viên 1] Nạp trọng số mô hình từ tệp checkpoint."""
        pass

    def preprocess(self, image: np.ndarray) -> Any:
        """TODO: [Thành viên 1] Tiền xử lý ảnh (Lọc màu HSV, cắt ROI hoặc trích xuất HOG)."""
        raise NotImplementedError("TODO: Thành viên 1 cài đặt preprocess cho ColorHogSvm")

    def inference(self, preprocessed_input: Any) -> Any:
        """TODO: [Thành viên 1] Thực thi suy luận mô hình SVM."""
        raise NotImplementedError("TODO: Thành viên 1 cài đặt inference cho ColorHogSvm")

    def postprocess(self, raw_output: Any) -> Dict[str, Any]:
        """
        TODO: [Thành viên 1] Trả về dict chuẩn theo quy định:
        {"class_name": str, "class_id": int, "confidence": float, "bbox": None}
        """
        raise NotImplementedError("TODO: Thành viên 1 cài đặt postprocess cho ColorHogSvm")
