"""
Phương pháp 2 (Thành viên 2 phụ trách): Template mô hình (Ví dụ: Custom CNN / LeNet / ResNet).
Kế thừa BaseDetector để tương thích với hệ thống Benchmark và App.
"""

from typing import Dict, Any, Optional
import numpy as np
from src.detection.base_detector import BaseDetector


class Model2Detector(BaseDetector):
    """Lớp suy luận cho Phương pháp 2 (Thành viên 2)."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(model_name="model_2", checkpoint_path=checkpoint_path)
        self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: Optional[str] = None):
        """TODO: [Thành viên 2] Nạp trọng số mô hình."""
        pass

    def preprocess(self, image: np.ndarray) -> Any:
        """TODO: [Thành viên 2] Tiền xử lý ảnh (resize, normalize, tensor convert...)."""
        raise NotImplementedError("TODO: Thành viên 2 cài đặt preprocess cho Model 2")

    def inference(self, preprocessed_input: Any) -> Any:
        """TODO: [Thành viên 2] Thực thi suy luận mô hình."""
        raise NotImplementedError("TODO: Thành viên 2 cài đặt inference cho Model 2")

    def postprocess(self, raw_output: Any) -> Dict[str, Any]:
        """
        TODO: [Thành viên 2] Trả về dict chuẩn:
        {"class_name": str, "class_id": int, "confidence": float, "bbox": None}
        """
        raise NotImplementedError("TODO: Thành viên 2 cài đặt postprocess cho Model 2")
