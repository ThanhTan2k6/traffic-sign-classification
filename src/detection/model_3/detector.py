"""
Phương pháp 3 (Thành viên 3 phụ trách): Template mô hình (Ví dụ: YOLO / MobileNet / k-NN).
Kế thừa BaseDetector để tương thích với hệ thống Benchmark và App.
"""

from typing import Dict, Any, Optional
import numpy as np
from src.detection.base_detector import BaseDetector


class Model3Detector(BaseDetector):
    """Lớp suy luận cho Phương pháp 3 (Thành viên 3)."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(model_name="model_3", checkpoint_path=checkpoint_path)
        self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: Optional[str] = None):
        """TODO: [Thành viên 3] Nạp trọng số mô hình."""
        pass

    def preprocess(self, image: np.ndarray) -> Any:
        """TODO: [Thành viên 3] Tiền xử lý ảnh đầu vào."""
        raise NotImplementedError("TODO: Thành viên 3 cài đặt preprocess cho Model 3")

    def inference(self, preprocessed_input: Any) -> Any:
        """TODO: [Thành viên 3] Thực thi suy luận mô hình."""
        raise NotImplementedError("TODO: Thành viên 3 cài đặt inference cho Model 3")

    def postprocess(self, raw_output: Any) -> Dict[str, Any]:
        """
        TODO: [Thành viên 3] Trả về dict chuẩn:
        {"class_name": str, "class_id": int, "confidence": float, "bbox": None}
        """
        raise NotImplementedError("TODO: Thành viên 3 cài đặt postprocess cho Model 3")
