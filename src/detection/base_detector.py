"""
Lớp cơ sở trừu tượng (BaseDetector) - Chuẩn hóa Interface cho tất cả các mô hình.
Tất cả các thành viên BẮT BUỘC phải kế thừa class này để đảm bảo:
- Thống nhất hàm gọi suy luận (Inference Interface)
- Định dạng kết quả trả về chuẩn hóa cho hệ thống đánh giá (Benchmark Module) và App
"""

from abc import ABC, abstractmethod
import time
from typing import Dict, Any, Optional
import numpy as np


class BaseDetector(ABC):
    """Lớp trừu tượng định nghĩa Interface chuẩn cho mọi mô hình nhận dạng biển báo."""

    def __init__(self, model_name: str, checkpoint_path: Optional[str] = None):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.model = None

    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Nạp trọng số mô hình từ tệp checkpoint."""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """Tiền xử lý ảnh đầu vào (BGR numpy array) trước khi đưa vào mô hình."""
        pass

    @abstractmethod
    def inference(self, preprocessed_input: Any) -> Any:
        """Thực thi dự đoán của mô hình."""
        pass

    @abstractmethod
    def postprocess(self, raw_output: Any) -> Dict[str, Any]:
        """
        Hậu xử lý kết quả dự đoán.
        Bắt buộc trả về dict chuẩn:
        {
            "class_name": str,       # Tên nhãn lớp dự đoán (VD: 'Bien_Cam')
            "class_id": int,         # ID nhãn lớp (0, 1, 2, 3)
            "confidence": float,     # Độ tin cậy [0.0, 1.0]
            "bbox": tuple | None,    # (x1, y1, x2, y2) nếu có khoanh vùng
        }
        """
        pass

    def predict(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Luồng dự đoán chuẩn mực kết hợp đo thời gian suy luận (Inference Time):
        preprocess -> inference -> postprocess
        """
        start_time = time.perf_counter()

        preprocessed = self.preprocess(image, **kwargs)
        raw_output = self.inference(preprocessed)
        result = self.postprocess(raw_output)

        latency = time.perf_counter() - start_time
        result["inference_time"] = latency
        result["fps"] = (1.0 / latency) if latency > 0 else 0.0
        result["model_name"] = self.model_name

        return result
