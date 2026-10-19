"""
Giai đoạn 2: Phân đoạn và Tìm vùng quan tâm (ROI - Region of Interest) - Khung sườn cho Thành viên 1.
Quy trình:
1. Áp dụng các phép biến đổi hình thái học (Morphology: Opening, Closing) để khử nhiễu và nối liền vùng.
2. Tìm kiếm đường viền (Contours Analysis bằng cv2.findContours).
3. Lọc contours dựa trên điều kiện hình học:
   - Diện tích (Area)
   - Tỷ lệ khung hình (Aspect Ratio = w / h)
4. Cắt vùng quan tâm (ROI) và chuẩn hóa kích thước (resize về 64x64).
"""

from typing import List, Tuple, Dict, Any
import numpy as np

from src.config import (
    ROI_SIZE,
    MORPH_KERNEL_SIZE,
    MORPH_OPEN_ITERS,
    MORPH_CLOSE_ITERS,
    MIN_CONTOUR_AREA,
    MAX_CONTOUR_AREA,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
)


class ROIExtractor:
    """Lớp xử lý khử nhiễu hình thái học, tìm đường viền và cắt ảnh ROI biển báo."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = MORPH_KERNEL_SIZE,
        min_area: float = MIN_CONTOUR_AREA,
        max_area: float = MAX_CONTOUR_AREA,
        aspect_ratio_range: Tuple[float, float] = (MIN_ASPECT_RATIO, MAX_ASPECT_RATIO),
        output_size: Tuple[int, int] = ROI_SIZE,
    ):
        self.kernel_size = kernel_size
        self.min_area = min_area
        self.max_area = max_area
        self.min_ar, self.max_ar = aspect_ratio_range
        self.output_size = output_size

    def clean_mask_morphology(
        self,
        binary_mask: np.ndarray,
        open_iters: int = MORPH_OPEN_ITERS,
        close_iters: int = MORPH_CLOSE_ITERS,
    ) -> np.ndarray:
        """Khử nhiễu bằng bộ lọc hình thái học (Morphological Operations)."""
        # TODO: [Thành viên 1] Cài đặt phép Opening và Closing
        raise NotImplementedError("TODO: Thành viên 1 cài đặt clean_mask_morphology")

    def extract_rois(
        self,
        bgr_image: np.ndarray,
        binary_mask: np.ndarray,
        padding_ratio: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Phân tích đường viền từ mặt nạ và trích xuất các vùng ROI đạt tiêu chuẩn."""
        # TODO: [Thành viên 1] Cài đặt tìm contours, lọc bbox và cắt ROI
        raise NotImplementedError("TODO: Thành viên 1 cài đặt extract_rois")

    @staticmethod
    def draw_detections(
        bgr_image: np.ndarray,
        detected_rois: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Vẽ bounding box của các biển báo tìm được lên ảnh gốc."""
        # TODO: [Thành viên 1] Cài đặt vẽ bounding box
        raise NotImplementedError("TODO: Thành viên 1 cài đặt draw_detections")
