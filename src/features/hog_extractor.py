"""
Giai đoạn 3: Trích xuất đặc trưng HOG - Khung sườn cho Thành viên 2.
Nhiệm vụ:
- Chuẩn hóa kích thước vùng ROI (mặc định 64x64).
- Chuyển sang ảnh xám (Grayscale).
- Tính toán vector đặc trưng hình dáng/cấu trúc HOG.
"""

from typing import Tuple, Optional
import numpy as np
from src.config import HOG_CONFIG


class HOGFeatureExtractor:
    """Lớp trích xuất vector đặc trưng HOG từ ảnh biển báo."""

    def __init__(
        self,
        orientations: int = HOG_CONFIG["orientations"],
        pixels_per_cell: Tuple[int, int] = HOG_CONFIG["pixels_per_cell"],
        cells_per_block: Tuple[int, int] = HOG_CONFIG["cells_per_block"],
        block_norm: str = HOG_CONFIG["block_norm"],
        transform_sqrt: bool = HOG_CONFIG["transform_sqrt"],
    ):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.transform_sqrt = transform_sqrt

    def extract(
        self,
        bgr_roi: np.ndarray,
        visualize: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Trích xuất vector đặc trưng HOG từ ảnh ROI.
        """
        # TODO: [Thành viên 2] Cài đặt trích xuất đặc trưng HOG
        raise NotImplementedError("TODO: Thành viên 2 cài đặt extract HOG")
