"""
Giai đoạn 3: Trích xuất đặc trưng HOG (Histogram of Oriented Gradients) - Thành viên 2 phụ trách.
Nhiệm vụ:
- Chuẩn hóa kích thước vùng ROI (mặc định 64x64).
- Chuyển sang ảnh xám (Grayscale).
- Tính toán vector đặc trưng hình dáng/cấu trúc HOG.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
from skimage.feature import hog

from src.config import HOG_CONFIG, ROI_SIZE


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
        
        Tham số:
        - bgr_roi: Ảnh ROI (định dạng BGR hoặc Gray).
        - visualize: Nếu True, trả về cả ảnh trực quan hóa HOG (dùng cho báo cáo/demo).

        Trả về:
        - features: 1D numpy array chứa các đặc trưng HOG.
        - hog_image: Ảnh biểu diễn HOG (nếu visualize=True), ngược lại là None.
        """
        # Đảm bảo ảnh có kích thước chuẩn 64x64
        if bgr_roi.shape[:2] != (ROI_SIZE[1], ROI_SIZE[0]):
            bgr_roi = cv2.resize(bgr_roi, ROI_SIZE, interpolation=cv2.INTER_AREA)

        # Chuyển sang ảnh xám
        if len(bgr_roi.shape) == 3:
            gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_roi

        if visualize:
            features, hog_image = hog(
                gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=True,
                transform_sqrt=self.transform_sqrt,
            )
            return features, hog_image
        else:
            features = hog(
                gray,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=False,
                transform_sqrt=self.transform_sqrt,
            )
            return features, None
