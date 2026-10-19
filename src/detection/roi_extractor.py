"""
Giai đoạn 2: Phân đoạn và Tìm vùng quan tâm (ROI - Region of Interest) - Thành viên 1 phụ trách.
Quy trình:
1. Áp dụng các phép biến đổi hình thái học (Morphology: Opening, Closing) để khử nhiễu và nối liền vùng.
2. Tìm kiếm đường viền (Contours Analysis bằng cv2.findContours).
3. Lọc contours dựa trên điều kiện hình học:
   - Diện tích (Area): loại bỏ các đốm nhiễu nhỏ hoặc các mảng nền quá lớn.
   - Tỷ lệ khung hình (Aspect Ratio = w / h): biển báo thường gần vuông/tròn/tam giác đều (0.5 - 2.0).
4. Cắt vùng quan tâm (ROI) và chuẩn hóa kích thước (resize về 64x64).
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
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
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
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
        """
        Khử nhiễu bằng bộ lọc hình thái học (Morphological Operations):
        - MORPH_OPEN: Xóa các hạt nhiễu trắng nhỏ lẻ bên ngoài
        - MORPH_CLOSE: Lấp đầy các lỗ trống và nối liền các vùng viền đứt đoạn
        """
        # 1. Opening: Xóa nhiễu ngoài
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel, iterations=open_iters)
        # 2. Closing: Lấp đầy khoảng trống bên trong biển báo
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel, iterations=close_iters)
        return closed

    def extract_rois(
        self,
        bgr_image: np.ndarray,
        binary_mask: np.ndarray,
        padding_ratio: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Phân tích đường viền từ mặt nạ và trích xuất các vùng ROI đạt tiêu chuẩn.

        Trả về danh sách các dict, mỗi dict chứa:
        - 'bbox': (x, y, w, h) tọa độ hộp bao trên ảnh gốc
        - 'roi_original': vùng ảnh cắt gốc (chưa resize)
        - 'roi_normalized': vùng ảnh sau khi chuẩn hóa về kích thước 64x64
        - 'contour': đường viền tìm được
        - 'area': diện tích contour
        - 'aspect_ratio': tỷ lệ w / h
        """
        # 1. Khử nhiễu hình thái học
        cleaned_mask = self.clean_mask_morphology(binary_mask)

        # 2. Tìm contours bên ngoài (RETR_EXTERNAL)
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected_rois = []
        img_h, img_w = bgr_image.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Lọc theo diện tích
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h)

            # Lọc theo tỷ lệ khung hình (Aspect Ratio)
            if not (self.min_ar <= aspect_ratio <= self.max_ar):
                continue

            # Mở rộng bounding box nhẹ một chút (padding) để không bị cắt lẹm viền
            pad_x = int(w * padding_ratio)
            pad_y = int(h * padding_ratio)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_w, x + w + pad_x)
            y2 = min(img_h, y + h + pad_y)

            # Cắt ROI từ ảnh gốc BGR
            roi_original = bgr_image[y1:y2, x1:x2]
            if roi_original.size == 0:
                continue

            # Chuẩn hóa kích thước (ví dụ 64x64) cho Thành viên 2 trích xuất HOG
            roi_normalized = cv2.resize(roi_original, self.output_size, interpolation=cv2.INTER_AREA)

            detected_rois.append({
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "roi_original": roi_original,
                "roi_normalized": roi_normalized,
                "contour": cnt,
                "area": area,
                "aspect_ratio": aspect_ratio,
            })

        return detected_rois

    @staticmethod
    def draw_detections(
        bgr_image: np.ndarray,
        detected_rois: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Vẽ bounding box của các biển báo tìm được lên bản sao ảnh gốc."""
        annotated = bgr_image.copy()
        for idx, item in enumerate(detected_rois):
            x, y, w, h = item["bbox"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(
                annotated,
                f"ROI #{idx + 1}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return annotated
