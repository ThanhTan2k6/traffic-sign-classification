"""
Giai đoạn 1: Tiền xử lý & Lọc màu (Color Filter & Masking) - Thành viên 1 phụ trách.
Quy trình:
1. Chuyển không gian màu từ BGR/RGB sang HSV.
2. Tạo mặt nạ nhị phân lọc các dải màu đặc trưng của biển báo Việt Nam:
   - Màu Đỏ (Red): Biển cấm (viền đỏ), Biển nguy hiểm (viền đỏ)
   - Màu Xanh dương (Blue): Biển hiệu lệnh, Biển chỉ dẫn
   - Màu Vàng (Yellow): Nền biển cảnh báo nguy hiểm
3. Hợp nhất các mặt nạ để thu được vùng khả nghi chứa biển báo.
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import HSV_COLOR_RANGES


class ColorFilter:
    """Lớp xử lý chuyển đổi không gian màu và tạo mặt nạ màu biển báo."""

    def __init__(self, color_ranges: Optional[Dict] = None):
        self.color_ranges = color_ranges if color_ranges is not None else HSV_COLOR_RANGES

    def to_hsv(self, bgr_image: np.ndarray) -> np.ndarray:
        """Chuyển đổi ảnh BGR sang HSV với tiền xử lý làm mờ nhẹ để giảm nhiễu hạt."""
        # Áp dụng GaussianBlur nhẹ trước khi đổi màu để giảm nhiễu hạt muối tiêu
        blurred = cv2.GaussianBlur(bgr_image, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv

    def get_color_mask(self, hsv_image: np.ndarray, color_name: str) -> np.ndarray:
        """
        Tạo mặt nạ nhị phân cho một màu cụ thể ('red', 'blue', 'yellow').
        Riêng màu đỏ có 2 dải do giá trị Hue nằm ở 2 đầu 0-10 và 170-180.
        """
        color_name = color_name.lower()
        if color_name not in self.color_ranges:
            raise ValueError(f"Không hỗ trợ màu '{color_name}'. Chỉ hỗ trợ: {list(self.color_ranges.keys())}")

        ranges = self.color_ranges[color_name]
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        for r in ranges:
            lower = r["lower"]
            upper = r["upper"]
            mask = cv2.inRange(hsv_image, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        return combined_mask

    def create_traffic_sign_mask(
        self,
        bgr_image: np.ndarray,
        colors: Tuple[str, ...] = ("red", "blue", "yellow")
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Tạo mặt nạ tổng hợp cho tất cả các màu đặc trưng của biển báo.
        
        Trả về:
        - final_mask: Mặt nạ nhị phân gộp chung (Đỏ OR Xanh OR Vàng)
        - masked_image: Ảnh gốc được áp mặt nạ (chỉ giữ lại vùng có màu biển báo)
        - individual_masks: Dict chứa từng mặt nạ màu riêng biệt để phân tích
        """
        hsv = self.to_hsv(bgr_image)
        final_mask = np.zeros(bgr_image.shape[:2], dtype=np.uint8)
        individual_masks = {}

        for color in colors:
            mask = self.get_color_mask(hsv, color)
            individual_masks[color] = mask
            final_mask = cv2.bitwise_or(final_mask, mask)

        masked_image = cv2.bitwise_and(bgr_image, bgr_image, mask=final_mask)
        return final_mask, masked_image, individual_masks
