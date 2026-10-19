"""
Giai đoạn 1: Tiền xử lý & Lọc màu (Color Filter & Masking) - Khung sườn cho Thành viên 1.
Quy trình:
1. Chuyển không gian màu từ BGR/RGB sang HSV.
2. Tạo mặt nạ nhị phân lọc các dải màu đặc trưng của biển báo Việt Nam:
   - Màu Đỏ (Red): Biển cấm (viền đỏ), Biển nguy hiểm (viền đỏ)
   - Màu Xanh dương (Blue): Biển hiệu lệnh, Biển chỉ dẫn
   - Màu Vàng (Yellow): Nền biển cảnh báo nguy hiểm
3. Hợp nhất các mặt nạ để thu được vùng khả nghi chứa biển báo.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from src.config import HSV_COLOR_RANGES


class ColorFilter:
    """Lớp xử lý chuyển đổi không gian màu và tạo mặt nạ màu biển báo."""

    def __init__(self, color_ranges: Optional[Dict] = None):
        self.color_ranges = color_ranges if color_ranges is not None else HSV_COLOR_RANGES

    def to_hsv(self, bgr_image: np.ndarray) -> np.ndarray:
        """Chuyển đổi ảnh BGR sang HSV."""
        # TODO: [Thành viên 1] Cài đặt chuyển đổi không gian màu BGR sang HSV
        raise NotImplementedError("TODO: Thành viên 1 cài đặt to_hsv")

    def get_color_mask(self, hsv_image: np.ndarray, color_name: str) -> np.ndarray:
        """Tạo mặt nạ nhị phân cho một màu cụ thể ('red', 'blue', 'yellow')."""
        # TODO: [Thành viên 1] Cài đặt tạo mask cho từng dải màu
        raise NotImplementedError("TODO: Thành viên 1 cài đặt get_color_mask")

    def create_traffic_sign_mask(
        self,
        bgr_image: np.ndarray,
        colors: Tuple[str, ...] = ("red", "blue", "yellow")
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Tạo mặt nạ tổng hợp cho tất cả các màu đặc trưng của biển báo.
        
        Trả về:
        - final_mask: Mặt nạ nhị phân gộp chung
        - masked_image: Ảnh gốc được áp mặt nạ
        - individual_masks: Dict chứa từng mặt nạ màu riêng biệt
        """
        # TODO: [Thành viên 1] Cài đặt kết hợp các mặt nạ màu
        raise NotImplementedError("TODO: Thành viên 1 cài đặt create_traffic_sign_mask")
