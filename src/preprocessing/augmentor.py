"""
Module Tăng cường Dữ liệu (Data Augmentation) - Khung sườn cho Thành viên 1.
Nhiệm vụ:
- Khắc phục hiện tượng mất cân bằng dữ liệu giữa các lớp biển báo (Class Imbalance).
- Áp dụng các biến đổi hình học và màu sắc phù hợp với biển báo giao thông:
  + Xoay góc nhỏ (-15 đến +15 độ)
  + Điều chỉnh độ sáng và độ tương phản
  + Làm mờ Gaussian (motion blur)
  + Dịch chuyển nhẹ (Translation)
"""

from pathlib import Path
from typing import Optional
import numpy as np

from src.config import CLEANED_DATA_DIR, AUGMENTED_DATA_DIR


class DataAugmentor:
    """Class cung cấp các phương thức biến đổi và tăng cường ảnh biển báo."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def apply_random_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Áp dụng chuỗi các phép biến đổi ngẫu nhiên lên 1 ảnh."""
        # TODO: [Thành viên 1] Triển khai các phép xoay, đổi sáng, làm mờ
        raise NotImplementedError("TODO: Thành viên 1 cài đặt apply_random_pipeline")

    def balance_dataset(
        self,
        source_dir: Path = CLEANED_DATA_DIR,
        target_dir: Path = AUGMENTED_DATA_DIR,
        target_samples_per_class: Optional[int] = None,
    ):
        """
        Tự động tăng cường dữ liệu cho các lớp ít ảnh để đạt target_samples_per_class.
        """
        # TODO: [Thành viên 1] Triển khai thuật toán cân bằng dữ liệu các lớp
        raise NotImplementedError("TODO: Thành viên 1 cài đặt balance_dataset")


if __name__ == "__main__":
    augmentor = DataAugmentor()
    augmentor.balance_dataset()
