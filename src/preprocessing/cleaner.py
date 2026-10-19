"""
Module Làm sạch Dữ liệu (Dataset Cleaner) - Khung sườn cho Thành viên 1.
Nhiệm vụ:
- Xóa bỏ/loại bỏ thư mục "Camera" (ảnh ngoại cảnh/nhiễu).
- Quét và loại bỏ các file ảnh lỗi, hỏng (corrupt), 0-byte hoặc kích thước quá nhỏ.
- Xuất dữ liệu sạch sang thư mục đích hoặc tạo báo cáo thống kê.
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd

from src.config import RAW_DATA_DIR, CLEANED_DATA_DIR, IGNORED_CLASSES


def validate_image_file(image_path: Path, min_size: int = 10) -> Tuple[bool, str]:
    """
    Kiểm tra xem một file ảnh có hợp lệ hay không.
    Trả về (is_valid, reason).
    """
    # TODO: [Thành viên 1] Cài đặt hàm kiểm tra tính hợp lệ của file ảnh
    raise NotImplementedError("TODO: Thành viên 1 cài đặt validate_image_file")


def clean_dataset(
    source_dir: Path = RAW_DATA_DIR,
    target_dir: Path = CLEANED_DATA_DIR,
    ignored_classes: List[str] = IGNORED_CLASSES,
    copy_files: bool = True,
) -> pd.DataFrame:
    """
    Làm sạch toàn bộ tập dữ liệu:
    1. Bỏ qua các thư mục trong danh sách ignored_classes (ví dụ: 'Camera').
    2. Quét kiểm tra tất cả các file ảnh trong từng thư mục lớp.
    3. Copy ảnh hợp lệ sang target_dir (nếu copy_files=True).
    4. Trả về DataFrame thống kê kết quả.
    """
    # TODO: [Thành viên 1] Cài đặt hàm quét làm sạch dataset và loại bỏ folder Camera
    raise NotImplementedError("TODO: Thành viên 1 cài đặt clean_dataset")


if __name__ == "__main__":
    clean_dataset()
