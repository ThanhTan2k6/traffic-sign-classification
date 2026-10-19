"""
Script tạo tập dữ liệu ROI chuẩn để bàn giao cho Thành viên 2 - Khung sườn cho Thành viên 1.
Nhiệm vụ:
- Duyệt qua từng thư mục lớp biển báo.
- Áp dụng Giai đoạn 1 (Lọc màu HSV) + Giai đoạn 2 (Tìm đường viền & Cắt ROI).
- Chuẩn hóa kích thước ROI về 64x64.
- Lưu kết quả vào thư mục 'data/rois/<class_name>/'.
"""

from pathlib import Path
from src.config import AUGMENTED_DATA_DIR, ROI_OUTPUT_DIR, ROI_SIZE


def process_and_export_rois(
    source_dir: Path = AUGMENTED_DATA_DIR,
    output_dir: Path = ROI_OUTPUT_DIR,
    target_size: tuple = ROI_SIZE,
):
    """
    Xử lý hàng loạt ảnh: phát hiện biển báo, cắt ROI chuẩn hóa và xuất ra thư mục đích.
    """
    # TODO: [Thành viên 1] Cài đặt trích xuất hàng loạt ROI và lưu vào output_dir
    raise NotImplementedError("TODO: Thành viên 1 cài đặt process_and_export_rois")


if __name__ == "__main__":
    process_and_export_rois()
