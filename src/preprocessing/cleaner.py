"""
Module Làm sạch & Gom nhóm Dữ liệu (Dataset Cleaner & Grouping) - Thành viên 1 phụ trách.
Nhiệm vụ:
- Xóa bỏ/loại bỏ thư mục ngoại cảnh "Camera" và biển phụ "B.8a".
- Quét và loại bỏ các file ảnh lỗi, hỏng (corrupt), 0-byte hoặc kích thước quá nhỏ.
- Gom 38 mã biển báo con về 4 NHÓM BIỂN BÁO CHÍNH:
    + Bien_Cam (tiền tố P.*): 2,823 ảnh
    + Bien_Nguy_Hiem (tiền tố W.*): 1,367 ảnh
    + Bien_Hieu_Lenh (tiền tố R.*): 572 ảnh
    + Bien_Chi_Dan (tiền tố I.*): 421 ảnh
- Xuất dữ liệu sạch và được gom nhóm sang 'data/cleaned/'.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import pandas as pd
from tqdm import tqdm

from src.config import (
    RAW_DATA_DIR,
    CLEANED_DATA_DIR,
    IGNORED_CLASSES,
    PREFIX_TO_CLASS,
    TARGET_CLASSES,
)
from src.utils import ensure_dir, imread_unicode


def validate_image_file(image_path: Path, min_size: int = 10) -> Tuple[bool, str]:
    """
    Kiểm tra xem một file ảnh có hợp lệ hay không.
    Trả về (is_valid, reason).
    """
    if not image_path.is_file():
        return False, "Không phải là file"

    if image_path.stat().st_size == 0:
        return False, "File dung lượng 0 bytes"

    # Đọc thử bằng PIL
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        return False, f"Lỗi header PIL: {e}"

    # Đọc thử bằng OpenCV
    try:
        cv_img = imread_unicode(image_path)
        if cv_img is None or cv_img.size == 0:
            return False, "OpenCV không thể decode ảnh"
        h, w = cv_img.shape[:2]
        if h < min_size or w < min_size:
            return False, f"Ảnh quá nhỏ ({w}x{h})"
    except Exception as e:
        return False, f"Lỗi decode OpenCV: {e}"

    return True, "Hợp lệ"


def get_target_class_from_folder(folder_name: str) -> str | None:
    """Xác định nhóm biển báo dựa trên tiền tố chữ cái đầu (P, W, R, I)."""
    prefix = folder_name.split(".")[0].upper()
    return PREFIX_TO_CLASS.get(prefix, None)


def clean_dataset(
    source_dir: Path = RAW_DATA_DIR,
    target_dir: Path = CLEANED_DATA_DIR,
    ignored_classes: List[str] = IGNORED_CLASSES,
    copy_files: bool = True,
) -> pd.DataFrame:
    """
    Làm sạch và gom nhóm toàn bộ tập dữ liệu về 4 nhóm biển báo chính:
    1. Loại bỏ các thư mục trong ignored_classes ('Camera', 'B.8a',...).
    2. Ánh xạ các thư mục con (P.*, W.*, R.*, I.*) vào 4 nhóm lớn.
    3. Quét loại bỏ ảnh corrupt/0-byte.
    4. Copy ảnh hợp lệ sang data/cleaned/<Target_Group>/.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    print("=" * 65)
    print("QUY TRÌNH LÀM SẠCH VÀ GOM NHÓM DỮ LIỆU (DATA CLEAN & GROUP)")
    print(f"[+] Thư mục nguồn: {source_path}")
    print(f"[+] Thư mục đích:   {target_path}")
    print(f"[+] 4 Nhóm đích:   {TARGET_CLASSES}")
    print(f"[+] Các lớp bỏ qua: {ignored_classes}")
    print("=" * 65)

    if not source_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {source_path}")

    # Tạo sẵn các thư mục nhóm đích
    for t_cls in TARGET_CLASSES:
        ensure_dir(target_path / t_cls)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}
    stats = {cls_name: {"Total": 0, "Valid": 0, "Corrupted": 0} for cls_name in TARGET_CLASSES}
    ignored_count = 0

    class_folders = [d for d in source_path.iterdir() if d.is_dir()]

    for folder in class_folders:
        folder_name = folder.name

        # 1. Bỏ qua các thư mục nhiễu (Camera, B.8a)
        if folder_name in ignored_classes:
            files_in_ignored = len(list(folder.glob("*.*")))
            ignored_count += files_in_ignored
            print(f"[!] Bỏ qua thư mục ngoại cảnh: '{folder_name}' ({files_in_ignored} ảnh)")
            continue

        # 2. Xác định nhóm đích (Bien_Cam, Bien_Nguy_Hiem, Bien_Hieu_Lenh, Bien_Chi_Dan)
        target_group = get_target_class_from_folder(folder_name)
        if target_group is None:
            print(f"[!] Bỏ qua thư mục không xác định được nhóm: '{folder_name}'")
            continue

        group_target_dir = target_path / target_group
        all_files = [f for f in folder.iterdir() if f.suffix.lower() in supported_exts]

        for file_path in tqdm(all_files, desc=f"Xử lý {folder_name:10s} -> {target_group}", leave=False):
            stats[target_group]["Total"] += 1
            is_valid, reason = validate_image_file(file_path)

            if is_valid:
                stats[target_group]["Valid"] += 1
                if copy_files:
                    # Đặt tên file mới kèm mã gốc để tránh trùng tên: P.102_0001.png
                    new_file_name = f"{folder_name}_{file_path.name}"
                    dst_path = group_target_dir / new_file_name
                    if not dst_path.exists():
                        shutil.copy2(file_path, dst_path)
            else:
                stats[target_group]["Corrupted"] += 1
                print(f"\n[-] Ảnh hỏng: {folder_name}/{file_path.name} ({reason})")

    # Tổng kết dạng bảng
    summary_data = []
    for cls_name, vals in stats.items():
        summary_data.append({
            "Nhóm Biển Báo": cls_name,
            "Tổng quét": vals["Total"],
            "Hợp lệ (Cleaned)": vals["Valid"],
            "Lỗi (Corrupted)": vals["Corrupted"],
        })

    df_stats = pd.DataFrame(summary_data)
    print("\n" + "=" * 65)
    print("BÁO CÁO TỔNG HỢP SAU KHI LÀM SẠCH VÀ GOM NHÓM:")
    print("=" * 65)
    print(df_stats.to_string(index=False))
    print(f"\n[+] Tổng số ảnh hợp lệ đưa vào huấn luyện: {df_stats['Hợp lệ (Cleaned)'].sum()} ảnh")
    print(f"[+] Tổng số ảnh nhiễu đã loại bỏ: {df_stats['Lỗi (Corrupted)'].sum() + ignored_count} ảnh")
    print("=" * 65)

    return df_stats


if __name__ == "__main__":
    clean_dataset()
