"""
Module Làm sạch Dữ liệu (Dataset Cleaner) - Thành viên 1 phụ trách.
Nhiệm vụ:
- Xóa bỏ/loại bỏ thư mục "Camera" (ảnh ngoại cảnh/nhiễu).
- Quét và loại bỏ các file ảnh lỗi, hỏng (corrupt), 0-byte hoặc kích thước quá nhỏ.
- Xuất dữ liệu sạch sang thư mục đích hoặc tạo báo cáo thống kê.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import pandas as pd
from tqdm import tqdm

from src.config import RAW_DATA_DIR, CLEANED_DATA_DIR, IGNORED_CLASSES
from src.utils import ensure_dir, imread_unicode


def validate_image_file(image_path: Path, min_size: int = 10) -> Tuple[bool, str]:
    """
    Kiểm tra xem một file ảnh có hợp lệ hay không.
    Trả về (is_valid, reason).
    """
    if not image_path.is_file():
        return False, "Không phải là file"

    # Kiểm tra kích thước file
    if image_path.stat().st_size == 0:
        return False, "File dung lượng 0 bytes"

    # Đọc thử bằng PIL
    try:
        with Image.open(image_path) as img:
            img.verify()  # Kiểm tra tính toàn vẹn của header
    except Exception as e:
        return False, f"Lỗi đọc header PIL: {e}"

    # Đọc thử bằng OpenCV
    try:
        cv_img = imread_unicode(image_path)
        if cv_img is None or cv_img.size == 0:
            return False, "OpenCV không thể giải mã ảnh"
        h, w = cv_img.shape[:2]
        if h < min_size or w < min_size:
            return False, f"Ảnh quá nhỏ ({w}x{h} < {min_size}x{min_size})"
    except Exception as e:
        return False, f"Lỗi decode OpenCV: {e}"

    return True, "Hợp lệ"


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
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    print("=" * 60)
    print("BẮT ĐẦU QUY TRÌNH LÀM SẠCH DỮ LIỆU (DATA CLEANING)")
    print(f"[+] Thư mục nguồn: {source_path}")
    print(f"[+] Thư mục đích:   {target_path}")
    print(f"[+] Các lớp loại bỏ: {ignored_classes}")
    print("=" * 60)

    if not source_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {source_path}")

    ensure_dir(target_path)

    stats = []
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}

    # Lấy danh sách thư mục con (các lớp biển báo)
    class_folders = [d for d in source_path.iterdir() if d.is_dir()]

    for folder in class_folders:
        class_name = folder.name

        # 1. Kiểm tra nếu nằm trong danh sách cần loại bỏ (ví dụ 'Camera')
        if class_name in ignored_classes:
            print(f"[!] Bỏ qua thư mục ngoại cảnh: '{class_name}'")
            continue

        class_target_dir = target_path / class_name
        if copy_files:
            ensure_dir(class_target_dir)

        # Lấy danh sách ảnh trong lớp
        all_files = [f for f in folder.iterdir() if f.suffix.lower() in supported_exts]
        valid_count = 0
        corrupt_count = 0

        for file_path in tqdm(all_files, desc=f"Đang lọc lớp {class_name:10s}", leave=False):
            is_valid, reason = validate_image_file(file_path)
            if is_valid:
                valid_count += 1
                if copy_files:
                    target_file = class_target_dir / file_path.name
                    if not target_file.exists():
                        shutil.copy2(file_path, target_file)
            else:
                corrupt_count += 1
                print(f"\n[-] Phát hiện ảnh lỗi: {file_path.name} ({reason})")

        stats.append({
            "Class": class_name,
            "Total Files": len(all_files),
            "Valid Files": valid_count,
            "Corrupted/Removed": corrupt_count,
        })

    df_stats = pd.DataFrame(stats)
    print("\n" + "=" * 60)
    print("KẾT QUẢ LÀM SẠCH TẬP DỮ LIỆU:")
    print("=" * 60)
    if not df_stats.empty:
        print(df_stats.to_string(index=False))
        total_valid = df_stats["Valid Files"].sum()
        total_corrupt = df_stats["Corrupted/Removed"].sum()
        print(f"\n[+] Tổng số ảnh hợp lệ: {total_valid}")
        print(f"[+] Tổng số ảnh lỗi bị loại: {total_corrupt}")
        print(f"[+] Tổng số lớp biển báo hợp lệ: {len(df_stats)}")
    else:
        print("[-] Không có dữ liệu nào được xử lý.")

    return df_stats


if __name__ == "__main__":
    clean_dataset()
