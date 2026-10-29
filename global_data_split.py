"""
================================================================================
🌍 GLOBAL DATA SPLITTER CHO HỆ THỐNG PHÂN LOẠI BIỂN BÁO GIAO THÔNG
================================================================================
Đọc toàn bộ ảnh từ dataset gốc, lọc bỏ lớp nhiễu (Camera, B.8a), ánh xạ về 4 nhóm chính.
Chia cố định thành 3 tập độc lập (Global Split):
    - Tập Train : 70% (Dùng để huấn luyện mô hình)
    - Tập Val   : 15% (Dùng để tinh chỉnh siêu tham số)
    - Tập Test  : 15% (Dùng để đánh giá độc lập / Benchmarking cuối cùng)

TẤT CẢ các mô hình (color_hog_svm, model_2, model_3) ĐỀU PHẢI dùng chung danh sách này
để đảm bảo tính công bằng 100% và chống Data Leakage tuyệt đối.
"""

from collections import Counter
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_DIR,
    GLOBAL_SPLITS_DIR,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_FILE,
    TEST_SPLIT_FILE,
    PREFIX_TO_CLASS,
    IGNORED_CLASSES,
    SPLIT_RATIOS,
    RANDOM_SEED,
)
from src.utils import ensure_dir


def main():
    print("=" * 65)
    print("🌍 KHỞI TẠO GLOBAL DATA SPLIT (3 TẬP ĐỘC LẬP: TRAIN / VAL / TEST)")
    print("=" * 65)

    if not RAW_DATA_DIR.exists():
        print(f"[LỖI] Không tìm thấy thư mục dataset gốc tại: {RAW_DATA_DIR}")
        return

    ensure_dir(GLOBAL_SPLITS_DIR)

    supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_samples = []  # Danh sách các tuple: (relative_path_str, class_name)

    # 1. Quét toàn bộ các thư mục con trong dataset gốc
    class_folders = [d for d in sorted(RAW_DATA_DIR.iterdir()) if d.is_dir()]

    for folder in class_folders:
        folder_name = folder.name

        # Bỏ qua các thư mục nhiễu
        if folder_name in IGNORED_CLASSES:
            continue

        prefix = folder_name.split(".")[0].upper()
        if prefix not in PREFIX_TO_CLASS:
            continue

        target_class = PREFIX_TO_CLASS[prefix]
        files = [f for f in folder.iterdir() if f.suffix.lower() in supported_exts]

        for file_path in files:
            # Lưu đường dẫn tương đối so với thư mục dataset gốc
            rel_path = file_path.relative_to(RAW_DATA_DIR).as_posix()
            all_samples.append((rel_path, target_class))

    total_images = len(all_samples)
    if total_images == 0:
        print("[LỖI] Không tìm thấy file ảnh hợp lệ nào!")
        return

    print(f"[*] Tổng số ảnh biển báo hợp lệ được thu thập: {total_images} ảnh")

    # Tách đường dẫn và nhãn
    image_paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    # 2. Bước 1: Tách tập Train (70%) và tập Tạm (Val + Test = 30%)
    val_test_ratio = SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]  # 0.30
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=val_test_ratio,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    # Bước 2: Tách tập Tạm thành Val (15%) và Test (15%) tỉ lệ 50/50
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_labels,
    )

    # 3. Ghi ra 3 file text cố định (format: relative_path class_name)
    def save_split_file(filepath: Path, paths, lbls):
        with open(filepath, "w", encoding="utf-8") as f:
            for p, l in zip(paths, lbls):
                f.write(f"{p}\t{l}\n")

    save_split_file(TRAIN_SPLIT_FILE, train_paths, train_labels)
    save_split_file(VAL_SPLIT_FILE, val_paths, val_labels)
    save_split_file(TEST_SPLIT_FILE, test_paths, test_labels)

    # 4. Thống kê phân bố lớp
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    test_counts = Counter(test_labels)

    stats_df = pd.DataFrame([
        {
            "Nhóm Biển Báo": cls_name,
            "Train (70%)": train_counts.get(cls_name, 0),
            "Val (15%)": val_counts.get(cls_name, 0),
            "Test (15%)": test_counts.get(cls_name, 0),
            "Tổng": train_counts.get(cls_name, 0) + val_counts.get(cls_name, 0) + test_counts.get(cls_name, 0),
        }
        for cls_name in sorted(set(labels))
    ])

    print("\n[THÀNH CÔNG] Đã tạo xong Global Data Split!")
    print(stats_df.to_string(index=False))
    print("\n📁 Danh sách các file phân chia đã tạo:")
    print(f"    ↳ Train Split : {len(train_paths)} ảnh -> {TRAIN_SPLIT_FILE}")
    print(f"    ↳ Val Split   : {len(val_paths)} ảnh -> {VAL_SPLIT_FILE}")
    print(f"    ↳ Test Split  : {len(test_paths)} ảnh -> {TEST_SPLIT_FILE}")
    print("=" * 65)
    print("📌 LƯU Ý: Tất cả các thành viên khi train/test đều nạp dữ liệu từ 3 file trên!")


if __name__ == "__main__":
    main()
