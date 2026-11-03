"""
================================================================================
🌍 GLOBAL DATA SPLITTER & ROI CROPPING PIPELINE (BENCHMARKING FRAMEWORK)
================================================================================
Chức năng chính:
1. Đọc dữ liệu từ dataset gốc (images & labels YOLO format).
2. Phân chia dữ liệu toàn cục (Global Split) ở cấp độ ẢNH (Image-level):
   - Train : 70% (Dùng huấn luyện mô hình)
   - Val   : 15% (Dùng tinh chỉnh siêu tham số)
   - Test  : 15% (Tập đánh giá mù độc lập / Benchmarking)
   => Đảm bảo ZERO DATA LEAKAGE giữa các tập dữ liệu.
3. Tự động cắt các vùng chứa biển báo (ROI Cropping) lưu vào `data/cropped/`:
   - Phân cấp thư mục theo chuẩn ImageFolder: {split}/{category}/
   - Phục vụ trực tiếp cho các mô hình phân loại (SVM+HOG, LeNet, ResNet, CNN).
4. Khởi tạo danh sách định danh cố định trong `data/global_splits/`:
   - Phân loại (Classification): train_ids.txt, val_ids.txt, test_ids.txt
   - Phát hiện (Detection YOLO): yolo_train.txt, yolo_val.txt, yolo_test.txt, traffic_signs_4cls.yaml
================================================================================
"""

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Đảm bảo đường dẫn gốc
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    IMAGES_DIR,
    LABELS_DIR,
    CLASSES_FILE,
    CROPPED_DATA_DIR,
    GLOBAL_SPLITS_DIR,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_FILE,
    TEST_SPLIT_FILE,
    YOLO_TRAIN_FILE,
    YOLO_VAL_FILE,
    YOLO_TEST_FILE,
    YOLO_DATA_YAML,
    TARGET_CLASSES,
    CLASS_TO_IDX,
    PREFIX_TO_CLASS,
    IGNORED_PREFIXES,
    SPLIT_RATIOS,
    RANDOM_SEED,
    get_target_category,
    load_52_classes,
)
from src.utils import ensure_dir, imread_unicode, imwrite_unicode


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int, classes_list: list):
    """
    Đọc file nhãn YOLO và trả về danh sách các box hợp lệ.
    """
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            if cid >= len(classes_list):
                continue

            raw_class_name = classes_list[cid]
            target_cat = get_target_category(raw_class_name)
            if target_cat is None:
                # Bỏ qua các nhãn nhiễu: Camera, Biển phụ S.*, B.*
                continue

            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = int(round((xc - bw / 2) * img_w))
            y1 = int(round((yc - bh / 2) * img_h))
            x2 = int(round((xc + bw / 2) * img_w))
            y2 = int(round((yc + bh / 2) * img_h))

            # Giới hạn tọa độ trong biên ảnh
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(x1 + 1, min(img_w, x2))
            y2 = max(y1 + 1, min(img_h, y2))

            boxes.append({
                "class_id_52": cid,
                "class_name_52": raw_class_name,
                "category": target_cat,
                "cat_id": CLASS_TO_IDX[target_cat],
                "bbox_pixel": (x1, y1, x2, y2),
                "bbox_norm": (xc, yc, bw, bh),
            })
    return boxes


def split_dataset_stratified(image_items, seed=RANDOM_SEED):
    """
    Chia danh sách ảnh thành 3 tập Train/Val/Test độc lập ở cấp độ ảnh.
    Stratified theo nhóm biển chính có trong ảnh.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Gom nhóm các ảnh theo nhóm biển ưu tiên (để phân bổ đều 4 nhóm)
    grouped_by_cat = defaultdict(list)
    for item in image_items:
        cats = [b["category"] for b in item["boxes"]]
        if not cats:
            primary_cat = "Negative"
        else:
            primary_cat = Counter(cats).most_common(1)[0][0]
        grouped_by_cat[primary_cat].append(item)

    train_set, val_set, test_set = [], [], []

    for cat, items in grouped_by_cat.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(round(n * SPLIT_RATIOS["train"]))
        n_val = int(round(n * SPLIT_RATIOS["val"]))

        # Đảm bảo phân bổ hợp lý
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

        train_set.extend(train_items)
        val_set.extend(val_items)
        test_set.extend(test_items)

    # Trộn ngẫu nhiên lần cuối
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    return train_set, val_set, test_set


def create_global_splits(do_crop: bool = True, min_box_size: int = 10, seed: int = RANDOM_SEED):
    print("=" * 70)
    print("🌍 KHỞI TẠO GLOBAL DATA SPLIT CHO HỆ THỐNG BENCHMARK (QCVN 41:2019)")
    print("=" * 70)

    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        print(f"[-] [LỖI] Không tìm thấy thư mục dataset gốc tại: {IMAGES_DIR} hoặc {LABELS_DIR}")
        return

    classes_52 = load_52_classes()
    print(f"[*] Đã nạp {len(classes_52)} lớp chi tiết từ: {CLASSES_FILE}")
    print(f"[*] 4 Nhóm mục tiêu: {TARGET_CLASSES}")

    # 1. Quét toàn bộ ảnh và file nhãn tương ứng
    image_paths = sorted(list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png")))
    total_images = len(image_paths)
    print(f"[*] Quét tổng cộng {total_images} ảnh trong {IMAGES_DIR}...")

    image_items = []
    total_valid_boxes = 0

    for img_p in tqdm(image_paths, desc="Quét nhãn ảnh"):
        lbl_p = LABELS_DIR / f"{img_p.stem}.txt"
        
        # Đọc nhanh kích thước ảnh bằng OpenCV
        img = imread_unicode(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]

        boxes = parse_yolo_labels(lbl_p, w, h, classes_52)
        # Lọc các box quá nhỏ (nếu có yêu cầu min_box_size)
        filtered_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = b["bbox_pixel"]
            bw, bh = x2 - x1, y2 - y1
            if bw >= min_box_size and bh >= min_box_size:
                filtered_boxes.append(b)

        total_valid_boxes += len(filtered_boxes)
        image_items.append({
            "image_path": img_p,
            "stem": img_p.stem,
            "width": w,
            "height": h,
            "boxes": filtered_boxes,
        })

    print(f"[*] Tổng số ảnh hợp lệ: {len(image_items)}")
    print(f"[*] Tổng số bounding boxes hợp lệ (sau khi lọc bỏ biển phụ & nhiễu): {total_valid_boxes} boxes")

    # 2. Phân chia Train / Val / Test cố định (Image-level split)
    train_items, val_items, test_items = split_dataset_stratified(image_items, seed=seed)
    print(f"\n[*] Phân chia ở cấp độ ảnh (Zero Data Leakage):")
    print(f"    • Tập Train : {len(train_items)} ảnh ({len(train_items)/len(image_items)*100:.1f}%)")
    print(f"    • Tập Val   : {len(val_items)} ảnh ({len(val_items)/len(image_items)*100:.1f}%)")
    print(f"    • Tập Test  : {len(test_items)} ảnh ({len(test_items)/len(image_items)*100:.1f}%)")

    ensure_dir(GLOBAL_SPLITS_DIR)
    ensure_dir(CROPPED_DATA_DIR)

    split_configs = [
        ("train", train_items, TRAIN_SPLIT_FILE, YOLO_TRAIN_FILE),
        ("val", val_items, VAL_SPLIT_FILE, YOLO_VAL_FILE),
        ("test", test_items, TEST_SPLIT_FILE, YOLO_TEST_FILE),
    ]

    # 3. Thực hiện Cắt ROI & Ghi file chỉ mục
    for split_name, items, cls_split_file, yolo_split_file in split_configs:
        print(f"\n[*] Đang xử lý tập [{split_name.upper()}] ({len(items)} ảnh)...")
        cls_records = []
        yolo_records = []
        cat_counter = Counter()

        for item in tqdm(items, desc=f"Cropping & indexing {split_name}"):
            img_p = item["image_path"]
            # Lưu đường dẫn tương đối của ảnh gốc cho YOLO
            rel_img = img_p.relative_to(PROJECT_ROOT).as_posix()
            yolo_records.append(rel_img)

            if not item["boxes"]:
                continue

            # Đọc ảnh để cắt ROI nếu do_crop=True
            img = None
            if do_crop:
                img = imread_unicode(img_p)

            for idx, box in enumerate(item["boxes"]):
                cat = box["category"]
                crop_rel_name = f"{item['stem']}_{idx:02d}.jpg"
                crop_save_rel = f"{split_name}/{cat}/{crop_rel_name}"

                crop_success = True
                if do_crop:
                    if img is None:
                        crop_success = False
                    else:
                        x1, y1, x2, y2 = box["bbox_pixel"]
                        crop = img[y1:y2, x1:x2]
                        if crop.size > 0:
                            save_dest = CROPPED_DATA_DIR / crop_save_rel
                            crop_success = imwrite_unicode(save_dest, crop)
                        else:
                            crop_success = False

                if crop_success:
                    cat_counter[cat] += 1
                    # Format file split: relative_path \t category \t class_name_52
                    cls_records.append(f"{crop_save_rel}\t{cat}\t{box['class_name_52']}")

        # Ghi file split phân loại
        with open(cls_split_file, "w", encoding="utf-8") as f:
            f.write("\n".join(cls_records) + "\n")
        print(f"    -> Đã tạo {cls_split_file.name}: {len(cls_records)} mẫu ROI biển báo")
        for cat in TARGET_CLASSES:
            print(f"       • {cat:15s}: {cat_counter[cat]:4d} mẫu")

        # Ghi file split YOLO detection
        with open(yolo_split_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_records) + "\n")
        print(f"    -> Đã tạo {yolo_split_file.name}: {len(yolo_records)} ảnh toàn cảnh cho YOLO")

    # 4. Ghi file cấu hình YOLO data YAML
    yolo_yaml_content = f"""# Cấu hình dữ liệu YOLOv8 / YOLOv11 cho bài toán Nhận dạng Biển báo (4 nhóm QCVN 41:2019)
path: {PROJECT_ROOT.as_posix()}
train: data/global_splits/yolo_train.txt
val: data/global_splits/yolo_val.txt
test: data/global_splits/yolo_test.txt

nc: 4
names:
  0: Bien_Cam
  1: Bien_Nguy_Hiem
  2: Bien_Hieu_Lenh
  3: Bien_Chi_Dan
"""
    with open(YOLO_DATA_YAML, "w", encoding="utf-8") as f:
        f.write(yolo_yaml_content)
    print(f"\n[+] Đã tạo file cấu hình YOLO: {YOLO_DATA_YAML}")

    print("\n" + "=" * 70)
    print("✅ KHỞI TẠO GLOBAL DATA SPLIT HOÀN TẤT THÀNH CÔNG!")
    print(f"[*] Dữ liệu ROI cắt sẵn: {CROPPED_DATA_DIR}")
    print(f"[*] File split phân loại : {TRAIN_SPLIT_FILE}, {VAL_SPLIT_FILE}, {TEST_SPLIT_FILE}")
    print(f"[*] File split YOLO      : {YOLO_TRAIN_FILE}, {YOLO_VAL_FILE}, {YOLO_TEST_FILE}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Global Data Split & ROI Extraction Pipeline")
    parser.add_argument("--crop", action="store_true", default=True, help="Tự động cắt ROI và lưu vào data/cropped/")
    parser.add_argument("--no-crop", dest="crop", action="store_false", help="Chỉ chia split text, không cắt ảnh")
    parser.add_argument("--min-size", type=int, default=10, help="Kích thước tối thiểu của biển báo (mặc định: 10px)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help=f"Random seed (mặc định: {RANDOM_SEED})")

    args = parser.parse_args()
    create_global_splits(do_crop=args.crop, min_box_size=args.min_size, seed=args.seed)


if __name__ == "__main__":
    main()
