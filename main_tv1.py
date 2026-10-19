"""
================================================================================
SCRIPT THỰC THI CHÍNH DÀNH CHO THÀNH VIÊN 1 (MEMBER 1 WORKFLOW)
Phụ trách: Dữ liệu, Tiền xử lý & Khoanh vùng (Khối 1 + Khối 2)
================================================================================
Các lệnh hỗ trợ:
    python main_tv1.py --all           # Chạy toàn bộ quy trình: Clean -> Augment -> Export ROI
    python main_tv1.py --clean         # Chỉ làm sạch dữ liệu (loại bỏ Camera, ảnh hỏng)
    python main_tv1.py --augment       # Tăng cường dữ liệu (cân bằng các lớp)
    python main_tv1.py --export-rois   # Lọc màu & Trích xuất ROI chuẩn 64x64 cho TV2
    python main_tv1.py --demo [path]   # Hiển thị trực quan từng bước trên 1 ảnh mẫu
"""

import argparse
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

from src.config import RAW_DATA_DIR, CLEANED_DATA_DIR, AUGMENTED_DATA_DIR, ROI_OUTPUT_DIR
from src.preprocessing.cleaner import clean_dataset
from src.preprocessing.augmentor import DataAugmentor
from src.detection.color_filter import ColorFilter
from src.detection.roi_extractor import ROIExtractor
from src.extract_roi_dataset import process_and_export_rois
from src.utils import imread_unicode, plot_image_steps


def run_demo(image_path: str = None):
    """Demo trực quan hóa từng bước xử lý của Khối 1 & Khối 2 trên 1 ảnh."""
    print("=" * 60)
    print("DEMO TRỰC QUAN HÓA GIAI ĐOẠN 1 & 2 TRÊN ẢNH MẪU")
    print("=" * 60)

    # Nếu không cung cấp ảnh, tự chọn 1 ảnh ngẫu nhiên trong dataset
    if image_path is None or not Path(image_path).exists():
        candidates = list(RAW_DATA_DIR.glob("*/*.png")) + list(RAW_DATA_DIR.glob("*/*.jpg"))
        if not candidates:
            print("[-] Không tìm thấy ảnh nào trong dataset để chạy demo!")
            return
        chosen_file = random.choice(candidates)
        print(f"[+] Chọn ảnh ngẫu nhiên: {chosen_file}")
    else:
        chosen_file = Path(image_path)
        print(f"[+] Sử dụng ảnh chỉ định: {chosen_file}")

    bgr_img = imread_unicode(chosen_file)
    if bgr_img is None:
        print("[-] Không thể đọc ảnh.")
        return

    # 1. Giai đoạn 1: Lọc màu
    color_filter = ColorFilter()
    final_mask, masked_img, masks = color_filter.create_traffic_sign_mask(bgr_img)

    # 2. Giai đoạn 2: Phân đoạn hình thái học & Tìm ROI
    roi_extractor = ROIExtractor()
    cleaned_mask = roi_extractor.clean_mask_morphology(final_mask)
    detected_rois = roi_extractor.extract_rois(bgr_img, final_mask)
    annotated = roi_extractor.draw_detections(bgr_img, detected_rois)

    print(f"[+] Số lượng vùng ROI phát hiện: {len(detected_rois)}")
    for idx, r in enumerate(detected_rois):
        print(f"    - ROI #{idx+1}: BBox={r['bbox']}, Diện tích={r['area']:.1f}, AspectRatio={r['aspect_ratio']:.2f}")

    # Hiển thị trực quan các bước
    steps = [
        bgr_img,
        final_mask,
        cleaned_mask,
        annotated,
    ]
    titles = [
        "1. Ảnh gốc (Input)",
        "2. Mặt nạ HSV (Color Mask)",
        "3. Sau lọc Morphology",
        f"4. Phát hiện ({len(detected_rois)} ROIs)",
    ]

    # Nếu có ROI cắt được, thêm ảnh ROI đầu tiên vào để xem
    if detected_rois:
        steps.append(detected_rois[0]["roi_normalized"])
        titles.append("5. ROI chuẩn hóa (64x64)")

    plot_image_steps(steps, titles, figsize=(16, 4))


def main():
    parser = argparse.ArgumentParser(description="Chương trình thực thi nhiệm vụ của Thành viên 1")
    parser.add_argument("--clean", action="store_true", help="Làm sạch dataset và loại bỏ thư mục Camera")
    parser.add_argument("--augment", action="store_true", help="Tăng cường dữ liệu cân bằng các lớp")
    parser.add_argument("--export-rois", action="store_true", help="Trích xuất và xuất toàn bộ ROI chuẩn hóa")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ quy trình từ Clean -> Augment -> Export ROI")
    parser.add_argument("--demo", nargs="?", const="random", default=None, help="Chạy demo hiển thị trực quan (tùy chọn đường dẫn ảnh)")

    args = parser.parse_args()

    # Mặc định nếu không truyền tham số nào thì in hướng dẫn hoặc chạy demo
    if not any([args.clean, args.augment, args.export_rois, args.all, args.demo]):
        print("\n" + "=" * 60)
        print("BẠN ĐANG CHẠY CHƯƠNG TRÌNH CỦA THÀNH VIÊN 1")
        print("Sử dụng một trong các cờ sau:")
        print("  python main_tv1.py --demo         : Chạy demo trực quan hóa trên 1 ảnh")
        print("  python main_tv1.py --clean        : Làm sạch dữ liệu")
        print("  python main_tv1.py --augment      : Tăng cường dữ liệu")
        print("  python main_tv1.py --export-rois  : Xuất tập ROI chuẩn hóa cho TV2")
        print("  python main_tv1.py --all          : Chạy toàn bộ 3 bước trên")
        print("=" * 60 + "\n")
        return

    if args.clean or args.all:
        clean_dataset()

    if args.augment or args.all:
        augmentor = DataAugmentor()
        augmentor.balance_dataset()

    if args.export_rois or args.all:
        process_and_export_rois()

    if args.demo:
        img_p = None if args.demo == "random" else args.demo
        run_demo(img_p)


if __name__ == "__main__":
    main()
