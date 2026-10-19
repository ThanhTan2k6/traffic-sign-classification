"""
Script tạo tập dữ liệu ROI chuẩn để bàn giao cho Thành viên 2.
Nhiệm vụ:
- Duyệt qua từng thư mục lớp biển báo (từ data/augmented hoặc data/cleaned).
- Áp dụng Giai đoạn 1 (Lọc màu HSV) + Giai đoạn 2 (Tìm đường viền & Cắt ROI).
- Chuẩn hóa kích thước ROI về 64x64.
- Lưu kết quả vào thư mục 'data/rois/<class_name>/'.
"""

from pathlib import Path
from tqdm import tqdm
import cv2

from src.config import AUGMENTED_DATA_DIR, CLEANED_DATA_DIR, ROI_OUTPUT_DIR, ROI_SIZE
from src.detection.color_filter import ColorFilter
from src.detection.roi_extractor import ROIExtractor
from src.utils import ensure_dir, imread_unicode, imwrite_unicode


def process_and_export_rois(
    source_dir: Path = AUGMENTED_DATA_DIR,
    output_dir: Path = ROI_OUTPUT_DIR,
    target_size: tuple = ROI_SIZE,
):
    """
    Xử lý hàng loạt ảnh: phát hiện biển báo, cắt ROI chuẩn hóa và xuất ra thư mục đích.
    """
    src_path = Path(source_dir)
    # Nếu chưa chạy bước augment thì chuyển sang dùng cleaned
    if not src_path.exists() or not any(src_path.iterdir()):
        src_path = Path(CLEANED_DATA_DIR)
        print(f"[!] Thư mục augmented chưa có, sử dụng thư mục: {src_path}")

    out_path = Path(output_dir)
    ensure_dir(out_path)

    color_filter = ColorFilter()
    roi_extractor = ROIExtractor(output_size=target_size)

    class_dirs = [d for d in src_path.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"[-] Không tìm thấy thư mục lớp nào tại: {src_path}")
        return

    print("=" * 60)
    print("XUẤT TẬP DỮ LIỆU ROI CHUẨN CHO THÀNH VIÊN 2 (ROI EXPORT)")
    print(f"[+] Nguồn: {src_path}")
    print(f"[+] Đích:  {out_path}")
    print(f"[+] Kích thước ROI chuẩn hóa: {target_size}")
    print("=" * 60)

    total_exported = 0
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for class_folder in class_dirs:
        class_name = class_folder.name
        class_out_dir = out_path / class_name
        ensure_dir(class_out_dir)

        image_files = [f for f in class_folder.iterdir() if f.suffix.lower() in supported_exts]
        exported_class_count = 0

        for img_file in tqdm(image_files, desc=f"Xuất ROI lớp {class_name:10s}", leave=False):
            img = imread_unicode(img_file)
            if img is None:
                continue

            # 1. Tạo mặt nạ lọc màu
            mask, _, _ = color_filter.create_traffic_sign_mask(img)

            # 2. Tìm và cắt các ROI
            rois = roi_extractor.extract_rois(img, mask)

            if rois:
                # Nếu tìm thấy ROI: lấy ROI có diện tích lớn nhất (chính là biển báo trong ảnh)
                best_roi = max(rois, key=lambda r: r["area"])
                roi_img = best_roi["roi_normalized"]
            else:
                # Dự phòng: Nếu ảnh đầu vào vốn là ảnh đã crop sẵn mà màu bị bạc/không bắt được mask,
                # thì resize trực tiếp ảnh gốc về chuẩn 64x64 để không làm mất dữ liệu huấn luyện.
                roi_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

            save_file = class_out_dir / f"{img_file.stem}_roi.png"
            if imwrite_unicode(save_file, roi_img):
                exported_class_count += 1
                total_exported += 1

        print(f"[✓] Lớp '{class_name}': Đã xuất {exported_class_count} ảnh ROI chuẩn")

    print("\n" + "=" * 60)
    print(f"[+] HOÀN THÀNH! Tổng cộng đã xuất {total_exported} ảnh ROI.")
    print(f"[+] Dữ liệu đã sẵn sàng tại: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    process_and_export_rois()
