"""
================================================================================
SCRIPT HUẤN LUYỆN MÔ HÌNH YOLO (LỰA CHỌN B CHO THÀNH VIÊN 2)
Huấn luyện mô hình phát hiện đối tượng End-to-End (YOLOv8 / YOLOv11) trên ảnh toàn cảnh.
Cấu hình dữ liệu: data/traffic_signs_4cls.yaml
Trọng số lưu tại: checkpoints/deep_learning/best_yolo.pt
================================================================================
"""

import shutil
from pathlib import Path

from src.config import YOLO_DATA_YAML, CHECKPOINTS_DIR
from src.utils import ensure_dir


def train_yolo(epochs=25, imgsz=640, model_type="yolov8n.pt", batch=16):
    print("=" * 65)
    print("🚀 [THÀNH VIÊN 2] BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH PHÁT HIỆN YOLOv8")
    print("=" * 65)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[-] Lỗi: Chưa cài đặt thư viện 'ultralytics'.")
        print("    Vui lòng chạy lệnh: pip install ultralytics")
        return

    if not YOLO_DATA_YAML.exists():
        print(f"[-] Không tìm thấy file cấu hình dữ liệu YOLO tại: {YOLO_DATA_YAML}")
        print("    Vui lòng chạy lệnh: python global_data_split.py để khởi tạo trước.")
        return

    print(f"[*] Cấu hình dữ liệu YOLO : {YOLO_DATA_YAML}")
    print(f"[*] Mô hình khởi tạo     : {model_type}")
    print(f"[*] Kích thước ảnh (imgsz): {imgsz}")
    print(f"[*] Số epochs            : {epochs}")

    # Khởi tạo mô hình
    model = YOLO(model_type)

    # Chạy huấn luyện
    results = model.train(
        data=str(YOLO_DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(CHECKPOINTS_DIR / "runs"),
        name="traffic_signs_yolo",
        exist_ok=True,
    )

    # Sao chép trọng số tốt nhất về thư mục checkpoints/deep_learning/
    save_dir = ensure_dir(CHECKPOINTS_DIR / "deep_learning")
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    dest_pt = save_dir / "best_yolo.pt"

    if best_pt.exists():
        shutil.copy(best_pt, dest_pt)
        print(f"\n[+] [BEST] Đã sao chép trọng số tốt nhất về: {dest_pt}")
    else:
        print(f"[!] Kiểm tra trọng số huấn luyện tại: {results.save_dir}")

    print("\n[+] Huấn luyện YOLO hoàn tất!")


if __name__ == "__main__":
    train_yolo()
