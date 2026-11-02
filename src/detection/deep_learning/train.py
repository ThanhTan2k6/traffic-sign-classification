"""
================================================================================
ENTRYPOINT HUẤN LUYỆN CHO THÀNH VIÊN 2 (DEEP LEARNING)
Thành viên 2 có thể tùy ý lựa chọn 1 trong 2 hướng tiếp cận:
1. Huấn luyện Custom CNN (phân loại trên ảnh ROI cắt sẵn):
    python -m src.detection.deep_learning.train --arch cnn --epochs 20

2. Huấn luyện YOLOv8 (phát hiện toàn trình trên ảnh lớn):
    python -m src.detection.deep_learning.train --arch yolo --epochs 25
================================================================================
"""

import argparse
import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Deep Learning cho Thành viên 2")
    parser.add_argument(
        "--arch",
        type=str,
        default="cnn",
        choices=["cnn", "yolo"],
        help="Chọn hướng triển khai: 'cnn' (Custom CNN trên ảnh ROI) hoặc 'yolo' (YOLOv8 phát hiện toàn cảnh)",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Số lượt huấn luyện (epochs)")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học (Learning Rate)")

    args = parser.parse_args()

    if args.arch == "cnn":
        from src.detection.deep_learning.train_cnn import train_cnn
        train_cnn(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.arch == "yolo":
        from src.detection.deep_learning.train_yolo import train_yolo
        train_yolo(epochs=args.epochs, batch=args.batch_size)


if __name__ == "__main__":
    main()
