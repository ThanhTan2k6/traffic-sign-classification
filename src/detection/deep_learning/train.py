"""
================================================================================
SCRIPT HUẤN LUYỆN MÔ HÌNH DEEP LEARNING (THÀNH VIÊN 2)
Huấn luyện mạng nơ-ron tích chập Enhanced LeNet-5 trên tập ROI biển báo (data/cropped/).
Tự động lưu trọng số tốt nhất vào: checkpoints/deep_learning/best_cnn.pth
================================================================================
"""

import argparse
import sys
from pathlib import Path

# Đảm bảo đường dẫn gốc
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.deep_learning.train_cnn import train_cnn


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Enhanced LeNet-5 cho Thành viên 2")
    parser.add_argument("--epochs", type=int, default=20, help="Số epochs huấn luyện (mặc định: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch (mặc định: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học Learning Rate (mặc định: 0.001)")

    args = parser.parse_args()
    print("=" * 65)
    print("🚀 [THÀNH VIÊN 2] BẮT ĐẦU HUẤN LUYỆN ENHANCED LENET-5 (PYTORCH)")
    print("=" * 65)
    train_cnn(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
