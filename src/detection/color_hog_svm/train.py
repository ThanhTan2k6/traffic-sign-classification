"""
Script huấn luyện cho Phương pháp 1: HOG + SVM (Thành viên 1).
Đọc dữ liệu tuân thủ Global Data Split (TRAIN_SPLIT_FILE, VAL_SPLIT_FILE).
"""

from src.config import TRAIN_SPLIT_FILE, VAL_SPLIT_FILE, CHECKPOINTS_DIR


def main():
    print("=" * 60)
    print("🚀 HUẤN LUYỆN MÔ HÌNH HOG + SVM (THÀNH VIÊN 1)")
    print("=" * 60)
    print(f"[*] Tập Train : {TRAIN_SPLIT_FILE}")
    print(f"[*] Tập Val   : {VAL_SPLIT_FILE}")
    print(f"[*] Thư mục lưu: {CHECKPOINTS_DIR / 'color_hog_svm'}")
    # TODO: [Thành viên 1] Triển khai huấn luyện và lưu checkpoint best_svm.pkl
    raise NotImplementedError("TODO: Thành viên 1 cài đặt huấn luyện cho ColorHogSvm")


if __name__ == "__main__":
    main()
