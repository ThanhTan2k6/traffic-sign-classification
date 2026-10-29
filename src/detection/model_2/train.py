"""
Script huấn luyện cho Phương pháp 2 (Thành viên 2).
Đọc dữ liệu tuân thủ Global Data Split (TRAIN_SPLIT_FILE, VAL_SPLIT_FILE).
"""

from src.config import TRAIN_SPLIT_FILE, VAL_SPLIT_FILE, CHECKPOINTS_DIR


def main():
    print("=" * 60)
    print("🚀 HUẤN LUYỆN MÔ HÌNH PHƯƠNG PHÁP 2 (THÀNH VIÊN 2)")
    print("=" * 60)
    print(f"[*] Tập Train : {TRAIN_SPLIT_FILE}")
    print(f"[*] Tập Val   : {VAL_SPLIT_FILE}")
    print(f"[*] Thư mục lưu: {CHECKPOINTS_DIR / 'model_2'}")
    # TODO: [Thành viên 2] Triển khai huấn luyện và lưu checkpoint
    raise NotImplementedError("TODO: Thành viên 2 cài đặt huấn luyện cho Model 2")


if __name__ == "__main__":
    main()
