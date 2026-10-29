"""
Script huấn luyện cho Phương pháp 3 (Thành viên 3).
Đọc dữ liệu tuân thủ Global Data Split (TRAIN_SPLIT_FILE, VAL_SPLIT_FILE).
"""

from src.config import TRAIN_SPLIT_FILE, VAL_SPLIT_FILE, CHECKPOINTS_DIR


def main():
    print("=" * 60)
    print("🚀 HUẤN LUYỆN MÔ HÌNH PHƯƠNG PHÁP 3 (THÀNH VIÊN 3)")
    print("=" * 60)
    print(f"[*] Tập Train : {TRAIN_SPLIT_FILE}")
    print(f"[*] Tập Val   : {VAL_SPLIT_FILE}")
    print(f"[*] Thư mục lưu: {CHECKPOINTS_DIR / 'model_3'}")
    # TODO: [Thành viên 3] Triển khai huấn luyện và lưu checkpoint
    raise NotImplementedError("TODO: Thành viên 3 cài đặt huấn luyện cho Model 3")


if __name__ == "__main__":
    main()
