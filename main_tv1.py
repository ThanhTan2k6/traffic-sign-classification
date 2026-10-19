"""
================================================================================
SCRIPT ĐIỀU PHỐI DÀNH CHO THÀNH VIÊN 1 (MEMBER 1 WORKFLOW SKELETON)
Phụ trách: Dữ liệu, Tiền xử lý & Khoanh vùng (Khối 1 + Khối 2)
================================================================================
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Chương trình thực thi nhiệm vụ của Thành viên 1")
    parser.add_argument("--clean", action="store_true", help="Làm sạch dataset và loại bỏ thư mục Camera")
    parser.add_argument("--augment", action="store_true", help="Tăng cường dữ liệu cân bằng các lớp")
    parser.add_argument("--export-rois", action="store_true", help="Trích xuất và xuất toàn bộ ROI chuẩn hóa")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ quy trình từ Clean -> Augment -> Export ROI")
    parser.add_argument("--demo", nargs="?", const="random", default=None, help="Chạy demo hiển thị trực quan")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("KHUNG THỰC THI NHIỆM VỤ THÀNH VIÊN 1")
    print("Các cờ lệnh: --clean, --augment, --export-rois, --all, --demo")
    print("TODO: Thành viên 1 hoàn thiện cài đặt các hàm trong src/ để chạy các chức năng này.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
