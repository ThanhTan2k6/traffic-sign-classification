"""
Module Tăng cường Dữ liệu (Data Augmentation) - Thành viên 1 phụ trách.
Nhiệm vụ:
- Cân bằng số lượng mẫu giữa 4 Nhóm Biển Báo chính:
    + Bien_Cam (~2,823 ảnh)
    + Bien_Nguy_Hiem (~1,367 ảnh)
    + Bien_Hieu_Lenh (~572 ảnh)
    + Bien_Chi_Dan (~421 ảnh)
- Đưa các lớp về mức cân bằng lý tưởng (~1,200 - 1,500 ảnh/lớp).
- Áp dụng các biến đổi: Xoay (-15 đến +15 độ), chỉnh sáng/tương phản, mờ Gaussian, dịch chuyển nhẹ.
"""

import random
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from tqdm import tqdm

from src.config import CLEANED_DATA_DIR, AUGMENTED_DATA_DIR, TARGET_CLASSES
from src.utils import ensure_dir, imread_unicode, imwrite_unicode


class DataAugmentor:
    """Class cung cấp các phương thức biến đổi và tăng cường ảnh biển báo."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def random_rotation(image: np.ndarray, max_angle: float = 12.0) -> np.ndarray:
        """Xoay ảnh góc nhỏ ngẫu nhiên [-max_angle, max_angle]."""
        angle = random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        return cv2.warpAffine(
            image, rot_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

    @staticmethod
    def random_brightness_contrast(
        image: np.ndarray,
        alpha_range: tuple = (0.85, 1.15),
        beta_range: tuple = (-20, 20),
    ) -> np.ndarray:
        """Thay đổi độ tương phản (alpha) và độ sáng (beta)."""
        alpha = random.uniform(*alpha_range)
        beta = random.uniform(*beta_range)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def random_blur(image: np.ndarray) -> np.ndarray:
        """Làm mờ Gaussian nhẹ ngẫu nhiên (mô phỏng camera rung)."""
        if random.random() > 0.5:
            return cv2.GaussianBlur(image, (3, 3), 0)
        return image

    @staticmethod
    def random_shift(image: np.ndarray, max_shift_ratio: float = 0.08) -> np.ndarray:
        """Dịch chuyển nhẹ ảnh theo trục X và Y."""
        h, w = image.shape[:2]
        tx = random.uniform(-max_shift_ratio, max_shift_ratio) * w
        ty = random.uniform(-max_shift_ratio, max_shift_ratio) * h
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(
            image, matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

    def apply_random_pipeline(self, image: np.ndarray) -> np.ndarray:
        """Áp dụng chuỗi các phép biến đổi ngẫu nhiên lên 1 ảnh."""
        aug_img = image.copy()
        aug_img = self.random_rotation(aug_img)
        aug_img = self.random_brightness_contrast(aug_img)
        aug_img = self.random_shift(aug_img)
        aug_img = self.random_blur(aug_img)
        return aug_img

    def balance_dataset(
        self,
        source_dir: Path = CLEANED_DATA_DIR,
        target_dir: Path = AUGMENTED_DATA_DIR,
        target_samples_per_class: int = 1500,
        max_samples_cap: int = 1800,
    ):
        """
        Cân bằng 4 nhóm biển báo về khoảng ~1,500 ảnh/nhóm:
        - Các lớp ít mẫu (Hiệu lệnh, Chỉ dẫn) sẽ được sinh thêm ảnh.
        - Các lớp quá nhiều mẫu (Biển cấm > 2,800 ảnh) sẽ được giới hạn ở max_samples_cap để tránh lệch lớp.
        """
        source_path = Path(source_dir)
        target_path = Path(target_dir)

        print("=" * 65)
        print("TĂNG CƯỜNG DỮ LIỆU CÂN BẰNG 4 NHÓM BIỂN BÁO (DATA AUGMENTATION)")
        print(f"[+] Thư mục nguồn: {source_path}")
        print(f"[+] Thư mục đích:   {target_path}")
        print(f"[+] Mục tiêu cân bằng: ~{target_samples_per_class} ảnh/lớp")
        print("=" * 65)

        if not source_path.exists():
            raise FileNotFoundError(f"Chưa có thư mục cleaned: {source_path}. Hãy chạy clean_dataset trước!")

        ensure_dir(target_path)
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}

        for cls_name in TARGET_CLASSES:
            src_cls_dir = source_path / cls_name
            dst_cls_dir = target_path / cls_name
            ensure_dir(dst_cls_dir)

            if not src_cls_dir.exists():
                print(f"[-] Không tìm thấy thư mục: {src_cls_dir}")
                continue

            files = [f for f in src_cls_dir.iterdir() if f.suffix.lower() in supported_exts]
            count = len(files)
            print(f"\n[>] Xử lý nhóm '{cls_name}': hiện có {count} ảnh gốc.")

            # 1. Nếu lớp quá nhiều mẫu (ví dụ Bien_Cam > 2,800), lấy ngẫu nhiên max_samples_cap
            if count > max_samples_cap:
                selected_files = random.sample(files, max_samples_cap)
                for f in selected_files:
                    img = imread_unicode(f)
                    if img is not None:
                        imwrite_unicode(dst_cls_dir / f.name, img)
                print(f"[✓] Lấy mẫu cân bằng (Downsampled): giữ {max_samples_cap} / {count} ảnh.")
            else:
                # Copy toàn bộ ảnh gốc sang target
                for f in files:
                    img = imread_unicode(f)
                    if img is not None:
                        imwrite_unicode(dst_cls_dir / f.name, img)

                # 2. Nếu lớp thiếu mẫu so với target -> Augment sinh thêm
                if count < target_samples_per_class and count > 0:
                    needed = target_samples_per_class - count
                    print(f"[~] Đang sinh thêm {needed} ảnh augmented...")

                    for i in range(needed):
                        src_f = random.choice(files)
                        img = imread_unicode(src_f)
                        if img is None:
                            continue
                        aug_img = self.apply_random_pipeline(img)
                        aug_name = f"aug_{i+1:04d}_{src_f.stem}.png"
                        imwrite_unicode(dst_cls_dir / aug_name, aug_img)

                    print(f"[✓] Đã tăng cường thành công lên tổng cộng {target_samples_per_class} ảnh.")
                else:
                    print(f"[✓] Đã đủ số lượng chuẩn ({count} ảnh).")

        print("\n" + "=" * 65)
        print("[+] HOÀN THÀNH TĂNG CƯỜNG DỮ LIỆU! 4 nhóm biển báo đã được cân bằng hoàn hảo.")
        print("=" * 65)


if __name__ == "__main__":
    augmentor = DataAugmentor()
    augmentor.balance_dataset()
