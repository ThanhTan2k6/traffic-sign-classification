"""
Module Tăng cường Dữ liệu (Data Augmentation) - Thành viên 1 phụ trách.
Nhiệm vụ:
- Khắc phục hiện tượng mất cân bằng dữ liệu giữa các lớp biển báo (Class Imbalance).
- Áp dụng các biến đổi hình học và màu sắc phù hợp với biển báo giao thông:
  + Xoay góc nhỏ (-15 đến +15 độ)
  + Điều chỉnh độ sáng và độ tương phản (mô phỏng nắng/tối/bóng râm)
  + Làm mờ Gaussian (mô phỏng mờ do chuyển động - motion blur)
  + Thu phóng (Zoom in / Zoom out nhẹ)
  + Dịch chuyển nhẹ (Translation)
"""

import random
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from tqdm import tqdm

from src.config import CLEANED_DATA_DIR, AUGMENTED_DATA_DIR
from src.utils import ensure_dir, imread_unicode, imwrite_unicode


class DataAugmentor:
    """Class cung cấp các phương thức biến đổi và tăng cường ảnh biển báo."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def random_rotation(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """Xoay ảnh một góc ngẫu nhiên trong khoảng [-max_angle, max_angle]."""
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
        alpha_range: tuple = (0.8, 1.2),
        beta_range: tuple = (-25, 25),
    ) -> np.ndarray:
        """Thay đổi độ tương phản (alpha) và độ sáng (beta)."""
        alpha = random.uniform(*alpha_range)
        beta = random.uniform(*beta_range)
        augmented = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return augmented

    @staticmethod
    def random_blur(image: np.ndarray, max_ksize: int = 3) -> np.ndarray:
        """Làm mờ Gaussian nhẹ ngẫu nhiên."""
        if random.random() > 0.5:
            ksize = random.choice([3, 5])
            if ksize <= max_ksize:
                return cv2.GaussianBlur(image, (ksize, ksize), 0)
        return image

    @staticmethod
    def random_shift(image: np.ndarray, max_shift_ratio: float = 0.1) -> np.ndarray:
        """Dịch chuyển ảnh theo chiều ngang và dọc."""
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
        target_samples_per_class: Optional[int] = None,
    ):
        """
        Tự động tăng cường dữ liệu cho các lớp ít ảnh để đạt target_samples_per_class.
        Nếu target_samples_per_class là None: sẽ lấy số mẫu của lớp nhiều nhất hoặc trung vị.
        """
        source_path = Path(source_dir)
        target_path = Path(target_dir)

        print("=" * 60)
        print("BẮT ĐẦU TĂNG CƯỜNG DỮ LIỆU ĐỂ CÂN BẰNG LỚP (DATA AUGMENTATION)")
        print(f"[+] Thư mục nguồn: {source_path}")
        print(f"[+] Thư mục đích:   {target_path}")
        print("=" * 60)

        classes = [d for d in source_path.iterdir() if d.is_dir()]
        if not classes:
            print("[-] Không tìm thấy thư mục lớp!")
            return

        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        class_files = {}

        for folder in classes:
            files = [f for f in folder.iterdir() if f.suffix.lower() in supported_exts]
            class_files[folder.name] = files

        counts = [len(f) for f in class_files.values()]
        max_count = max(counts)
        median_count = int(np.median(counts))

        # Nếu không chỉ định, lấy mức trung vị hoặc tối thiểu 50 ảnh
        if target_samples_per_class is None:
            target_samples = max(median_count, 50)
        else:
            target_samples = target_samples_per_class

        print(f"[+] Số mẫu nhiều nhất: {max_count} | Trung vị: {median_count}")
        print(f"[+] Mục tiêu cân bằng cho mỗi lớp (Target count): {target_samples} ảnh\n")

        for class_name, files in class_files.items():
            class_target_dir = target_path / class_name
            ensure_dir(class_target_dir)

            count = len(files)
            # Copy ảnh gốc sang thư mục target trước
            for f in files:
                dst = class_target_dir / f.name
                if not dst.exists():
                    img = imread_unicode(f)
                    if img is not None:
                        imwrite_unicode(dst, img)

            # Nếu lớp bị thiếu mẫu so với target -> tiến hành Augmentation
            if count < target_samples and count > 0:
                needed = target_samples - count
                print(f"[~] Lớp '{class_name}': có {count} ảnh -> Đang tạo thêm {needed} ảnh augmented...")

                for i in range(needed):
                    src_file = random.choice(files)
                    img = imread_unicode(src_file)
                    if img is None:
                        continue
                    aug_img = self.apply_random_pipeline(img)
                    aug_name = f"aug_{i+1:04d}_{src_file.stem}.png"
                    imwrite_unicode(class_target_dir / aug_name, aug_img)
            else:
                print(f"[✓] Lớp '{class_name}': đã có đủ {count} ảnh (>= {target_samples})")

        print("\n[+] Hoàn thành tăng cường dữ liệu!")


if __name__ == "__main__":
    augmentor = DataAugmentor()
    augmentor.balance_dataset()
