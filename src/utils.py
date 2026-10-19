"""
Hàm tiện ích (Utility functions) dùng chung trong dự án:
- Đọc/ghi ảnh an toàn với đường dẫn tiếng Việt / Windows
- Hiển thị và so sánh các bước xử lý ảnh
- Tạo thư mục tự động
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(dir_path: Path | str) -> Path:
    """Tạo thư mục nếu chưa tồn tại."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def imread_unicode(file_path: Path | str) -> Optional[np.ndarray]:
    """
    Đọc ảnh an toàn trên Windows với đường dẫn có dấu hoặc ký tự đặc biệt.
    OpenCV cv2.imread thường trả về None nếu đường dẫn có Unicode trên Windows.
    """
    try:
        path_str = str(file_path)
        with open(path_str, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        print(f"[-] Lỗi đọc ảnh {file_path}: {e}")
        return None


def imwrite_unicode(file_path: Path | str, image: np.ndarray) -> bool:
    """
    Lưu ảnh an toàn trên Windows với đường dẫn có dấu hoặc ký tự đặc biệt.
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix if path.suffix else ".png"
        success, encoded_img = cv2.imencode(ext, image)
        if success:
            with open(path, "wb") as f:
                f.write(encoded_img)
            return True
        return False
    except Exception as e:
        print(f"[-] Lỗi ghi ảnh {file_path}: {e}")
        return False


def plot_image_steps(
    images: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = (15, 5),
    cmap: Optional[str] = None,
):
    """
    Hiển thị trực quan các bước xử lý ảnh cạnh nhau bằng matplotlib.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:  # Ảnh xám / Mặt nạ nhị phân
            ax.imshow(img, cmap="gray")
        else:  # Ảnh màu BGR -> chuyển sang RGB để matplotlib hiển thị đúng màu
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
