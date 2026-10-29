import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


def ensure_dir(dir_path: Path | str) -> Path:
    """Tạo thư mục nếu chưa tồn tại."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def imread_unicode(file_path: Path | str) -> Optional[np.ndarray]:
    """
    Đọc ảnh an toàn trên Windows với đường dẫn Unicode tiếng Việt.
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
    Lưu ảnh an toàn trên Windows với đường dẫn Unicode tiếng Việt.
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
