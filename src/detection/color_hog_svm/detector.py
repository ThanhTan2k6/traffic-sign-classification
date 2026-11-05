"""
================================================================================
PHƯƠNG PHÁP 1 (THÀNH VIÊN 1 PHỤ TRÁCH): COLOR + HOG + SVM DETECTOR / CLASSIFIER
Pipeline: Lọc màu HSV + Lưới không gian + Hình học + HOG + Bộ phân loại SVM RBF
================================================================================
"""

from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import joblib
import numpy as np
from scipy.special import softmax

from src.config import (
    TARGET_CLASSES,
    CLASS_TO_IDX,
    PROJECT_ROOT,
    CHECKPOINTS_DIR,
)
from src.detection.base_detector import BaseDetector
from src.detection.color_hog_svm.features import extract_s3_features


class ColorHogSvmDetector(BaseDetector):
    """
    Bộ phân loại biển báo sử dụng HSV + Grid + Geometry + HOG kết hợp SVM RBF.
    Kế thừa BaseDetector để tương thích 100% với hệ thống Benchmark và App.
    """

    LABEL_MAPPING = {
        "cấm (prohibitory)": "Bien_Cam",
        "cảnh báo (warning)": "Bien_Nguy_Hiem",
        "hiệu lệnh (mandatory)": "Bien_Hieu_Lenh",
        "chỉ dẫn (information)": "Bien_Chi_Dan",
        "phụ (supplementary)": "Bien_Chi_Dan",
        # Hỗ trợ nhãn tiếng Anh / chuẩn QCVN nếu có
        "Bien_Cam": "Bien_Cam",
        "Bien_Nguy_Hiem": "Bien_Nguy_Hiem",
        "Bien_Hieu_Lenh": "Bien_Hieu_Lenh",
        "Bien_Chi_Dan": "Bien_Chi_Dan",
        "prohibitory": "Bien_Cam",
        "warning": "Bien_Nguy_Hiem",
        "mandatory": "Bien_Hieu_Lenh",
        "information": "Bien_Chi_Dan",
    }

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__(model_name="color_hog_svm", checkpoint_path=checkpoint_path)
        self.scaler = None
        self.classes_ = None
        self.load_model(checkpoint_path)

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Nạp mô hình SVM và Scaler từ file checkpoint."""
        if checkpoint_path is None:
            # Ưu tiên Stage 2 release
            cand_paths = [
                PROJECT_ROOT / "models" / "stage2_best.joblib",
                CHECKPOINTS_DIR / "color_hog_svm" / "best_svm.pkl",
                PROJECT_ROOT / "models" / "stage1_hsv_vn_best.joblib",
            ]
            for p in cand_paths:
                if p.exists():
                    checkpoint_path = str(p)
                    break

        if checkpoint_path is None or not Path(checkpoint_path).exists():
            self.model = None
            return

        self.checkpoint_path = checkpoint_path
        try:
            bundle = joblib.load(checkpoint_path)
            if isinstance(bundle, dict):
                self.model = bundle.get("model")
                self.scaler = bundle.get("scaler")
                self.classes_ = getattr(self.model, "classes_", bundle.get("canonical_groups"))
            else:
                self.model = bundle
                self.scaler = None
                self.classes_ = getattr(self.model, "classes_", None)

            print(f"[+] [ColorHogSVM] Đã nạp thành công mô hình từ: {checkpoint_path}")
        except Exception as e:
            print(f"[-] [ColorHogSVM] Lỗi khi nạp checkpoint {checkpoint_path}: {e}")
            self.model = None

    def preprocess(self, image: np.ndarray, **kwargs) -> Any:
        """Trích xuất vector đặc trưng S3 (457 chiều) và chuẩn hóa qua Scaler."""
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        feat = extract_s3_features(image, w_orig=w, h_orig=h)

        if self.scaler is not None:
            scaled = self.scaler.transform([feat])
            return scaled
        return np.array([feat], dtype=np.float64)

    def inference(self, preprocessed_input: Any) -> Any:
        """Dự đoán nhãn và tính độ tin cậy từ decision function."""
        if self.model is None:
            raise NotImplementedError("Mô hình ColorHogSVM chưa nạp checkpoint!")

        if preprocessed_input is None:
            return None

        pred_raw = self.model.predict(preprocessed_input)[0]

        # Tính toán confidence score
        confidence = 0.85
        try:
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(preprocessed_input)[0]
                confidence = float(np.max(probs))
            elif hasattr(self.model, "decision_function"):
                df = self.model.decision_function(preprocessed_input)
                if df.ndim == 1:
                    probs = 1.0 / (1.0 + np.exp(-df))
                    confidence = float(probs[0])
                else:
                    probs = softmax(df, axis=1)[0]
                    confidence = float(np.max(probs))
        except Exception:
            confidence = 0.85

        return {
            "raw_pred": pred_raw,
            "confidence": confidence,
        }

    def postprocess(self, raw_output: Any) -> Dict[str, Any]:
        """Quy đổi nhãn dự đoán về chuẩn TARGET_CLASSES ('Bien_Cam', 'Bien_Nguy_Hiem',...)."""
        if raw_output is None:
            return {
                "class_name": "Unknown",
                "class_id": -1,
                "confidence": 0.0,
                "bbox": None,
            }

        raw_label = str(raw_output.get("raw_pred", "Unknown"))
        confidence = float(raw_output.get("confidence", 0.0))

        # Ánh xạ nhãn tiếng Việt / chi tiết sang 4 nhóm chuẩn
        canonical = self.LABEL_MAPPING.get(raw_label, raw_label)
        if canonical not in TARGET_CLASSES:
            # Fallback tra cứu từ khóa
            low = canonical.lower()
            if "cam" in low:
                canonical = "Bien_Cam"
            elif "nguy_hiem" in low or "canh_bao" in low or "warning" in low:
                canonical = "Bien_Nguy_Hiem"
            elif "hieu_lenh" in low or "mandatory" in low:
                canonical = "Bien_Hieu_Lenh"
            elif "chi_dan" in low or "information" in low:
                canonical = "Bien_Chi_Dan"
            else:
                canonical = "Unknown"

        class_id = CLASS_TO_IDX.get(canonical, -1)
        return {
            "class_name": canonical,
            "class_id": class_id,
            "confidence": confidence,
            "bbox": None,
        }
