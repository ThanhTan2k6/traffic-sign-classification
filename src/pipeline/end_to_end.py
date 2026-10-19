"""
Pipeline Tổng thể Tích hợp (End-to-End Traffic Sign Recognition) - Thành viên 3 phụ trách.
Luồng xử lý hoàn chỉnh 4 Giai đoạn:
[Ảnh đầu vào RGB/BGR]
        │
        ▼
[Giai đoạn 1: Lọc màu HSV (ColorFilter)]
        │
        ▼
[Giai đoạn 2: Phân đoạn, Khử nhiễu Morphology & Cắt ROI (ROIExtractor)]
        │
        ▼
[Giai đoạn 3: Chuẩn hóa 64x64 & Trích xuất HOG (HOGFeatureExtractor)]
        │
        ▼
[Giai đoạn 4: Phân loại bằng Mô hình SVM (Joblib Loaded Model)]
        │
        ▼
[Kết quả: Bounding Box + Tên biển báo + Độ tin cậy (Confidence)]
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import joblib
import numpy as np

from src.config import MODEL_PATH, LABEL_ENCODER_PATH
from src.detection.color_filter import ColorFilter
from src.detection.roi_extractor import ROIExtractor
from src.features.hog_extractor import HOGFeatureExtractor


class TrafficSignPipeline:
    """Hệ thống tích hợp toàn bộ các giai đoạn nhận diện biển báo."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        encoder_path: Path = LABEL_ENCODER_PATH,
    ):
        self.color_filter = ColorFilter()
        self.roi_extractor = ROIExtractor()
        self.hog_extractor = HOGFeatureExtractor()

        self.model = None
        self.label_encoder = None

        if Path(model_path).exists() and Path(encoder_path).exists():
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            print("[+] Pipeline: Đã nạp mô hình và encoder thành công!")
        else:
            print("[!] Pipeline: Chưa nạp mô hình SVM (hãy chạy bước train của TV2 trước).")

    def predict(self, bgr_image: np.ndarray) -> Dict[str, Any]:
        """
        Nhận ảnh đầu vào BGR và trả về kết quả phát hiện, phân loại kèm các bước trung gian.
        """
        # Giai đoạn 1: Lọc màu
        mask, masked_img, individual_masks = self.color_filter.create_traffic_sign_mask(bgr_image)

        # Giai đoạn 2: Trích xuất ROI
        rois = self.roi_extractor.extract_rois(bgr_image, mask)

        annotated_image = bgr_image.copy()
        predictions = []

        for item in rois:
            roi_norm = item["roi_normalized"]
            x, y, w, h = item["bbox"]

            predicted_label = "Unknown"
            confidence = 0.0

            # Giai đoạn 3 & 4: HOG & SVM Prediction (nếu model sẵn sàng)
            if self.model is not None and self.label_encoder is not None:
                features, _ = self.hog_extractor.extract(roi_norm)
                features = features.reshape(1, -1)

                pred_class_idx = self.model.predict(features)[0]
                predicted_label = self.label_encoder.inverse_transform([pred_class_idx])[0]

                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(features)[0]
                    confidence = float(np.max(probs))

            predictions.append({
                "bbox": (x, y, w, h),
                "label": predicted_label,
                "confidence": confidence,
                "roi_normalized": roi_norm,
                "area": item["area"],
            })

            # Vẽ lên ảnh annotated
            text = f"{predicted_label} ({confidence*100:.1f}%)" if confidence > 0 else predicted_label
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated_image,
                text,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return {
            "annotated_image": annotated_image,
            "mask": mask,
            "masked_image": masked_img,
            "individual_masks": individual_masks,
            "predictions": predictions,
        }
