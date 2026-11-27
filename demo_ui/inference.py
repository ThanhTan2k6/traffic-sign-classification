"""Image decoding, Two-Stage YOLO + Classifier inference, annotation, and export."""

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.config import IDX_TO_CLASS, TARGET_CLASSES
from demo_ui.constants import CLASS_METADATA


def decode_image_bytes(raw_bytes: bytes) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Decode raw image bytes safely:
    - Applies PIL ImageOps.exif_transpose for rotated smartphone JPEGs.
    - Flattens PNG alpha channels over a clean white background.
    - Returns (img_bgr, info_dict) at full original resolution.
    """
    info: Dict[str, Any] = {
        "width": 0,
        "height": 0,
        "orig_mode": "",
        "exif_orientation": 1,
        "exif_rotated": False,
        "had_alpha": False,
    }
    if not raw_bytes:
        return None, info

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        info["orig_mode"] = pil_img.mode
        exif = pil_img.getexif()
        orientation = int(exif.get(274, 1)) if exif else 1
        info["exif_orientation"] = orientation
        info["exif_rotated"] = orientation != 1

        pil_img = ImageOps.exif_transpose(pil_img)

        had_alpha = (
            pil_img.mode in ("RGBA", "LA", "PA")
            or (pil_img.mode == "P" and "transparency" in pil_img.info)
        )
        info["had_alpha"] = had_alpha
        if had_alpha:
            rgba = pil_img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            rgb_img = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            rgb_img = pil_img.convert("RGB")

        rgb_arr = np.asarray(rgb_img, dtype=np.uint8)
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        ih, iw = bgr_arr.shape[:2]
        info["width"] = int(iw)
        info["height"] = int(ih)
        return bgr_arr, info
    except Exception:
        return None, info


def load_image_path(path: Path) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Read image file from disk through the unified EXIF/alpha-safe decoder."""
    if not path.exists():
        return None, {}
    return decode_image_bytes(path.read_bytes())


def resize_for_display(img_rgb: np.ndarray, max_edge: int = 1600) -> Tuple[np.ndarray, bool]:
    """Downscale image strictly for browser rendering if either dimension exceeds max_edge."""
    ih, iw = img_rgb.shape[:2]
    largest = max(ih, iw)
    if largest <= max_edge:
        return img_rgb, False
    scale = max_edge / float(largest)
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), True


def upscale_crop_for_display(crop_rgb: np.ndarray, target_min: int = 140) -> np.ndarray:
    """Upscale tiny ROI crops strictly for human preview; never pass this to inference."""
    ih, iw = crop_rgb.shape[:2]
    smallest = min(ih, iw)
    if smallest >= target_min or smallest <= 0:
        return crop_rgb
    scale = min(6.0, target_min / float(smallest))
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    return cv2.resize(crop_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def extract_probabilities(detector: Any, img_bgr: np.ndarray, model_type: str) -> Dict[str, float]:
    """
    Extract 4-class score distribution:
    - LeNet-5: Softmax on 4-class logits.
    - SVM: Softmax on 5-class OvR decision_function, summed into 4 canonical groups
      (accumulating both 'chỉ dẫn' and 'phụ' into Bien_Chi_Dan instead of overwriting).
    """
    probs_out = {cls: 0.0 for cls in TARGET_CLASSES}
    if detector is None or img_bgr is None or img_bgr.size == 0:
        return probs_out

    try:
        if model_type == "lenet":
            import torch

            prep = detector.preprocess(img_bgr)
            logits = detector.inference(prep)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            for i, p_val in enumerate(probs):
                cls_name = IDX_TO_CLASS.get(i)
                if cls_name in probs_out:
                    probs_out[cls_name] += float(p_val)
            return probs_out

        if model_type == "svm":
            from scipy.special import softmax

            prep = detector.preprocess(img_bgr)
            if prep is not None and hasattr(detector.model, "decision_function"):
                df = detector.model.decision_function(prep)
                raw_probs = softmax(df, axis=1)[0]
                classes = getattr(detector, "classes_", None)
                if classes is None and hasattr(detector.model, "classes_"):
                    classes = detector.model.classes_
                if classes is not None:
                    for raw_label, prob_val in zip(classes, raw_probs):
                        mapped = detector.LABEL_MAPPING.get(str(raw_label), str(raw_label))
                        if mapped in probs_out:
                            probs_out[mapped] += float(prob_val)
                    total = sum(probs_out.values())
                    if total > 0:
                        for k in probs_out:
                            probs_out[k] /= total
                    return probs_out
    except Exception:
        pass

    return probs_out


def classify_single_crop(
    crop_bgr: np.ndarray,
    det_lenet: Any,
    det_svm: Any,
    model_key: str,
    include_both: bool = False,
) -> Dict[str, Any]:
    """Classify one BGR crop with the chosen model (and optionally both for comparison)."""
    out: Dict[str, Any] = {}
    need_lenet = det_lenet is not None and (model_key in ("lenet", "compare") or include_both)
    need_svm = det_svm is not None and (model_key in ("svm", "compare") or include_both)

    if need_lenet:
        r_lenet = det_lenet.predict(crop_bgr)
        out["lenet"] = {
            "class_name": r_lenet.get("class_name", "Unknown"),
            "confidence": float(r_lenet.get("confidence", 0.0)),
            "latency_ms": float(r_lenet.get("inference_time", 0.0)) * 1000.0,
            "probs": extract_probabilities(det_lenet, crop_bgr, "lenet"),
        }

    if need_svm:
        r_svm = det_svm.predict(crop_bgr)
        out["svm"] = {
            "class_name": r_svm.get("class_name", "Unknown"),
            "confidence": float(r_svm.get("confidence", 0.0)),
            "latency_ms": float(r_svm.get("inference_time", 0.0)) * 1000.0,
            "probs": extract_probabilities(det_svm, crop_bgr, "svm"),
        }

    if model_key == "svm" and "svm" in out:
        primary = out["svm"]
    elif "lenet" in out:
        primary = out["lenet"]
    elif "svm" in out:
        primary = out["svm"]
    else:
        primary = {
            "class_name": "Unknown",
            "confidence": 0.0,
            "latency_ms": 0.0,
            "probs": {cls: 0.0 for cls in TARGET_CLASSES},
        }

    out["primary_class"] = primary["class_name"]
    out["primary_conf"] = primary["confidence"]
    out["primary_latency_ms"] = primary["latency_ms"]
    out["primary_probs"] = primary["probs"]
    return out


def run_scene_inference(
    img_bgr: np.ndarray,
    yolo_model: Any,
    det_lenet: Any,
    det_svm: Any,
    model_key: str = "lenet",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.60,
) -> Dict[str, Any]:
    """
    Two-Stage inference preserving exact CLI logic:
    YOLO11s (imgsz=800) -> exact raw crop (no padding) -> Classifier(s).
    """
    import time

    ih, iw = img_bgr.shape[:2]
    t0 = time.perf_counter()
    results = yolo_model.predict(
        source=img_bgr,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
        imgsz=800,
    )
    yolo_ms = (time.perf_counter() - t0) * 1000.0

    raw_boxes: List[Tuple[int, int, int, int, float]] = []
    for r in results:
        if r.boxes is not None:
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                x1 = max(0, min(iw - 1, int(xyxy[0])))
                y1 = max(0, min(ih - 1, int(xyxy[1])))
                x2 = max(x1 + 2, min(iw, int(xyxy[2])))
                y2 = max(y1 + 2, min(ih, int(xyxy[3])))
                yolo_conf = float(b.conf[0].cpu().numpy())
                raw_boxes.append((x1, y1, x2, y2, yolo_conf))

    detections: List[Dict[str, Any]] = []
    cls_total_ms = 0.0
    for idx, (x1, y1, x2, y2, y_conf) in enumerate(raw_boxes, start=1):
        crop_bgr = img_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        t_c0 = time.perf_counter()
        cls_res = classify_single_crop(
            crop_bgr=crop_bgr,
            det_lenet=det_lenet,
            det_svm=det_svm,
            model_key=model_key,
            include_both=(model_key == "compare"),
        )
        cls_total_ms += (time.perf_counter() - t_c0) * 1000.0

        detections.append(
            {
                "index": idx,
                "bbox": (x1, y1, x2, y2),
                "width": x2 - x1,
                "height": y2 - y1,
                "yolo_conf": y_conf,
                "crop_bgr": crop_bgr,
                **cls_res,
            }
        )

    total_ms = yolo_ms + cls_total_ms
    fps = 1000.0 / max(1.0, total_ms)
    return {
        "detections": detections,
        "yolo_ms": yolo_ms,
        "cls_total_ms": cls_total_ms,
        "total_ms": total_ms,
        "fps": fps,
    }


def draw_scene_boxes(img_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draw color-coded bounding boxes and crisp ASCII labels (#1 P.* 100%)
    scaled proportionally to image dimensions.
    """
    vis = img_bgr.copy()
    ih, iw = vis.shape[:2]
    min_dim = max(1, min(ih, iw))
    box_thick = max(2, int(round(min_dim / 300.0)))
    font_scale = max(0.45, min(1.25, min_dim / 950.0))
    font_thick = max(1, int(round(box_thick * 0.7)))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cname = det["primary_class"]
        conf = det["primary_conf"]
        meta = CLASS_METADATA.get(cname, CLASS_METADATA["Unknown"])
        bgr_col = meta["color_bgr"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), bgr_col, box_thick)

        tag_text = f"#{det['index']} {meta['code_prefix']} {conf * 100:.0f}%"
        (tw, th), baseline = cv2.getTextSize(
            tag_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick
        )
        pad_x = max(4, int(round(4 * font_scale)))
        pad_y = max(4, int(round(4 * font_scale)))
        box_w = tw + pad_x * 2
        box_h = th + baseline + pad_y * 2

        tx1 = max(0, min(iw - box_w, x1))
        if y1 - box_h >= 0:
            ty1 = y1 - box_h
            ty2 = y1
        else:
            ty1 = min(ih - box_h, y2)
            ty2 = min(ih, ty1 + box_h)

        cv2.rectangle(vis, (tx1, ty1), (tx1 + box_w, ty2), bgr_col, -1)
        cv2.putText(
            vis,
            tag_text,
            (tx1 + pad_x, ty2 - baseline - pad_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    return vis


def build_detections_dataframe(detections: List[Dict[str, Any]], model_key: str) -> pd.DataFrame:
    """Build structured results table with explicit separate YOLO and classification columns."""
    rows = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        meta = CLASS_METADATA.get(det["primary_class"], CLASS_METADATA["Unknown"])
        row: Dict[str, Any] = {
            "#": f"#{det['index']}",
            "Tọa độ [x1, y1, x2, y2]": f"[{x1}, {y1}, {x2}, {y2}]",
            "Kích thước (px)": f"{det['width']} × {det['height']}",
            "Độ tin cậy YOLO (%)": round(det["yolo_conf"] * 100.0, 1),
        }
        if model_key == "compare":
            l_res = det.get("lenet", {})
            s_res = det.get("svm", {})
            l_meta = CLASS_METADATA.get(l_res.get("class_name", "Unknown"), CLASS_METADATA["Unknown"])
            s_meta = CLASS_METADATA.get(s_res.get("class_name", "Unknown"), CLASS_METADATA["Unknown"])
            row["LeNet-5 (Nhóm)"] = f"{l_meta['vn_name']} ({l_meta['code_prefix']})"
            row["LeNet-5 Tin cậy (%)"] = round(l_res.get("confidence", 0.0) * 100.0, 1)
            row["SVM (Nhóm)"] = f"{s_meta['vn_name']} ({s_meta['code_prefix']})"
            row["SVM Tin cậy (%)"] = round(s_res.get("confidence", 0.0) * 100.0, 1)
        else:
            row["Nhóm dự đoán"] = f"{meta['vn_name']} ({meta['code_prefix']})"
            row["Độ tin cậy phân loại (%)"] = round(det["primary_conf"] * 100.0, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def to_png_bytes(img_bgr: np.ndarray) -> bytes:
    """Encode BGR numpy image into PNG bytes."""
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes() if ok else b""


def to_csv_bytes(detections: List[Dict[str, Any]], model_key: str) -> bytes:
    """Export detections table to UTF-8-SIG CSV bytes for Excel compatibility."""
    df = build_detections_dataframe(detections, model_key)
    return df.to_csv(index=False).encode("utf-8-sig")


def to_crops_zip_bytes(detections: List[Dict[str, Any]]) -> bytes:
    """Bundle raw exact crops (without padding or resize) into a ZIP archive."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for det in detections:
            cname = det["primary_class"]
            fname = f"sign_{det['index']:02d}_{cname}_{det['width']}x{det['height']}.png"
            png_data = to_png_bytes(det["crop_bgr"])
            if png_data:
                zf.writestr(fname, png_data)
    return bio.getvalue()
