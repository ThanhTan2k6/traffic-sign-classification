"""
================================================================================
ĐỐI CHUẨN HIỆU NĂNG TOÀN TRÌNH TRÊN ẢNH TOÀN CẢNH ĐƯỜNG PHỐ (END-TO-END SCENE BENCHMARK)
================================================================================
Kiến trúc 2 giai đoạn (Two-Stage Pipeline):
1. Định vị (Localization): YOLO11s 1-class detector (runs/detect/vn_s/weights/best.pt)
2. Phân loại (Classification):
   - Phương pháp 1: Color + HOG + SVM RBF (Thành viên 1)
   - Phương pháp 2: Enhanced LeNet-5 CNN (Thành viên 2)
================================================================================
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from src.config import (
    PROJECT_ROOT,
    RESULTS_DIR,
    TARGET_CLASSES,
    CLASS_TO_IDX,
    YOLO_TEST_FILE,
    YOLO_VAL_FILE,
)
from src.detection.base_detector import BaseDetector
from src.detection.color_hog_svm import ColorHogSvmDetector
from src.detection.deep_learning import DeepLearningDetector
from src.utils import imread_unicode, ensure_dir


def box_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Tính IoU giữa 2 bounding box (x1, y1, x2, y2)."""
    xa = max(b1[0], b2[0])
    ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2])
    yb = min(b1[3], b2[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def load_yolo_detector(weights_path: Path = None):
    """Nạp YOLO11s detector để phát hiện biển báo trên ảnh toàn cảnh."""
    from ultralytics import YOLO

    if weights_path is None or not weights_path.exists():
        cand_paths = [
            PROJECT_ROOT / "checkpoints" / "yolo" / "best.pt",
            PROJECT_ROOT / "runs" / "detect" / "vn_s" / "weights" / "best.pt",
            PROJECT_ROOT / "checkpoints" / "deep_learning" / "best_yolo.pt",
        ]
        for p in cand_paths:
            if p.exists():
                weights_path = p
                break

    if weights_path is None or not weights_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy checkpoint YOLO detector tại checkpoints/yolo/best.pt hoặc runs/detect/vn_s/weights/best.pt. "
            "Vui lòng tải weights theo hướng dẫn trong docs/WEIGHTS.md"
        )
    return YOLO(str(weights_path))


def evaluate_detector_on_scenes(
    detector: BaseDetector,
    scene_split_file: Path,
    yolo_model=None,
    max_samples: int = None,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.5,
) -> Dict[str, Any]:
    """
    Đánh giá hệ thống Two-Stage (YOLO Detector + Classifier) trên ảnh toàn cảnh.
    """
    if not scene_split_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file split toàn cảnh: {scene_split_file}")

    if yolo_model is None:
        yolo_model = load_yolo_detector()

    with open(scene_split_file, "r", encoding="utf-8") as f:
        img_lines = [l.strip() for l in f if l.strip()]

    if max_samples:
        img_lines = img_lines[:max_samples]

    total_gt = 0
    total_det = 0
    tp_localization = 0
    tp_classification = 0
    latencies = []

    print(f"\n[*] Đang đánh giá toàn trình [{detector.model_name}] trên {len(img_lines)} ảnh toàn cảnh...")

    for img_rel in tqdm(img_lines, desc=f"Scene Eval [{detector.model_name}]", leave=False):
        img_path = PROJECT_ROOT / img_rel
        if not img_path.exists():
            img_path = Path(img_rel)
        if not img_path.exists():
            continue

        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not label_path.exists():
            label_path = PROJECT_ROOT / "dataset" / "labels" / f"{img_path.stem}.txt"

        img = imread_unicode(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]

        # Đọc Ground Truth từ file nhãn
        gt_boxes = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8", errors="ignore") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(float(parts[0]))
                        if cid < len(TARGET_CLASSES):
                            xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                            gx1 = (xc - bw / 2.0) * iw
                            gy1 = (yc - bh / 2.0) * ih
                            gx2 = (xc + bw / 2.0) * iw
                            gy2 = (yc + bh / 2.0) * ih
                            gt_boxes.append((gx1, gy1, gx2, gy2, TARGET_CLASSES[cid]))

        total_gt += len(gt_boxes)

        # 1. Định vị ứng viên qua YOLO11s
        t0 = time.perf_counter()
        results = yolo_model.predict(
            source=img,
            conf=conf_thresh,
            iou=0.6,
            verbose=False,
            imgsz=800,
        )

        pred_boxes = []
        for r in results:
            if r.boxes is not None:
                for b in r.boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    px1 = max(0, min(iw - 1, int(xyxy[0])))
                    py1 = max(0, min(ih - 1, int(xyxy[1])))
                    px2 = max(px1 + 2, min(iw, int(xyxy[2])))
                    py2 = max(py1 + 2, min(ih, int(xyxy[3])))
                    conf = float(b.conf[0].cpu().numpy())
                    pred_boxes.append((px1, py1, px2, py2, conf))

        total_det += len(pred_boxes)

        # 2. Phân loại từng candidate crop bằng mô hình
        classified_candidates = []
        for (px1, py1, px2, py2, pconf) in pred_boxes:
            crop = img[py1:py2, px1:px2]
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            pred_res = detector.predict(crop)
            pred_cls = pred_res.get("class_name", "Unknown")
            classified_candidates.append((px1, py1, px2, py2, pred_cls))

        latencies.append(time.perf_counter() - t0)

        # 3. Ghép cặp đánh giá IoU
        matched_gt = set()
        for (px1, py1, px2, py2, pred_cls) in classified_candidates:
            best_iou = 0.0
            best_idx = -1
            for g_idx, (gx1, gy1, gx2, gy2, gt_cls) in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = box_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx

            if best_iou >= iou_thresh and best_idx != -1:
                matched_gt.add(best_idx)
                tp_localization += 1
                if pred_cls == gt_boxes[best_idx][4]:
                    tp_classification += 1

    # Tính toán các chỉ số
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    fps = (1.0 / avg_latency) if avg_latency > 0 else 0.0

    loc_rate = (tp_localization / total_gt * 100.0) if total_gt > 0 else 0.0
    cls_acc_on_loc = (tp_classification / tp_localization * 100.0) if tp_localization > 0 else 0.0

    precision = (tp_classification / total_det * 100.0) if total_det > 0 else 0.0
    recall = (tp_classification / total_gt * 100.0) if total_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "model_name": detector.model_name,
        "total_images": len(img_lines),
        "total_gt": total_gt,
        "total_det": total_det,
        "localized_gt": tp_localization,
        "localization_rate": round(loc_rate, 2),
        "cls_acc_on_loc": round(cls_acc_on_loc, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "latency_ms": round(avg_latency * 1000.0, 2),
        "fps": round(fps, 1),
    }


def plot_scene_benchmark_comparison(df_results: pd.DataFrame, output_path: Path):
    """Vẽ biểu đồ đối chuẩn toàn cảnh 4 bảng nhỏ."""
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    models = df_results["Phương Pháp / Pipeline"].tolist()
    colors = ["#2ecc71", "#3498db", "#e74c3c"][: len(models)]

    # 1. Độ chính xác phân loại khi định vị được
    bars1 = ax1.bar(models, df_results["Đoán Đúng Khi Tìm Thấy (%)"], color=colors, edgecolor="black", alpha=0.85)
    for b in bars1:
        y = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2.0, y + 1, f"{y:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Độ Chính Xác Phân Loại Khi Định Vị Được Biển", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Độ chính xác (%)")
    ax1.set_ylim(0, 115)

    # 2. Tỷ lệ Recall toàn trình (End-to-End Recall)
    bars2 = ax2.bar(models, df_results["E2E Recall (%)"], color=colors, edgecolor="black", alpha=0.85)
    for b in bars2:
        y = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, y + 1, f"{y:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax2.set_title("Tỷ Lệ Bắt Trúng Toàn Trình (End-to-End Recall)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Recall (%)")
    ax2.set_ylim(0, 115)

    # 3. Tốc độ xử lý (FPS)
    bars3 = ax3.bar(models, df_results["Tốc Độ (FPS)"], color=colors, edgecolor="black", alpha=0.85)
    for b in bars3:
        y = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2.0, y + 0.5, f"{y:.1f} FPS", ha="center", va="bottom", fontweight="bold")
    ax3.set_title("Tốc Độ Xử Lý Trên Ảnh Toàn Cảnh (FPS)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Khung hình / giây (FPS)")
    ax3.set_ylim(0, max(df_results["Tốc Độ (FPS)"]) * 1.3)

    # 4. Độ trễ toàn trình (Latency ms)
    bars4 = ax4.bar(models, df_results["Độ Trễ (ms)"], color=colors, edgecolor="black", alpha=0.85)
    for b in bars4:
        y = b.get_height()
        ax4.text(b.get_x() + b.get_width() / 2.0, y + 1, f"{y:.1f} ms", ha="center", va="bottom", fontweight="bold")
    ax4.set_title("Độ Trễ Xử Lý Toàn Trình (Latency ms)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Mili-giây (ms)")
    ax4.set_ylim(0, max(df_results["Độ Trễ (ms)"]) * 1.3)

    plt.suptitle("ĐỐI CHUẨN HIỆU NĂNG TOÀN TRÌNH: YOLO11s DETECTOR + PHÂN LOẠI (END-TO-END SCENE BENCHMARK)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ đối chuẩn toàn cảnh tại: {output_path}")


def run_scene_benchmark(split_name: str = "test", max_samples: int = None) -> pd.DataFrame:
    """Chạy đánh giá đối sánh toàn cảnh tất cả các mô hình."""
    split_file = YOLO_TEST_FILE if split_name.lower() == "test" else YOLO_VAL_FILE

    print("=" * 75)
    print("🏆 BẮT ĐẦU CHẠY BENCHMARK TRÊN ẢNH ĐƯỜNG PHỐ TOÀN CẢNH (END-TO-END)")
    print(f"[*] Tập dữ liệu: {split_file}")
    print("=" * 75)

    yolo_model = load_yolo_detector()
    detectors = [
        ColorHogSvmDetector(),
        DeepLearningDetector(),
    ]

    results = []
    for det in detectors:
        try:
            m = evaluate_detector_on_scenes(
                detector=det,
                scene_split_file=split_file,
                yolo_model=yolo_model,
                max_samples=max_samples,
            )
            name_label = (
                "Two-Stage SVM (Thành viên 1)"
                if det.model_name == "color_hog_svm"
                else "Two-Stage LeNet (Thành viên 2)"
            )
            results.append({
                "Phương Pháp / Pipeline": name_label,
                "Số Ảnh": m["total_images"],
                "Tổng Biển (GT)": m["total_gt"],
                "Định Vị Được (GT)": m["localized_gt"],
                "Tỷ Lệ Định Vị (%)": m["localization_rate"],
                "Đoán Đúng Khi Tìm Thấy (%)": m["cls_acc_on_loc"],
                "E2E Precision (%)": m["precision"],
                "E2E Recall (%)": m["recall"],
                "E2E F1 (%)": m["f1_score"],
                "Độ Trễ (ms)": m["latency_ms"],
                "Tốc Độ (FPS)": m["fps"],
            })
        except Exception as e:
            print(f"[-] Lỗi khi benchmark toàn cảnh {det.model_name}: {e}")

    df_results = pd.DataFrame(results)
    if df_results.empty:
        raise RuntimeError("[-] Không có kết quả đánh giá nào được sinh ra từ các detector.")

    print("\n" + "=" * 75)
    print("📊 BẢNG XẾP HẠNG TOÀN CẢNH ĐƯỜNG PHỐ (END-TO-END SCENE LEADERBOARD):")
    print("=" * 75)
    print(tabulate(df_results, headers="keys", tablefmt="fancy_grid", showindex=True))
    print("=" * 75)

    ensure_dir(RESULTS_DIR)
    out_chart = RESULTS_DIR / "scene_benchmark_comparison.png"
    plot_scene_benchmark_comparison(df_results, out_chart)

    return df_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="End-to-End Scene Evaluation Pipeline")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Tập split (test hoặc val)")
    parser.add_argument("--samples", type=int, default=None, help="Giới hạn số ảnh toàn cảnh đánh giá")

    args = parser.parse_args()
    run_scene_benchmark(split_name=args.split, max_samples=args.samples)


if __name__ == "__main__":
    main()
