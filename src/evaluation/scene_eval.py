"""
================================================================================
🌍 BỘ ĐÁNH GIÁ TOÀN CẢNH ĐƯỜNG PHỐ (END-TO-END SCENE BENCHMARK)
Đánh giá năng lực phát hiện và phân loại biển báo trên ảnh chụp toàn cảnh
chưa qua cắt (Full Street Scenes) từ tập Blind Test (yolo_test.txt).
================================================================================
Quy trình đánh giá Two-Stage:
1. Đọc ảnh chụp toàn cảnh đường phố (960x540 / 1920x1080) và file nhãn ground-truth.
2. Giai đoạn 1 (Detection): Định vị các ứng viên biển báo (Candidate Proposals) qua HSV + Morphology.
3. Giai đoạn 2 (Classification): Phân loại từng vùng ứng viên bằng mô hình (HOG+SVM hoặc Enhanced LeNet-5).
4. Khớp IoU giữa dự đoán và Ground Truth:
   - Localization Rate: Tỷ lệ biển báo thực tế được tìm thấy (IoU >= 0.4).
   - Classification on Localized: Độ chính xác phân loại trên các biển tìm thấy.
   - End-to-End Precision, Recall, F1: Độ đo toàn trình từ phát hiện đến phân loại.
   - Tốc độ xử lý (FPS) và Độ trễ (ms) trên ảnh toàn cảnh.
================================================================================
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate
from tqdm import tqdm

# Windows UTF-8 stdout fix
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src.config import (
    PROJECT_ROOT,
    RESULTS_DIR,
    TARGET_CLASSES,
    YOLO_TEST_FILE,
    YOLO_VAL_FILE,
    load_52_classes,
    get_target_category,
)
from src.utils import ensure_dir
from src.detection.base_detector import BaseDetector
from src.detection.color_hog_svm import ColorHogSvmDetector
from src.detection.deep_learning import DeepLearningDetector
from src.utils.image_io import imread_unicode


def get_scene_proposals(image: np.ndarray, max_proposals: int = 40) -> List[Tuple[int, int, int, int]]:
    """
    Sinh các bounding box ứng viên (Candidate Proposals) từ ảnh chụp đường phố
    bằng bộ lọc không gian màu HSV và phép toán hình thái học (Morphology).
    """
    h_img, w_img = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Mặt nạ dải màu biển báo giao thông Việt Nam
    # Màu đỏ: Viền biển cấm, biển nguy hiểm
    m_red1 = cv2.inRange(hsv, (0, 60, 40), (10, 255, 255))
    m_red2 = cv2.inRange(hsv, (170, 60, 40), (180, 255, 255))
    # Màu xanh lam: Nền biển hiệu lệnh, chỉ dẫn
    m_blue = cv2.inRange(hsv, (95, 50, 40), (135, 255, 255))
    # Màu vàng: Nền biển nguy hiểm
    m_yellow = cv2.inRange(hsv, (12, 60, 40), (38, 255, 255))

    mask = m_red1 | m_red2 | m_blue | m_yellow

    # 2. Morphology đóng viền và kết nối các vùng đứt đoạn
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals = []

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        aspect_ratio = bw / max(1.0, bh)

        # Lọc các bounding box thỏa mãn kích thước biển báo
        if 12 <= bw <= 550 and 12 <= bh <= 550 and 0.35 <= aspect_ratio <= 2.8:
            # Thêm padding nhẹ 2px để bao trọn viền biển
            px1 = max(0, x - 2)
            py1 = max(0, y - 2)
            px2 = min(w_img, x + bw + 2)
            py2 = min(h_img, y + bh + 2)
            proposals.append((px1, py1, px2, py2, area))

    # Sắp xếp ưu tiên các vùng có diện tích hợp lý
    proposals.sort(key=lambda item: item[4], reverse=True)
    return [(b[0], b[1], b[2], b[3]) for b in proposals[:max_proposals]]


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Tính chỉ số Intersection over Union (IoU) giữa 2 bounding box."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_bipartite_max_cardinality(
    gt_boxes: List[Dict[str, Any]],
    candidate_boxes: List[Tuple[int, int, int, int]],
    iou_thresh: float = 0.4,
) -> List[Tuple[int, int]]:
    """
    Tìm ghép cặp 1-1 cực đại (Maximum Cardinality Maximum Weight Bipartite Matching)
    bằng thuật toán Hungarian (linear_sum_assignment).
    Tối đa hóa số lượng cặp thỏa mãn IoU >= iou_thresh trước, sau đó tối đa hóa tổng IoU.
    """
    n_gt = len(gt_boxes)
    n_cand = len(candidate_boxes)
    if n_gt == 0 or n_cand == 0:
        return []

    cost_matrix = np.zeros((n_gt, n_cand), dtype=np.float64)
    has_valid_edge = False

    for i in range(n_gt):
        for j in range(n_cand):
            iou = compute_iou(gt_boxes[i]["bbox"], candidate_boxes[j])
            if iou >= iou_thresh:
                cost_matrix[i, j] = -(1000.0 + iou)
                has_valid_edge = True

    if not has_valid_edge:
        return []

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 0:
            matches.append((r, c))
    return matches


def match_e2e_max_cardinality(
    gt_boxes: List[Dict[str, Any]],
    predicted_boxes: List[Dict[str, Any]],
    iou_thresh: float = 0.4,
) -> List[Tuple[int, int]]:
    """
    Ghép cặp 1-1 toàn trình nhận thức lớp (Class-aware Bipartite Matching)
    bằng thuật toán Hungarian, tối đa hóa số lượng dự đoán đúng lớp trước,
    sau đó tối đa hóa tổng IoU.
    """
    n_gt = len(gt_boxes)
    n_pred = len(predicted_boxes)
    if n_gt == 0 or n_pred == 0:
        return []

    cost_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    has_valid_edge = False

    for i in range(n_gt):
        for j in range(n_pred):
            if predicted_boxes[j]["class_name"] == gt_boxes[i]["category"]:
                iou = compute_iou(gt_boxes[i]["bbox"], predicted_boxes[j]["bbox"])
                if iou >= iou_thresh:
                    cost_matrix[i, j] = -(1000.0 + iou)
                    has_valid_edge = True

    if not has_valid_edge:
        return []

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 0:
            matches.append((r, c))
    return matches


def evaluate_detector_on_scenes(
    detector: BaseDetector,
    scene_split_file: Path,
    iou_thresh: float = 0.4,
    conf_thresh: float = 0.5,
    max_samples: int = None,
) -> Dict[str, Any]:
    """
    Đánh giá mô hình theo mô hình Two-Stage trên tập ảnh đường phố toàn cảnh.
    """
    if not scene_split_file.exists():
        raise FileNotFoundError(f"[-] Không tìm thấy file split toàn cảnh: {scene_split_file}")

    classes_52 = load_52_classes()
    if not classes_52:
        raise FileNotFoundError("[-] Không đọc được danh sách 52 lớp (classes.txt trống hoặc không tồn tại).")
    with open(scene_split_file, "r", encoding="utf-8") as f:
        rel_paths = [Path(line.strip()) for line in f if line.strip()]

    if max_samples:
        rel_paths = rel_paths[:max_samples]

    total_gt = 0
    localized_gt = 0
    correct_e2e = 0
    false_positives = 0
    total_images = len(rel_paths)
    inference_times = []

    if getattr(detector, "model", None) is None:
        raise RuntimeError(f"Mô hình [{detector.model_name}] chưa nạp checkpoint, không thể benchmark toàn cảnh.")

    inference_errors = 0
    successful_predictions = 0
    first_inference_error = None
    skipped_images = 0
    missing_labels = 0

    print(f"\n[*] Đang đánh giá toàn cảnh mô hình: [{detector.model_name}] trên {total_images} ảnh...")

    for rel_p in tqdm(rel_paths, desc=f"Scene Eval [{detector.model_name}]"):
        img_path = PROJECT_ROOT / rel_p
        if not img_path.exists():
            skipped_images += 1
            continue

        img = imread_unicode(img_path)
        if img is None:
            skipped_images += 1
            continue
        h_img, w_img = img.shape[:2]

        # 1. Đọc nhãn Ground-Truth từ file YOLO tương ứng
        lbl_path = PROJECT_ROOT / "dataset" / "labels" / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            missing_labels += 1
            skipped_images += 1
            continue

        gt_boxes = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                if cid >= len(classes_52):
                    continue
                cat = get_target_category(classes_52[cid])
                if cat is None:
                    continue

                xc, yc, bw, bh = map(float, parts[1:5])
                gx1 = int((xc - bw / 2.0) * w_img)
                gy1 = int((yc - bh / 2.0) * h_img)
                gx2 = int((xc + bw / 2.0) * w_img)
                gy2 = int((yc + bh / 2.0) * h_img)

                # Lọc biển kích thước tối thiểu >= 10px
                if (gx2 - gx1) >= 10 and (gy2 - gy1) >= 10:
                    gt_boxes.append({"bbox": (gx1, gy1, gx2, gy2), "category": cat, "matched_loc": False})

        total_gt += len(gt_boxes)

        # 2. Đo đạc thời gian suy luận toàn trình (Proposal + Classification)
        t_start = time.perf_counter()
        proposals = get_scene_proposals(img)

        predicted_boxes = []
        for proposal_idx, pbox in enumerate(proposals):
            px1, py1, px2, py2 = pbox
            patch = img[py1:py2, px1:px2]
            if patch.size == 0:
                continue

            # Phân loại ROI bằng mô hình
            try:
                res = detector.predict(patch, is_roi=True)
                successful_predictions += 1
                conf = res.get("confidence", 0.0)
                pred_cls = res.get("class_name", "Unknown")
                if conf >= conf_thresh and pred_cls in TARGET_CLASSES:
                    predicted_boxes.append({
                        "proposal_idx": proposal_idx,
                        "bbox": pbox,
                        "class_name": pred_cls,
                        "confidence": conf,
                    })
            except Exception as e:
                inference_errors += 1
                if first_inference_error is None:
                    first_inference_error = e
                continue

        t_end = time.perf_counter()
        inference_times.append(t_end - t_start)

        # 3. Bước 1: Khớp định vị độc lập (Localization Matching - Class agnostic)
        # Tối đa hóa số lượng biển báo được tìm thấy (Maximum Cardinality) qua thuật toán Hungarian
        loc_matches = match_bipartite_max_cardinality(gt_boxes, proposals, iou_thresh=iou_thresh)
        matched_gts_loc = {r for r, c in loc_matches}

        # Bước 2: Khớp toàn trình nhận thức lớp (Class-aware End-to-End Matching)
        # Đánh giá độc lập với localization để không loại trừ dự đoán đúng lớp thỏa mãn IoU
        e2e_matches = match_e2e_max_cardinality(gt_boxes, predicted_boxes, iou_thresh=iou_thresh)
        matched_gts_e2e = {r for r, c in e2e_matches}
        matched_preds_e2e = {c for r, c in e2e_matches}
        correct_e2e += len(e2e_matches)

        # Hợp các GT được định vị từ cả 2 bước đảm bảo localized_gt >= correct_e2e
        all_localized_gts = matched_gts_loc | matched_gts_e2e
        for g_idx in all_localized_gts:
            gt_boxes[g_idx]["matched_loc"] = True
        localized_gt += len(all_localized_gts)

        # Đếm các dự đoán báo động giả (False Positives)
        false_positives += len(predicted_boxes) - len(matched_preds_e2e)

    if inference_errors > 0:
        print(f"[!] Cảnh báo: Có {inference_errors} ROI gặp lỗi suy luận. Lỗi đầu tiên: {first_inference_error}")

    if missing_labels > 0:
        print(f"[!] Cảnh báo: Có {missing_labels} ảnh không tìm thấy file nhãn txt.")

    if skipped_images > 0:
        print(f"[!] Cảnh báo: Bỏ qua {skipped_images}/{total_images} ảnh (không tồn tại hoặc không giải mã được).")

    evaluated_images = total_images - skipped_images
    if evaluated_images == 0:
        raise RuntimeError(f"[-] Không có ảnh hợp lệ nào để đánh giá trong: {scene_split_file}")
    if total_gt == 0:
        raise RuntimeError(
            f"[-] Không nạp được ground-truth nào (thiếu nhãn: {missing_labels}/{evaluated_images} ảnh). "
            "Kiểm tra đường dẫn dataset/labels."
        )
    if successful_predictions == 0 and inference_errors > 0:
        raise RuntimeError(
            f"[-] Không có ROI nào được phân loại thành công (lỗi suy luận: {inference_errors}). "
            f"Lỗi đầu tiên: {first_inference_error}"
        )

    # Tính toán các chỉ số học thuật
    loc_rate = (localized_gt / max(1, total_gt)) * 100.0
    cls_acc_on_loc = (correct_e2e / max(1, localized_gt)) * 100.0

    precision = (correct_e2e / max(1, (correct_e2e + false_positives))) * 100.0
    recall = (correct_e2e / max(1, total_gt)) * 100.0
    f1 = (2.0 * precision * recall / max(1e-6, (precision + recall)))

    avg_latency_ms = (float(np.mean(inference_times)) * 1000.0) if inference_times else 0.0
    fps = (1000.0 / avg_latency_ms) if avg_latency_ms > 0 else 0.0

    metrics = {
        "model_name": detector.model_name,
        "total_images": evaluated_images,
        "total_gt": total_gt,
        "localized_gt": localized_gt,
        "correct_e2e": correct_e2e,
        "false_positives": false_positives,
        "localization_rate": round(loc_rate, 2),
        "cls_acc_on_loc": round(cls_acc_on_loc, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 1),
    }
    return metrics


def plot_scene_benchmark_comparison(df_results: pd.DataFrame, output_path: Path):
    """Xuất biểu đồ đối sánh toàn cảnh độ phân giải cao 300 DPI."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    models = df_results["Phương Pháp / Pipeline"]
    colors = ["#2980b9", "#e67e22"]

    # 1. Tỷ lệ phân loại đúng trên biển đã định vị (Classification on Localized)
    bars1 = ax1.bar(models, df_results["Đoán Đúng Khi Tìm Thấy (%)"], color=colors, edgecolor="black", alpha=0.85)
    for b in bars1:
        y = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2.0, y + 1, f"{y:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Độ Chính Xác Phân Loại Khi Định Vị Được Biển", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Độ chính xác (%)")
    ax1.set_ylim(0, 110)

    # 2. Tỷ lệ Recall toàn trình (End-to-End Recall)
    bars2 = ax2.bar(models, df_results["E2E Recall (%)"], color=colors, edgecolor="black", alpha=0.85)
    for b in bars2:
        y = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2.0, y + 1, f"{y:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax2.set_title("Tỷ Lệ Bắt Trúng Toàn Trình (End-to-End Recall)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Recall (%)")
    ax2.set_ylim(0, 110)

    # 3. Tốc độ xử lý (FPS)
    bars3 = ax3.bar(models, df_results["Tốc Độ (FPS)"], color=colors, edgecolor="black", alpha=0.85)
    for b in bars3:
        y = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2.0, y + 1, f"{y:.1f} FPS", ha="center", va="bottom", fontweight="bold")
    ax3.set_title("Tốc Độ Xử Lý Trên Ảnh Toàn Cảnh (FPS)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Khung hình / giây (FPS)")
    ax3.set_ylim(0, max(df_results["Tốc Độ (FPS)"]) * 1.2)

    # 4. Độ trễ toàn trình (Latency ms)
    bars4 = ax4.bar(models, df_results["Độ Trễ (ms)"], color=colors, edgecolor="black", alpha=0.85)
    for b in bars4:
        y = b.get_height()
        ax4.text(b.get_x() + b.get_width() / 2.0, y + 1, f"{y:.1f} ms", ha="center", va="bottom", fontweight="bold")
    ax4.set_title("Độ Trễ Xử Lý Toàn Trình (Latency ms)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Mili-giây (ms)")
    ax4.set_ylim(0, max(df_results["Độ Trễ (ms)"]) * 1.25)

    plt.suptitle("ĐỐI CHUẨN HIỆU NĂNG TRÊN ẢNH ĐƯỜNG PHỐ TOÀN CẢNH (END-TO-END SCENE BENCHMARK)", fontsize=14, fontweight="bold")
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

    detectors = [
        DeepLearningDetector(),
        ColorHogSvmDetector(),
    ]

    results = []
    for det in detectors:
        try:
            m = evaluate_detector_on_scenes(
                detector=det,
                scene_split_file=split_file,
                max_samples=max_samples,
            )
            name_label = "Two-Stage LeNet (Thành viên 2)" if det.model_name == "deep_learning" else "Two-Stage SVM (Thành viên 1)"
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
    parser.add_argument("--samples", type=int, default=None, help="Giới hạn số ảnh toàn cảnh đánh giá (mặc định: tất cả)")

    args = parser.parse_args()
    run_scene_benchmark(split_name=args.split, max_samples=args.samples)


if __name__ == "__main__":
    main()
