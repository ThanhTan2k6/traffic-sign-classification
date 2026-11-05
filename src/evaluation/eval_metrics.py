"""
================================================================================
BỘ ĐÁNH GIÁ HIỆU NĂNG MÔ HÌNH (BENCHMARKING ENGINE)
================================================================================
Độc lập với mô hình (Model-agnostic), đo đạc các chỉ số:
- Accuracy, Precision, Recall, F1-Score (Macro & Per-class)
- Tốc độ suy luận: Latency (ms) và FPS
- Ma trận nhầm lẫn (Confusion Matrix)
================================================================================
"""

import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tabulate import tabulate
from tqdm import tqdm

from src.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    CROPPED_DATA_DIR,
    TARGET_CLASSES,
    CLASS_TO_IDX,
)
from src.detection.base_detector import BaseDetector
from src.utils import imread_unicode


def evaluate_detector(
    detector: BaseDetector,
    split_file: Path,
    max_samples: int = None,
) -> Dict[str, Any]:
    """
    Đánh giá một mô hình cụ thể trên tập dữ liệu chỉ định (Val hoặc Test).
    """
    if not split_file.exists():
        raise FileNotFoundError(f"[-] Không tìm thấy file split: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        raw_lines = [line.strip().split("\t") for line in f if line.strip()]

    if max_samples:
        raw_lines = raw_lines[:max_samples]

    y_true = []
    y_pred = []
    latencies = []

    print(f"\n[*] Đang đánh giá model: [{detector.model_name}] trên tập: {split_file.stem} ({len(raw_lines)} mẫu)...")

    for parts in tqdm(raw_lines, desc=f"Evaluating {detector.model_name}", leave=False):
        if len(parts) < 2:
            continue
        rel_path = parts[0]
        true_class = parts[1]

        # Kiểm tra đường dẫn ảnh trong data/cropped hoặc project root
        full_path = CROPPED_DATA_DIR / rel_path
        if not full_path.exists():
            full_path = RAW_DATA_DIR / rel_path
        if not full_path.exists():
            full_path = PROJECT_ROOT / rel_path

        img = imread_unicode(full_path)
        if img is None:
            continue

        try:
            res = detector.predict(img)
            pred_class = res.get("class_name", "Unknown")
            latencies.append(res.get("inference_time", 0.0))

            y_true.append(true_class)
            y_pred.append(pred_class)
        except NotImplementedError:
            # Mô hình skeleton chưa hoàn thiện code bởi thành viên
            y_true.append(true_class)
            y_pred.append("NotImplemented")
        except Exception as e:
            y_true.append(true_class)
            y_pred.append("Error")

    if not y_true or not latencies:
        print("[-] Không có mẫu nào được dự đoán thành công!")
        return {}

    # Tính toán các chỉ số
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    fps = (1.0 / avg_latency) if avg_latency > 0 else 0.0

    cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)

    metrics = {
        "model_name": detector.model_name,
        "split": split_file.stem,
        "samples": len(y_true),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "latency_ms": avg_latency * 1000.0,
        "fps": fps,
        "confusion_matrix": cm,
        "report": classification_report(y_true, y_pred, labels=TARGET_CLASSES, target_names=TARGET_CLASSES, zero_division=0),
    }

    return metrics


def benchmark_all_models(
    detectors: List[BaseDetector],
    split_file: Path,
    max_samples: int = None,
) -> pd.DataFrame:
    """
    Chạy đánh giá toàn diện tất cả các mô hình và in Bảng Xếp Hạng (Leaderboard).
    """
    results = []

    print("=" * 70)
    print("🏆 BẮT ĐẦU CHẠY BENCHMARK TOÀN DIỆN TẤT CẢ CÁC MÔ HÌNH")
    print(f"[*] Tập đánh giá: {split_file}")
    print("=" * 70)

    for detector in detectors:
        try:
            m = evaluate_detector(detector, split_file, max_samples=max_samples)
            if m:
                results.append({
                    "Phương Pháp / Model": m["model_name"],
                    "Tập Dữ Liệu": m["split"],
                    "Số Mẫu": m["samples"],
                    "Accuracy (%)": round(m["accuracy"] * 100, 2),
                    "Precision (%)": round(m["precision"] * 100, 2),
                    "Recall (%)": round(m["recall"] * 100, 2),
                    "F1-Score (%)": round(m["f1_score"] * 100, 2),
                    "Độ Trễ (ms)": round(m["latency_ms"], 2),
                    "Tốc độ (FPS)": round(m["fps"], 1),
                })
        except Exception as e:
            print(f"[-] Lỗi khi benchmark {detector.model_name}: {e}")

    df_leaderboard = pd.DataFrame(results)

    if not df_leaderboard.empty:
        # Sắp xếp theo F1-Score giảm dần
        df_leaderboard = df_leaderboard.sort_values(by="F1-Score (%)", ascending=False).reset_index(drop=True)
        print("\n" + "=" * 70)
        print("📊 BẢNG XẾP HẠNG KẾT QUẢ BENCHMARK (LEADERBOARD):")
        print("=" * 70)
        print(tabulate(df_leaderboard, headers="keys", tablefmt="fancy_grid", showindex=True))
        print("=" * 70)

    return df_leaderboard
