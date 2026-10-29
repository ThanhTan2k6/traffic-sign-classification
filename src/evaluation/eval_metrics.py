"""
Module Đánh giá Hiệu năng (Evaluation & Benchmarking Metrics).
Độc lập với mô hình (Model-agnostic), đo đạc các chỉ số:
- Accuracy, Precision, Recall, F1-Score
- Tốc độ suy luận: Latency (ms) và FPS
- Ma trận nhầm lẫn (Confusion Matrix)
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
from tqdm import tqdm

from src.config import RAW_DATA_DIR, TARGET_CLASSES, CLASS_TO_IDX
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
        raise FileNotFoundError(f"Không tìm thấy file split: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split("\t") for line in f if line.strip()]

    if max_samples:
        lines = lines[:max_samples]

    y_true = []
    y_pred = []
    latencies = []

    print(f"\n[*] Đang đánh giá model: [{detector.model_name}] trên tập: {split_file.stem} ({len(lines)} mẫu)...")

    for rel_path, true_class in tqdm(lines, desc=f"Evaluating {detector.model_name}", leave=False):
        full_path = RAW_DATA_DIR / rel_path
        img = imread_unicode(full_path)
        if img is None:
            continue

        try:
            res = detector.predict(img)
            pred_class = res["class_name"]
            latencies.append(res.get("inference_time", 0.0))

            y_true.append(true_class)
            y_pred.append(pred_class)
        except Exception as e:
            # Mô hình chưa được huấn luyện hoặc lỗi suy luận
            y_true.append(true_class)
            y_pred.append("Unknown")

    if not y_true:
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
        "report": classification_report(y_true, y_pred, target_names=TARGET_CLASSES, zero_division=0),
    }

    return metrics


def benchmark_all_models(
    detectors: List[BaseDetector],
    split_file: Path,
) -> pd.DataFrame:
    """
    Chạy đánh giá so sánh (Benchmarking) tất cả các mô hình trên cùng 1 tập split.
    Xuất ra Leaderboard dạng bảng đối chiếu.
    """
    results = []
    print("=" * 70)
    print(f"📊 BẮT ĐẦU BENCHMARK TRÊN TẬP: {split_file.name}")
    print("=" * 70)

    for det in detectors:
        try:
            m = evaluate_detector(det, split_file)
            if m:
                results.append({
                    "Phương Pháp / Model": m["model_name"],
                    "Accuracy (%)": round(m["accuracy"] * 100, 2),
                    "Precision (%)": round(m["precision"] * 100, 2),
                    "Recall (%)": round(m["recall"] * 100, 2),
                    "F1-Score (%)": round(m["f1_score"] * 100, 2),
                    "Latency (ms)": round(m["latency_ms"], 2),
                    "Tốc độ (FPS)": round(m["fps"], 1),
                })
        except Exception as e:
            print(f"[-] Bỏ qua model {det.model_name}: {e}")

    df_leaderboard = pd.DataFrame(results)
    if not df_leaderboard.empty:
        df_leaderboard = df_leaderboard.sort_values(by="Accuracy (%)", ascending=False)
        print("\n" + "=" * 70)
        print("🏆 BẢNG XẾP HẠNG KẾT QUẢ SO SÁNH (BENCHMARK LEADERBOARD):")
        print("=" * 70)
        print(df_leaderboard.to_string(index=False))
        print("=" * 70)
    else:
        print("[-] Chưa có mô hình nào nạp được trọng số để đánh giá.")

    return df_leaderboard
