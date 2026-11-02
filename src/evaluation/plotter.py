"""
Module Trực quan hóa Kết quả Đánh giá & So sánh (Plotter).
Xuất biểu đồ phục vụ Báo cáo đồ án và Slide thuyết trình.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import RESULTS_DIR, TARGET_CLASSES
from src.utils import ensure_dir


def plot_benchmark_comparison(df_leaderboard: pd.DataFrame, save_path: Path = None):
    """Vẽ biểu đồ cột so sánh Accuracy, F1-Score và FPS giữa các mô hình."""
    if df_leaderboard.empty:
        return

    ensure_dir(RESULTS_DIR)
    if save_path is None:
        save_path = RESULTS_DIR / "benchmark_comparison.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Biểu đồ 1: Accuracy & F1-Score
    df_melted = pd.melt(
        df_leaderboard,
        id_vars=["Phương Pháp / Model"],
        value_vars=["Accuracy (%)", "F1-Score (%)"],
        var_name="Chỉ số",
        value_name="Điểm số (%)",
    )
    sns.barplot(data=df_melted, x="Phương Pháp / Model", y="Điểm số (%)", hue="Chỉ số", ax=ax1, palette="viridis")
    ax1.set_title("So sánh Độ chính xác & F1-Score", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 105)

    # Biểu đồ 2: Tốc độ suy luận (FPS)
    sns.barplot(data=df_leaderboard, x="Phương Pháp / Model", y="Tốc độ (FPS)", ax=ax2, hue="Phương Pháp / Model", palette="magma", legend=False)
    ax2.set_title("So sánh Tốc độ Xử lý (Frames Per Second - FPS)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[+] Đã lưu biểu đồ so sánh tại: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, model_name: str, save_path: Path = None):
    """Vẽ Heatmap ma trận nhầm lẫn cho 1 mô hình cụ thể."""
    ensure_dir(RESULTS_DIR)
    if save_path is None:
        save_path = RESULTS_DIR / f"confusion_matrix_{model_name}.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=TARGET_CLASSES,
        yticklabels=TARGET_CLASSES,
    )
    plt.title(f"Ma trận nhầm lẫn - [{model_name}]", fontsize=13, fontweight="bold")
    plt.xlabel("Nhãn Dự đoán (Predicted)", fontsize=11)
    plt.ylabel("Nhãn Thực tế (Actual)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[+] Đã lưu Confusion Matrix tại: {save_path}")
    plt.close()
