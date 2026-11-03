"""
================================================================================
📊 EDA DATASET TRAFFIC SIGNS (KHÁM PHÁ & TRỰC QUAN HÓA DỮ LIỆU CHUNG)
Dự án Nhận dạng Biển báo Giao thông (QCVN 41:2019/BGTVT)
Phục vụ mục 4.1 (Mô tả dataset - 10đ), 2.2 (Hình minh họa - 5đ) và 5 (Nhận xét - 5đ).
Tự động xuất toàn bộ biểu đồ vào: res/outputs/eda/ để chèn trực tiếp vào Slide.
================================================================================
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# Reconfigure stdout cho Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    IMAGES_DIR,
    LABELS_DIR,
    CLASSES_FILE,
    CROPPED_DATA_DIR,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_FILE,
    TEST_SPLIT_FILE,
    TARGET_CLASSES,
    RESULTS_DIR,
    get_target_category,
    load_52_classes,
    load_52_classes_vie,
)
from src.utils import ensure_dir, imread_unicode

# Thiết lập phong cách đồ họa Seaborn
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

EDA_OUTPUT_DIR = ensure_dir(RESULTS_DIR / "eda")


def load_dataset_info():
    """Quét và tổng hợp thông tin toàn bộ ảnh và nhãn."""
    classes_52 = load_52_classes()
    classes_vie = load_52_classes_vie()

    label_files = list(LABELS_DIR.glob("*.txt"))
    img_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))

    box_records = []
    img_resolutions = []

    for img_p in img_files:
        lbl_p = LABELS_DIR / f"{img_p.stem}.txt"
        img = imread_unicode(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_resolutions.append((w, h))

        if not lbl_p.exists():
            continue

        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                if cid >= len(classes_52):
                    continue
                cname = classes_52[cid]
                cat = get_target_category(cname)
                if cat is None:
                    continue

                xc, yc, bw, bh = map(float, parts[1:5])
                pixel_w = bw * w
                pixel_h = bh * h

                # Lọc bỏ bounding box nhiễu dưới 10px đồng bộ với pipeline split
                if pixel_w < 10 or pixel_h < 10:
                    continue

                box_records.append({
                    "image": img_p.name,
                    "class_id_52": cid,
                    "class_name_52": cname,
                    "class_vie": classes_vie.get(cname, cname),
                    "category": cat,
                    "width_px": pixel_w,
                    "height_px": pixel_h,
                    "aspect_ratio": pixel_w / max(1.0, pixel_h),
                    "area_px": pixel_w * pixel_h,
                })

    df_boxes = pd.DataFrame(box_records)
    df_imgs = pd.DataFrame(img_resolutions, columns=["Width", "Height"])
    return df_boxes, df_imgs, classes_52, classes_vie


def plot_class_distribution(df_boxes):
    """Biểu đồ 1: Phân bố số lượng và tỷ lệ của 4 nhóm biển báo chính."""
    cat_counts = df_boxes["category"].value_counts().reindex(TARGET_CLASSES, fill_value=0)
    total = len(df_boxes)
    df_cat = pd.DataFrame({
        "Nhóm Biển Báo": TARGET_CLASSES,
        "Số Lượng": [int(cat_counts[c]) for c in TARGET_CLASSES],
        "Tỷ Lệ (%)": [round(float(cat_counts[c]) / total * 100, 2) if total > 0 else 0.0 for c in TARGET_CLASSES],
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Cột phân bố
    colors = ["#e74c3c", "#f39c12", "#2980b9", "#27ae60"]
    bars = ax1.bar(df_cat["Nhóm Biển Báo"], df_cat["Số Lượng"], color=colors, edgecolor="black", alpha=0.85)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{int(yval)}", ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Phân bố số lượng Bounding Box theo 4 Nhóm Biển Báo", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Số lượng mẫu (Boxes)")
    ax1.set_ylim(0, max(df_cat["Số Lượng"]) * 1.15)

    # Biểu đồ tròn
    wedges, texts, autotexts = ax2.pie(
        df_cat["Tỷ Lệ (%)"],
        labels=df_cat["Nhóm Biển Báo"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=(0.05, 0.05, 0.05, 0.05),
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    ax2.set_title("Tỷ lệ phân bố (%) giữa các nhóm (Class Imbalance)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out_path = EDA_OUTPUT_DIR / "1_class_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Biểu đồ 1: {out_path}")


def plot_bbox_dimensions(df_boxes):
    """Biểu đồ 2: Phân tích kích thước ROI biển báo (Width, Height, Aspect Ratio)."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Scatter plot Width vs Height
    sns.scatterplot(
        data=df_boxes,
        x="width_px",
        y="height_px",
        hue="category",
        alpha=0.6,
        palette="tab10",
        ax=ax1,
    )
    ax1.set_title("Biểu đồ tương quan Chiều rộng vs Chiều cao của Biển báo", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Chiều rộng (Pixel)")
    ax1.set_ylabel("Chiều cao (Pixel)")
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)

    # Histogram kích thước cạnh
    sns.histplot(df_boxes["width_px"], bins=40, kde=True, color="#3498db", ax=ax2)
    ax2.axvline(64, color="red", linestyle="--", label="Chuẩn hóa 64x64")
    ax2.set_title("Phân phối kích thước chiều rộng biển báo", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Pixel")
    ax2.set_xlim(0, 160)
    ax2.legend()

    # Boxplot tỉ lệ khung hình Aspect Ratio theo nhóm
    sns.boxplot(data=df_boxes, x="category", y="aspect_ratio", palette="Set2", ax=ax3)
    ax3.axhline(1.0, color="red", linestyle="--", label="Tỉ lệ 1:1 (Vuông/Tròn)")
    ax3.set_title("Tỉ lệ khung hình (Aspect Ratio = W / H) theo từng nhóm", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Aspect Ratio")
    ax3.set_ylim(0.5, 2.0)
    ax3.legend()

    # Phân bố độ phân giải ảnh gốc
    img_res = df_boxes["image"].drop_duplicates()
    sns.histplot(df_boxes["area_px"], bins=35, kde=True, color="#9b59b6", ax=ax4)
    ax4.set_title("Phân phối Diện tích Bounding Box (Width x Height)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Diện tích (Pixel vuông)")
    ax4.set_xlim(0, 15000)

    plt.tight_layout()
    out_path = EDA_OUTPUT_DIR / "2_bbox_dimensions.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Biểu đồ 2: {out_path}")


def plot_sample_grid():
    """Biểu đồ 3: Lưới ảnh minh họa 4 nhóm biển báo phục vụ Slide thuyết trình."""
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    fig.suptitle("HÌNH ẢNH MẪU CÁC BIỂN BÁO THEO 4 NHÓM QUY CHUẨN QCVN 41:2019/BGTVT", fontsize=15, fontweight="bold")

    for row_idx, cat in enumerate(TARGET_CLASSES):
        cat_dir = CROPPED_DATA_DIR / "train" / cat
        if not cat_dir.exists():
            continue
        sample_imgs = list(cat_dir.glob("*.jpg"))[:6]
        for col_idx, img_p in enumerate(sample_imgs):
            ax = axes[row_idx, col_idx]
            img = Image.open(img_p)
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_title(f"[{cat}]", fontsize=12, fontweight="bold", loc="left")

    plt.tight_layout()
    out_path = EDA_OUTPUT_DIR / "3_sample_traffic_signs_grid.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Biểu đồ 3: {out_path}")


def plot_split_distribution():
    """Biểu đồ 4: So sánh phân bố giữa các tập Train / Val / Test (Chứng minh Zero Leakage)."""
    splits = [("Train", TRAIN_SPLIT_FILE), ("Val", VAL_SPLIT_FILE), ("Test", TEST_SPLIT_FILE)]
    data = []
    for split_name, s_file in splits:
        if not s_file.exists():
            continue
        with open(s_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    data.append({"Split": split_name, "Category": parts[1]})

    df_split = pd.DataFrame(data)
    if df_split.empty:
        return

    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=df_split, x="Category", hue="Split", palette="deep")
    plt.title("Phân bổ số lượng mẫu đồng đều giữa 3 tập Train - Val - Test", fontsize=13, fontweight="bold")
    plt.xlabel("Nhóm Biển Báo")
    plt.ylabel("Số lượng mẫu (ROIs)")

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                        textcoords="offset points")

    plt.tight_layout()
    out_path = EDA_OUTPUT_DIR / "4_split_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Biểu đồ 4: {out_path}")


def main():
    print("=" * 65)
    print("🚀 BẮT ĐẦU CHẠY PHÂN TÍCH KHÁM PHÁ DỮ LIỆU EDA (DATASET INSIGHTS)")
    print("=" * 65)

    df_boxes, df_imgs, classes_52, classes_vie = load_dataset_info()

    print(f"\n[1] Thống kê tổng quan:")
    print(f"    • Tổng số ảnh toàn cảnh : {len(df_imgs)} ảnh")
    print(f"    • Độ phân giải ảnh      : {dict(df_imgs.value_counts())}")
    print(f"    • Tổng số bounding box  : {len(df_boxes)} boxes hợp lệ")

    print(f"\n[2] Phân bố theo 4 nhóm mục tiêu:")
    for cat in TARGET_CLASSES:
        cnt = (df_boxes["category"] == cat).sum()
        pct = cnt / len(df_boxes) * 100
        print(f"    • {cat:15s}: {cnt:4d} boxes ({pct:.2f}%)")

    # Xuất các biểu đồ
    plot_class_distribution(df_boxes)
    plot_bbox_dimensions(df_boxes)
    plot_sample_grid()
    plot_split_distribution()

    print("\n" + "=" * 65)
    print(f"✅ TOÀN BỘ BIỂU ĐỒ EDA ĐÃ ĐƯỢC XUẤT THÀNH CÔNG VÀO:")
    print(f"   📂 {EDA_OUTPUT_DIR}")
    print("   👉 Bạn có thể mở thư mục này và chèn thẳng các file .png vào Slide Báo Cáo!")
    print("=" * 65)


if __name__ == "__main__":
    main()
