"""Stage 1 Part 2: Crop traffic signs and classify using basic HSV histogram + classical ML.

Target: Vietnam dataset (archive_new / data/vn).
Features: HSV (36-bin Hue with +30 shift + 8-bin Saturation), L1 normalized (44 dims).
Models: SVM (RBF, balanced) and kNN.
Evaluation: Level A (GT crops) & Level B (YOLO detected crops).
Best model saved to: models/stage1_hsv_vn_best.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from ultralytics import YOLO


def extract_hsv_feature(crop_bgr: np.ndarray) -> np.ndarray:
    """Extract 44-dim HSV feature vector.
    - Resize 32x32
    - Hue shifted by +30 (modulo 180) to make red contiguous
    - 36 bins H + 8 bins S, L1 normalized
    """
    resized = cv2.resize(crop_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1]

    # Shift hue by +30 to prevent red wrap-around
    h_shifted = ((h + 30) % 180).astype(np.uint8)

    hist_h = cv2.calcHist([h_shifted], [0], None, [36], [0, 180]).flatten()
    hist_s = cv2.calcHist([s], [0], None, [8], [0, 256]).flatten()

    feat = np.concatenate([hist_h, hist_s])
    feat = feat / (np.sum(feat) + 1e-7)
    return feat


def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def plot_and_save_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    out_rep_dir = Path("reports/stage1_classify_vn")
    out_rep_dir.mkdir(parents=True, exist_ok=True)
    misc_dir = out_rep_dir / "misclassified"
    misc_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("data/vn/manifest.csv")
    gmap_path = Path("data/vn/group_map.csv")
    yolo_root = Path("data/vn/yolo")

    if not manifest_path.exists() or not gmap_path.exists():
        raise FileNotFoundError("Missing data/vn/manifest.csv or data/vn/group_map.csv")

    manifest = pd.read_csv(manifest_path)
    gmap = pd.read_csv(gmap_path)

    # =========================================================================
    # Bước 0: Nhãn nhóm theo tiền tố QCVN 41
    # =========================================================================
    print("=" * 70)
    print("BƯỚC 0: KIỂM TRA VÀ GOM NHÃN THEO TIỀN TỐ QCVN 41")
    print("=" * 70)

    prefix_to_group = {
        "P": "cấm (prohibitory)",
        "R": "hiệu lệnh (mandatory)",
        "W": "cảnh báo (warning)",
        "I": "chỉ dẫn (information)",
        "S": "phụ (supplementary)",
    }

    mismatches = []
    cls_to_group = {}
    for r in gmap.itertuples():
        p = r.name.split(".")[0]
        rule_grp = prefix_to_group.get(p, "chưa xác định")
        cls_to_group[r.name] = rule_grp

        # Compare with existing group_map.csv group column
        map_grp_raw = r.group
        # map_grp normalized
        norm_map = {
            "prohibitory": "cấm (prohibitory)",
            "mandatory": "hiệu lệnh (mandatory)",
            "warning": "cảnh báo (warning)",
            "information": "chỉ dẫn (information)",
            "supplementary": "phụ (supplementary)",
        }.get(map_grp_raw, "chưa xác định")

        if rule_grp != norm_map:
            mismatches.append({
                "id": r.id,
                "name": r.name,
                "group_map": map_grp_raw,
                "prefix_rule": rule_grp,
            })

    if mismatches:
        print(f"\n[Cảnh báo] Phát hiện {len(mismatches)} lớp có tiền tố lệch với group_map.csv:")
        for m in mismatches:
            print(f"  - ID {m['id']} '{m['name']}': trong group_map là '{m['group_map']}' -> tiền tố quy tắc gán là '{m['prefix_rule']}'")
        print("  -> Theo chỉ đạo, sử dụng QUY TẮC TIỀN TỐ cho các lớp này.\n")
    else:
        print("\nToàn bộ 52 lớp khớp hoàn toàn với group_map.csv.")

    manifest["group_label"] = manifest["label_fine"].map(cls_to_group)

    # Bảng thống kê nhóm
    summary_rows = []
    warnings_list = []
    all_groups = sorted(manifest["group_label"].unique())

    for g in all_groups:
        gdf = manifest[manifest["group_label"] == g]
        num_cls = gdf["label_fine"].nunique()
        tr_cnt = (gdf["split"] == "train").sum()
        val_cnt = (gdf["split"] == "val").sum()
        test_cnt = (gdf["split"] == "test").sum()
        summary_rows.append({
            "Nhóm": g,
            "Số class": num_cls,
            "Train bboxes": tr_cnt,
            "Val bboxes": val_cnt,
            "Test bboxes": test_cnt,
        })
        if tr_cnt < 30:
            warnings_list.append(f"CẢNH BÁO: Nhóm '{g}' chỉ có {tr_cnt} bbox trong tập train (< 30)!")

    summary_df = pd.DataFrame(summary_rows)
    print("BẢNG THỐNG KÊ NHÓM BIỂN BÁO (TIỀN TỐ QCVN 41):")
    print(summary_df.to_string(index=False))

    if warnings_list:
        print()
        for w in warnings_list:
            print(w)
    else:
        print("\nTất cả các nhóm đều có $\\ge$ 30 bboxes trong tập train.")

    # =========================================================================
    # Bước 1: Trích xuất đặc trưng & Màu chủ đạo
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG HSV (44-DIMS) VÀ TÍNH MÀU CHỦ ĐẠO")
    print("=" * 70)

    # Filter out bboxes < 8 px from train
    manifest["bw"] = manifest["x2"] - manifest["x1"]
    manifest["bh"] = manifest["y2"] - manifest["y1"]
    train_mask_raw = (manifest["split"] == "train")
    too_small_train = train_mask_raw & ((manifest["bw"] < 8) | (manifest["bh"] < 8))
    num_dropped = too_small_train.sum()
    print(f"Số bbox train bị loại do kích thước < 8 px: {num_dropped} / {train_mask_raw.sum()}")

    # Valid mask for train
    manifest["valid_train"] = train_mask_raw & (~too_small_train)

    # Group crops by image for fast disk I/O
    def extract_dataset(df_subset, desc="Dataset"):
        X_list = []
        y_list = []
        info_list = []
        mean_h_list = []
        mean_s_list = []

        grouped = df_subset.groupby("yolo_img")
        print(f"Loading & extracting {len(df_subset)} bboxes from {len(grouped)} images ({desc})...")

        for rel_img_path, rows in grouped:
            full_img_p = yolo_root / rel_img_path
            img = cv2.imread(str(full_img_p))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            for r in rows.itertuples():
                x1 = max(0, min(iw - 1, int(r.x1)))
                y1 = max(0, min(ih - 1, int(r.y1)))
                x2 = max(x1 + 1, min(iw, int(r.x2)))
                y2 = max(y1 + 1, min(ih, int(r.y2)))

                crop = img[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                    continue

                feat = extract_hsv_feature(crop)
                X_list.append(feat)
                y_list.append(r.group_label)
                info_list.append({
                    "row_id": r.row_id,
                    "img_stem": r.img_stem,
                    "yolo_img": r.yolo_img,
                    "box": [x1, y1, x2, y2],
                    "label_fine": r.label_fine,
                    "group_label": r.group_label,
                })

                # Compute mean H (shifted +30) and S for dominant color analysis (focusing on colored pixels S > 35)
                crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                colored_mask = crop_hsv[:, :, 1] > 35
                if np.sum(colored_mask) >= 4:
                    h_shift = (crop_hsv[:, :, 0][colored_mask].astype(int) + 30) % 180
                    mean_h_list.append(float(np.mean(h_shift)))
                    mean_s_list.append(float(np.mean(crop_hsv[:, :, 1][colored_mask])))
                else:
                    mean_h_list.append(float((np.mean(crop_hsv[:, :, 0]) + 30) % 180))
                    mean_s_list.append(float(np.mean(crop_hsv[:, :, 1])))

        return np.array(X_list), np.array(y_list), info_list, mean_h_list, mean_s_list

    train_subset = manifest[manifest["valid_train"]].copy()
    val_subset = manifest[manifest["split"] == "val"].copy()
    test_subset = manifest[manifest["split"] == "test"].copy()

    X_train, y_train, info_train, _, _ = extract_dataset(train_subset, "Train GT")
    X_val, y_val, info_val, _, _ = extract_dataset(val_subset, "Val GT")
    X_test_gt, y_test_gt, info_test_gt, test_h, test_s = extract_dataset(test_subset, "Test GT")

    print(f"\nSố mẫu trích xuất thành công: Train={len(X_train)}, Val={len(X_val)}, Test GT={len(X_test_gt)}")

    # Bảng màu chủ đạo theo nhóm
    color_records = []
    for g in all_groups:
        idxs = [i for i, y in enumerate(y_test_gt) if y == g]
        if idxs:
            mean_shifted_h = np.mean([test_h[i] for i in idxs])
            avg_h = (mean_shifted_h - 30.0) % 180.0
            avg_s = np.mean([test_s[i] for i in idxs])

            if avg_s < 35:
                tone = "Trắng / Bạc / Đơn sắc"
            elif avg_h <= 12 or avg_h >= 168:
                tone = "Đỏ (Red)"
            elif 13 <= avg_h <= 35:
                tone = "Vàng / Cam (Yellow/Amber)"
            elif 90 <= avg_h <= 135:
                tone = "Xanh lam (Blue)"
            else:
                tone = f"H ~ {avg_h:.1f}"
            color_records.append({
                "Nhóm": g,
                "Số mẫu Test": len(idxs),
                "H trung bình (0-179)": round(avg_h, 1),
                "S trung bình (0-255)": round(avg_s, 1),
                "Màu chủ đạo nhận định": tone,
            })
        else:
            color_records.append({
                "Nhóm": g,
                "Số mẫu Test": 0,
                "H trung bình (0-179)": "N/A",
                "S trung bình (0-255)": "N/A",
                "Màu chủ đạo nhận định": "Không có mẫu",
            })

    color_df = pd.DataFrame(color_records)
    print("\nBẢNG MÀU CHỦ ĐẠO THEO NHÓM (TRUNG BÌNH H & S):")
    print(color_df.to_string(index=False))

    # =========================================================================
    # Bước 2: Huấn luyện & Tối ưu tham số trên VAL (SVM & kNN)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 2: TỐI ƯU THAM SỐ TRÊN TẬP VAL (SVM & kNN)")
    print("=" * 70)

    # 1. Grid search SVM (RBF, class_weight='balanced')
    best_svm = None
    best_svm_score = -1.0
    best_svm_params = {}

    for c in [1.0, 5.0, 10.0]:
        for g in ["scale", 0.5, 1.0]:
            print(f"  Fitting SVM(C={c}, gamma={g})...", flush=True)
            clf = SVC(kernel="rbf", C=c, gamma=g, class_weight="balanced", random_state=42)
            clf.fit(X_train, y_train)
            pred_v = clf.predict(X_val)
            f1 = f1_score(y_val, pred_v, average="macro", zero_division=0)
            acc = accuracy_score(y_val, pred_v)
            if f1 > best_svm_score:
                best_svm_score = f1
                best_svm = clf
                best_svm_params = {"C": c, "gamma": g, "val_macro_f1": f1, "val_acc": acc}

    print(f"SVM tốt nhất: {best_svm_params}", flush=True)

    # 2. Grid search kNN
    best_knn = None
    best_knn_score = -1.0
    best_knn_params = {}

    for k in [3, 5, 7, 11, 15]:
        for w in ["uniform", "distance"]:
            knn = KNeighborsClassifier(n_neighbors=k, weights=w)
            knn.fit(X_train, y_train)
            pred_v = knn.predict(X_val)
            f1 = f1_score(y_val, pred_v, average="macro", zero_division=0)
            acc = accuracy_score(y_val, pred_v)
            if f1 > best_knn_score:
                best_knn_score = f1
                best_knn = knn
                best_knn_params = {"k": k, "weights": w, "val_macro_f1": f1, "val_acc": acc}

    print(f"kNN tốt nhất: {best_knn_params}", flush=True)

    # =========================================================================
    # Bước 3: Đánh giá MỨC (A) - Crop từ GT Bbox trên TEST VN (Cận trên)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 3: ĐÁNH GIÁ MỨC (A) - CROP TỪ GROUND-TRUTH BBOX TRÊN TEST VN")
    print("=" * 70)

    # SVM Level A
    pred_a_svm = best_svm.predict(X_test_gt)
    acc_a_svm = accuracy_score(y_test_gt, pred_a_svm)
    f1_a_svm = f1_score(y_test_gt, pred_a_svm, average="macro", zero_division=0)
    print(f"\n[Mức A - SVM] Accuracy: {acc_a_svm * 100:.2f}% | Macro-F1: {f1_a_svm:.4f}")
    print(classification_report(y_test_gt, pred_a_svm, labels=all_groups, zero_division=0))
    plot_and_save_confusion_matrix(
        y_test_gt,
        pred_a_svm,
        all_groups,
        "Level A (GT Crops) - SVM Confusion Matrix",
        out_rep_dir / "cm_level_a_svm.png",
    )

    # kNN Level A
    pred_a_knn = best_knn.predict(X_test_gt)
    acc_a_knn = accuracy_score(y_test_gt, pred_a_knn)
    f1_a_knn = f1_score(y_test_gt, pred_a_knn, average="macro", zero_division=0)
    print(f"\n[Mức A - kNN] Accuracy: {acc_a_knn * 100:.2f}% | Macro-F1: {f1_a_knn:.4f}")
    print(classification_report(y_test_gt, pred_a_knn, labels=all_groups, zero_division=0))
    plot_and_save_confusion_matrix(
        y_test_gt,
        pred_a_knn,
        all_groups,
        "Level A (GT Crops) - kNN Confusion Matrix",
        out_rep_dir / "cm_level_a_knn.png",
    )

    # =========================================================================
    # Bước 4: Đánh giá MỨC (B) - Crop từ Bbox YOLO phát hiện (conf=0.25, IoU>=0.5)
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 4: ĐÁNH GIÁ MỨC (B) - CROP TỪ BBOX YOLO PHÁT HIỆN")
    print("=" * 70)

    yolo_weights = Path("runs/detect/vn_s/weights/best.pt")
    if not yolo_weights.exists():
        raise FileNotFoundError(f"Missing YOLO detector weights at {yolo_weights}")

    yolo_model = YOLO(str(yolo_weights))

    # Run detection on all 550 test images
    test_img_dir = yolo_root / "images" / "test"
    test_stems = sorted(test_subset["img_stem"].unique().tolist())
    print(f"Running YOLO inference on {len(test_stems)} test images...")

    matched_gt_labels = []
    matched_crops = []
    matched_features = []
    det_fp_count = 0
    det_missed_gt_count = 0
    misclassified_candidates = []

    for s in test_stems:
        stem_str = f"{int(s):04d}"
        img_p = test_img_dir / f"{stem_str}.jpg"
        if not img_p.exists():
            img_p = test_img_dir / f"{stem_str}.png"
        if not img_p.exists():
            continue

        img = cv2.imread(str(img_p))
        ih, iw = img.shape[:2]

        # GT boxes for this stem
        gt_rows = test_subset[test_subset["img_stem"] == s]
        gt_boxes = []
        gt_lbls = []
        for r in gt_rows.itertuples():
            x1 = max(0, min(iw - 1, int(r.x1)))
            y1 = max(0, min(ih - 1, int(r.y1)))
            x2 = max(x1 + 1, min(iw, int(r.x2)))
            y2 = max(y1 + 1, min(ih, int(r.y2)))
            gt_boxes.append([x1, y1, x2, y2])
            gt_lbls.append(r.group_label)

        # YOLO predict
        pred_res = yolo_model.predict(img_p, conf=0.25, imgsz=800, device=0, verbose=False)[0]
        pred_boxes = []
        for b in pred_res.boxes:
            b_xyxy = b.xyxy[0].cpu().numpy().astype(int)
            px1 = max(0, min(iw - 1, int(b_xyxy[0])))
            py1 = max(0, min(ih - 1, int(b_xyxy[1])))
            px2 = max(px1 + 1, min(iw, int(b_xyxy[2])))
            py2 = max(py1 + 1, min(ih, int(b_xyxy[3])))
            pred_boxes.append([px1, py1, px2, py2])

        # Match IoU >= 0.5
        matched_gt_set = set()
        matched_pred_set = set()

        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt_set:
                    continue
                iou = box_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= 0.5 and best_g_idx != -1:
                matched_gt_set.add(best_g_idx)
                matched_pred_set.add(p_idx)

                # Crop from YOLO predicted box
                pcrop = img[p_box[1]:p_box[3], p_box[0]:p_box[2]]
                if pcrop.size > 0 and pcrop.shape[0] >= 2 and pcrop.shape[1] >= 2:
                    pfeat = extract_hsv_feature(pcrop)
                    matched_crops.append(pcrop)
                    matched_features.append(pfeat)
                    matched_gt_labels.append(gt_lbls[best_g_idx])
                    misclassified_candidates.append({
                        "crop": pcrop,
                        "stem": stem_str,
                        "pred_box": p_box,
                        "gt_box": gt_boxes[best_g_idx],
                        "true_group": gt_lbls[best_g_idx],
                        "iou": round(best_iou, 3),
                    })

        # Count FP and FN in detection
        det_fp_count += (len(pred_boxes) - len(matched_pred_set))
        det_missed_gt_count += (len(gt_boxes) - len(matched_gt_set))

    X_test_det = np.array(matched_features)
    y_test_det_gt = np.array(matched_gt_labels)

    print(f"\nTổng GT boxes trên test: {len(test_subset)}")
    print(f"Số GT ghép thành công với YOLO box (IoU >= 0.5): {len(y_test_det_gt)} ({len(y_test_det_gt)/len(test_subset)*100:.1f}%)")
    print(f"Số GT bỏ sót (FN detector): {det_missed_gt_count}")
    print(f"Số bbox YOLO phát hiện không ghép được GT (FP detector): {det_fp_count}")

    # Level B - SVM
    pred_b_svm = best_svm.predict(X_test_det)
    acc_b_svm = accuracy_score(y_test_det_gt, pred_b_svm)
    f1_b_svm = f1_score(y_test_det_gt, pred_b_svm, average="macro", zero_division=0)
    print(f"\n[Mức B - SVM] Accuracy (trên tập matched): {acc_b_svm * 100:.2f}% | Macro-F1: {f1_b_svm:.4f}")
    print(classification_report(y_test_det_gt, pred_b_svm, labels=all_groups, zero_division=0))
    plot_and_save_confusion_matrix(
        y_test_det_gt,
        pred_b_svm,
        all_groups,
        "Level B (YOLO Crops) - SVM Confusion Matrix",
        out_rep_dir / "cm_level_b_svm.png",
    )

    # Level B - kNN
    pred_b_knn = best_knn.predict(X_test_det)
    acc_b_knn = accuracy_score(y_test_det_gt, pred_b_knn)
    f1_b_knn = f1_score(y_test_det_gt, pred_b_knn, average="macro", zero_division=0)
    print(f"\n[Mức B - kNN] Accuracy (trên tập matched): {acc_b_knn * 100:.2f}% | Macro-F1: {f1_b_knn:.4f}")
    print(classification_report(y_test_det_gt, pred_b_knn, labels=all_groups, zero_division=0))
    plot_and_save_confusion_matrix(
        y_test_det_gt,
        pred_b_knn,
        all_groups,
        "Level B (YOLO Crops) - kNN Confusion Matrix",
        out_rep_dir / "cm_level_b_knn.png",
    )

    # =========================================================================
    # Bước 5: Lưu ~20 ảnh crop bị phân loại sai kèm nhãn thật / nhãn dự đoán
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 5: LƯU CÁC ẢNH CROP BỊ PHÂN LOẠI SAI")
    print("=" * 70)

    # Identify errors in Level B (or Level A) using SVM
    saved_misc = 0
    misc_log = []

    for idx, (cand, true_y, pred_y) in enumerate(zip(misclassified_candidates, y_test_det_gt, pred_b_svm)):
        if true_y != pred_y:
            saved_misc += 1
            crop_img = cand["crop"]
            ch, cw = crop_img.shape[:2]

            # Shorten labels for filename
            short_true = true_y.split()[0]
            short_pred = pred_y.split()[0]
            fname = f"err_{saved_misc:02d}_stem{cand['stem']}_True_{short_true}_Pred_{short_pred}.jpg"
            fpath = misc_dir / fname

            # Resize slightly if too tiny for visual inspection, and add banner
            disp_crop = cv2.resize(crop_img, (max(96, cw * 3), max(96, ch * 3)), interpolation=cv2.INTER_NEAREST)
            dh, dw = disp_crop.shape[:2]

            banner = np.zeros((36, dw, 3), dtype=np.uint8)
            banner[:] = (30, 30, 30)
            txt1 = f"True: {short_true}"
            txt2 = f"Pred: {short_pred}"
            cv2.putText(banner, txt1, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(banner, txt2, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            vis_img = np.vstack([banner, disp_crop])
            cv2.imwrite(str(fpath), vis_img)

            misc_log.append({
                "index": saved_misc,
                "file": fname,
                "stem": cand["stem"],
                "true_label": true_y,
                "pred_label": pred_y,
                "crop_size": f"{cw}x{ch}",
                "iou": cand["iou"],
            })

            if saved_misc >= 20:
                break

    (out_rep_dir / "misclassified_examples.json").write_text(json.dumps(misc_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã lưu {saved_misc} ảnh crop phân loại sai vào: {misc_dir}")

    # =========================================================================
    # Bước 6: Lưu model tốt nhất vào models/stage1_hsv_vn_best.joblib
    # =========================================================================
    print("\n" + "=" * 70)
    print("BƯỚC 6: LƯU MODEL TỐT NHẤT")
    print("=" * 70)

    best_model_name = "SVM" if best_svm_score >= best_knn_score else "kNN"
    best_obj = best_svm if best_svm_score >= best_knn_score else best_knn
    best_val_score = max(best_svm_score, best_knn_score)

    save_model_path = model_dir / "stage1_hsv_vn_best.joblib"
    save_payload = {
        "model_type": best_model_name,
        "model": best_obj,
        "all_groups": all_groups,
        "prefix_to_group": prefix_to_group,
        "cls_to_group": cls_to_group,
        "val_macro_f1": best_val_score,
        "svm_params": best_svm_params,
        "knn_params": best_knn_params,
    }
    joblib.dump(save_payload, save_model_path)
    print(f"Model tốt nhất ({best_model_name}, Val Macro-F1: {best_val_score:.4f}) đã lưu tại: {save_model_path.resolve()}")

    # Summary table
    final_summary = {
        "model_selection": {
            "SVM": best_svm_params,
            "kNN": best_knn_params,
            "selected": best_model_name,
        },
        "level_a_gt_crops": {
            "SVM": {"accuracy": acc_a_svm, "macro_f1": f1_a_svm},
            "kNN": {"accuracy": acc_a_knn, "macro_f1": f1_a_knn},
        },
        "level_b_yolo_crops": {
            "det_total_gt": len(test_subset),
            "det_matched_gt": len(y_test_det_gt),
            "det_fn_missed": det_missed_gt_count,
            "det_fp_extra": det_fp_count,
            "SVM": {"accuracy": acc_b_svm, "macro_f1": f1_b_svm},
            "kNN": {"accuracy": acc_b_knn, "macro_f1": f1_b_knn},
        },
    }
    (out_rep_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[Hoàn thành] Đã xuất báo cáo và ma trận nhầm lẫn vào:", out_rep_dir)


if __name__ == "__main__":
    main()
