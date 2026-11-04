"""Stage 2: Upgrading the classifier on the VIETNAM dataset.
- Relabeling: B.8a (ID 34) -> cấm (prohibitory), Camera (ID 31) -> EXCLUDED from classifier.
- S0: Stage 1 baseline on new labels (H 36-bin shifted + S 8-bin, L1).
- S1: S0 + aspect ratio w/h.
- S2: S1 + spatial grid (compares 2x2 vs 3x3, selects best on val, reports both).
- S3: S2 + HOG 64x64.
- Models: SVM (RBF, balanced), kNN, Random Forest (300 trees, balanced).
- Evaluation: GroupKFold 5-fold (mean +- std) + Fixed Split (A, A', B, end-to-end recall).
"""
from __future__ import annotations

import json
import os
import random
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
from scipy.spatial.distance import cdist
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ultralytics import YOLO

# Global constants & seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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

def extract_raw_features(crop_bgr: np.ndarray, w_orig: float, h_orig: float):
    """Extract S0, S1, S2_2x2, S2_3x3, and HOG features for a single crop."""
    # S0
    c32 = cv2.resize(crop_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    hsv32 = cv2.cvtColor(c32, cv2.COLOR_BGR2HSV)
    h32 = ((hsv32[:, :, 0].astype(np.int32) + 30) % 180).astype(np.uint8)
    s32 = hsv32[:, :, 1]
    hist_h = cv2.calcHist([h32], [0], None, [36], [0, 180]).flatten()
    hist_s = cv2.calcHist([s32], [0], None, [8], [0, 256]).flatten()
    s0 = np.concatenate([hist_h, hist_s])
    s0 = s0 / (np.sum(s0) + 1e-7)  # 44 dims

    # S1
    ar = float(w_orig) / float(max(h_orig, 1.0))
    s1 = np.append(s0, ar)  # 45 dims

    # S2_2x2: 4 cells of 16x16, H(18) + S(4) = 22 dims per cell -> 88 dims
    cells_2x2 = []
    for r in range(2):
        for c in range(2):
            cell_h = h32[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16]
            cell_s = s32[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16]
            hh = cv2.calcHist([cell_h], [0], None, [18], [0, 180]).flatten()
            ss = cv2.calcHist([cell_s], [0], None, [4], [0, 256]).flatten()
            c_feat = np.concatenate([hh, ss])
            c_feat = c_feat / (np.sum(c_feat) + 1e-7)
            cells_2x2.append(c_feat)
    s2_2x2 = np.concatenate([s1, np.concatenate(cells_2x2)])  # 133 dims

    # S2_3x3: 9 cells of 12x12 on 36x36, H(18) + S(4) = 22 dims per cell -> 198 dims
    c36 = cv2.resize(crop_bgr, (36, 36), interpolation=cv2.INTER_AREA)
    hsv36 = cv2.cvtColor(c36, cv2.COLOR_BGR2HSV)
    h36 = ((hsv36[:, :, 0].astype(np.int32) + 30) % 180).astype(np.uint8)
    s36 = hsv36[:, :, 1]
    cells_3x3 = []
    for r in range(3):
        for c in range(3):
            cell_h = h36[r * 12 : (r + 1) * 12, c * 12 : (c + 1) * 12]
            cell_s = s36[r * 12 : (r + 1) * 12, c * 12 : (c + 1) * 12]
            hh = cv2.calcHist([cell_h], [0], None, [18], [0, 180]).flatten()
            ss = cv2.calcHist([cell_s], [0], None, [4], [0, 256]).flatten()
            c_feat = np.concatenate([hh, ss])
            c_feat = c_feat / (np.sum(c_feat) + 1e-7)
            cells_3x3.append(c_feat)
    s2_3x3 = np.concatenate([s1, np.concatenate(cells_3x3)])  # 243 dims

    # HOG (64x64, 9 orientations, 16x16 pixels per cell, 2x2 cells per block) -> 324 dims
    c64 = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray64 = cv2.cvtColor(c64, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray64,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )  # 324 dims

    return s0, s1, s2_2x2, s2_3x3, hog_feat

def plot_confusion_matrix(y_true, y_pred, labels, title, ax):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=[l.split()[0] for l in labels],
        yticklabels=[l.split()[0] for l in labels],
        title=title,
        ylabel="True Label",
        xlabel="Pred Label",
    )
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
                fontsize=8,
            )

def main():
    out_dir = Path("reports/stage2_vn")
    out_dir.mkdir(parents=True, exist_ok=True)
    misc_dir = out_dir / "misclassified"
    misc_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir = out_dir / "fixed_samples"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("data/vn/manifest.csv")
    gmap_path = Path("data/vn/group_map.csv")
    yolo_root = Path("data/vn/yolo")
    yolo_weights = Path("runs/detect/vn_s/weights/best.pt")

    manifest = pd.read_csv(manifest_path)
    gmap = pd.read_csv(gmap_path)

    # =========================================================================
    # Bước 0: Sửa nhãn (B.8a -> cấm, Camera -> LOẠI)
    # =========================================================================
    print("=" * 70, flush=True)
    print("BƯỚC 0: SỬA NHÃN & THỐNG KÊ PHÂN BỐ", flush=True)
    print("=" * 70, flush=True)

    # 1. Bảng trước khi sửa nhãn (Stage 1 rule)
    s1_prefix_to_group = {
        "P": "cấm (prohibitory)",
        "R": "hiệu lệnh (mandatory)",
        "W": "cảnh báo (warning)",
        "I": "chỉ dẫn (information)",
        "S": "phụ (supplementary)",
    }
    manifest["group_s1"] = manifest["label_fine"].apply(
        lambda n: s1_prefix_to_group.get(n.split(".")[0], "chưa xác định")
    )

    table_before = []
    for g in sorted(manifest["group_s1"].unique()):
        sub = manifest[manifest["group_s1"] == g]
        table_before.append({
            "nhóm": g,
            "số class": sub["label_fine"].nunique(),
            "train": int((sub["split"] == "train").sum()),
            "val": int((sub["split"] == "val").sum()),
            "test": int((sub["split"] == "test").sum()),
            "tổng": len(sub),
        })
    df_before = pd.DataFrame(table_before)
    print("\n--- BẢNG TRƯỚC SỬA NHÃN (STAGE 1) ---", flush=True)
    print(df_before.to_string(index=False), flush=True)

    # 2. Quy tắc sửa nhãn Stage 2
    def map_stage2_group(label_fine: str) -> str | None:
        if label_fine == "Camera":
            return None  # EXCLUDED
        if label_fine == "B.8a":
            return "cấm (prohibitory)"
        p = label_fine.split(".")[0]
        return s1_prefix_to_group.get(p, "chưa xác định")

    manifest["group_s2"] = manifest["label_fine"].apply(map_stage2_group)
    
    # Filter dataset for classifier: remove Camera
    clf_manifest = manifest[manifest["group_s2"].notna()].copy()
    
    # Check drop bboxes < 8px from train
    clf_manifest["bw"] = clf_manifest["x2"] - clf_manifest["x1"]
    clf_manifest["bh"] = clf_manifest["y2"] - clf_manifest["y1"]
    
    train_mask = clf_manifest["split"] == "train"
    small_train_mask = train_mask & ((clf_manifest["bw"] < 8) | (clf_manifest["bh"] < 8))
    num_small_dropped = int(small_train_mask.sum())
    print(f"\nSố bbox train bị loại do < 8 px: {num_small_dropped} / {int(train_mask.sum())}", flush=True)
    
    valid_clf_manifest = clf_manifest[~small_train_mask].copy()

    # Bảng sau sửa nhãn
    table_after = []
    canonical_groups = [
        "cấm (prohibitory)",
        "cảnh báo (warning)",
        "hiệu lệnh (mandatory)",
        "chỉ dẫn (information)",
        "phụ (supplementary)",
    ]
    for g in canonical_groups:
        sub = valid_clf_manifest[valid_clf_manifest["group_s2"] == g]
        table_after.append({
            "nhóm": g,
            "số class": sub["label_fine"].nunique(),
            "train": int((sub["split"] == "train").sum()),
            "val": int((sub["split"] == "val").sum()),
            "test": int((sub["split"] == "test").sum()),
            "tổng": len(sub),
        })
    df_after = pd.DataFrame(table_after)
    print("\n--- BẢNG SAU SỬA NHÃN (STAGE 2) ---", flush=True)
    print(df_after.to_string(index=False), flush=True)

    # Build continuous block group_ids for GroupKFold
    stems = sorted(valid_clf_manifest["img_stem"].unique())
    stem_to_group = {}
    curr_group = 0
    prev_s = stems[0]
    for s in stems:
        if s - prev_s > 3:
            curr_group += 1
        stem_to_group[s] = curr_group
        prev_s = s
    valid_clf_manifest["group_id"] = valid_clf_manifest["img_stem"].map(stem_to_group)
    print(f"\nTổng số khối video liên tục (GroupKFold groups): {valid_clf_manifest['group_id'].nunique()}", flush=True)

    # Verify zero leakage across folds
    gkf = GroupKFold(n_splits=5)
    fold_assignments = np.zeros(len(valid_clf_manifest), dtype=int)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(valid_clf_manifest, groups=valid_clf_manifest["group_id"])):
        fold_assignments[val_idx] = fold
    valid_clf_manifest["fold"] = fold_assignments

    stem_to_fold = valid_clf_manifest.groupby("img_stem")["fold"].first().to_dict()
    adj_diff_fold = 0
    for s in stems:
        if s + 1 in stem_to_fold and stem_to_fold[s] != stem_to_fold[s + 1]:
            adj_diff_fold += 1
    print(f"Số cặp frame kề nhau nằm ở 2 fold khác nhau: {adj_diff_fold} (Phải = 0)", flush=True)

    # =========================================================================
    # Trích xuất đặc trưng hàng loạt cho toàn bộ valid_clf_manifest
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("TRÍCH XUẤT ĐẶC TRƯNG HÀNG LOẠT (S0, S1, S2_2x2, S2_3x3, S3)", flush=True)
    print("=" * 70, flush=True)

    feat_cache_file = Path("data/stage2_feat_cache.joblib")
    if feat_cache_file.exists():
        print(f"Tải đặc trưng từ cache: {feat_cache_file}", flush=True)
        cached_data = joblib.load(feat_cache_file)
        X_s0 = cached_data["X_s0"]
        X_s1 = cached_data["X_s1"]
        X_s2_2x2 = cached_data["X_s2_2x2"]
        X_s2_3x3 = cached_data["X_s2_3x3"]
        X_hog = cached_data["X_hog"]
        y_all = cached_data["y_all"]
        splits_all = cached_data["splits_all"]
        folds_all = cached_data["folds_all"]
        meta_list = cached_data["meta_list"]
    else:
        feats_s0 = []
        feats_s1 = []
        feats_s2_2x2 = []
        feats_s2_3x3 = []
        feats_hog = []
        labels_list = []
        meta_list = []

        grouped = valid_clf_manifest.groupby("yolo_img")
        print(f"Đang đọc và trích xuất đặc trưng cho {len(valid_clf_manifest)} crops từ {len(grouped)} ảnh...", flush=True)

        for rel_img_p, rows in grouped:
            full_img_p = yolo_root / rel_img_p
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

                s0, s1, s2_2, s2_3, hf = extract_raw_features(crop, x2 - x1, y2 - y1)
                feats_s0.append(s0.astype(np.float32))
                feats_s1.append(s1.astype(np.float32))
                feats_s2_2x2.append(s2_2.astype(np.float32))
                feats_s2_3x3.append(s2_3.astype(np.float32))
                feats_hog.append(hf.astype(np.float32))
                labels_list.append(r.group_s2)
                meta_list.append({
                    "stem": int(r.img_stem),
                    "box": [x1, y1, x2, y2],
                    "split": r.split,
                    "fold": r.fold,
                    "label_fine": r.label_fine,
                    "group": r.group_s2,
                })

        X_s0 = np.array(feats_s0, dtype=np.float32)
        X_s1 = np.array(feats_s1, dtype=np.float32)
        X_s2_2x2 = np.array(feats_s2_2x2, dtype=np.float32)
        X_s2_3x3 = np.array(feats_s2_3x3, dtype=np.float32)
        X_hog = np.array(feats_hog, dtype=np.float32)
        y_all = np.array(labels_list)
        splits_all = np.array([m["split"] for m in meta_list])
        folds_all = np.array([m["fold"] for m in meta_list])

        joblib.dump({
            "X_s0": X_s0,
            "X_s1": X_s1,
            "X_s2_2x2": X_s2_2x2,
            "X_s2_3x3": X_s2_3x3,
            "X_hog": X_hog,
            "y_all": y_all,
            "splits_all": splits_all,
            "folds_all": folds_all,
            "meta_list": meta_list,
        }, feat_cache_file)
        print(f"Đã lưu cache đặc trưng vào {feat_cache_file}", flush=True)

    print(f"Trích xuất thành công: {len(y_all)} mẫu", flush=True)
    print(f"Số chiều: S0={X_s0.shape[1]}, S1={X_s1.shape[1]}, S2_2x2={X_s2_2x2.shape[1]}, S2_3x3={X_s2_3x3.shape[1]}, HOG={X_hog.shape[1]}", flush=True)

    # Evaluate S2 2x2 vs 3x3 on validation split to pick the best for S3
    train_idx_fixed = np.where(splits_all == "train")[0]
    val_idx_fixed = np.where(splits_all == "val")[0]
    test_idx_fixed = np.where(splits_all == "test")[0]

    # Test SVM on Val for 2x2 vs 3x3
    scaler_2x2 = StandardScaler().fit(X_s2_2x2[train_idx_fixed])
    svm_2x2 = SVC(C=5.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=SEED)
    svm_2x2.fit(scaler_2x2.transform(X_s2_2x2[train_idx_fixed]), y_all[train_idx_fixed])
    val_pred_2x2 = svm_2x2.predict(scaler_2x2.transform(X_s2_2x2[val_idx_fixed]))
    val_f1_2x2 = f1_score(y_all[val_idx_fixed], val_pred_2x2, average="macro", zero_division=0)
    val_acc_2x2 = accuracy_score(y_all[val_idx_fixed], val_pred_2x2)

    scaler_3x3 = StandardScaler().fit(X_s2_3x3[train_idx_fixed])
    svm_3x3 = SVC(C=5.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=SEED)
    svm_3x3.fit(scaler_3x3.transform(X_s2_3x3[train_idx_fixed]), y_all[train_idx_fixed])
    val_pred_3x3 = svm_3x3.predict(scaler_3x3.transform(X_s2_3x3[val_idx_fixed]))
    val_f1_3x3 = f1_score(y_all[val_idx_fixed], val_pred_3x3, average="macro", zero_division=0)
    val_acc_3x3 = accuracy_score(y_all[val_idx_fixed], val_pred_3x3)

    print(f"\nSo sánh lưới trên Val: 2x2 (Acc={val_acc_2x2*100:.2f}%, F1={val_f1_2x2:.4f}) vs 3x3 (Acc={val_acc_3x3*100:.2f}%, F1={val_f1_3x3:.4f})", flush=True)
    chosen_s2_name = "2x2" if val_f1_2x2 >= val_f1_3x3 else "3x3"
    X_s2_chosen = X_s2_2x2 if chosen_s2_name == "2x2" else X_s2_3x3
    print(f"-> Chọn S2 ({chosen_s2_name}) cho chuỗi S3.", flush=True)

    # S3 combination
    X_s3 = np.concatenate([X_s2_chosen, X_hog], axis=1)
    print(f"Số chiều S3: {X_s3.shape[1]}", flush=True)

    # Feature sets dictionary
    feature_sets = {
        "S0 (H36+S8)": X_s0,
        "S1 (+w/h)": X_s1,
        "S2_2x2 (+grid 2x2)": X_s2_2x2,
        "S2_3x3 (+grid 3x3)": X_s2_3x3,
        "S3 (+HOG)": X_s3,
    }

    # =========================================================================
    # Phần 1 & Phần 2: Đánh giá chuỗi đặc trưng trên GroupKFold và Fixed Split
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("ĐÁNH GIÁ CHUỖI ĐẶC TRƯNG (GROUPKFOLD 5-FOLD & FIXED SPLIT)", flush=True)
    print("=" * 70, flush=True)

    # Grid definitions
    models_grid = {
        "SVM": {
            "param_name": "C",
            "candidates": [
                {"C": 5.0, "gamma": "scale"},
            ],
            "builder": lambda p: SVC(C=p["C"], gamma=p["gamma"], kernel="rbf", class_weight="balanced", random_state=SEED, cache_size=1000),
        },
        "kNN": {
            "param_name": "k",
            "candidates": [
                {"k": 11, "weights": "uniform"},
            ],
            "builder": lambda p: KNeighborsClassifier(n_neighbors=p["k"], weights=p["weights"]),
        },
        "Random Forest": {
            "param_name": "max_depth",
            "candidates": [
                {"max_depth": 25, "n_estimators": 300},
            ],
            "builder": lambda p: RandomForestClassifier(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                class_weight="balanced",
                random_state=SEED,
                n_jobs=4,
            ),
        },
    }

    results_table = []
    chain_plot_data = {"SVM": [], "kNN": [], "Random Forest": []}
    detailed_metrics = {}
    saved_models = {}

    eval_cache_file = Path("data/stage2_eval_cache.joblib")
    if eval_cache_file.exists():
        print(f"Tải kết quả đánh giá đặc trưng từ cache: {eval_cache_file}", flush=True)
        cached_eval = joblib.load(eval_cache_file)
        results_table = cached_eval["results_table"]
        chain_plot_data = cached_eval["chain_plot_data"]
        detailed_metrics = cached_eval["detailed_metrics"]
        saved_models = cached_eval["saved_models"]
    else:
        for f_name, X_feat in feature_sets.items():
            import gc
            gc.collect()
            print(f"\n--- Đặc trưng: {f_name} (Kích thước: {X_feat.shape[1]} dims) ---", flush=True)
            detailed_metrics[f_name] = {}

            for m_name, m_cfg in models_grid.items():
                gc.collect()
                # 1. GroupKFold 5-fold evaluation
                fold_f1s = []
                for fold in range(5):
                    tr_mask = (folds_all != fold)
                    te_mask = (folds_all == fold)

                    scaler = StandardScaler().fit(X_feat[tr_mask])
                    X_tr = scaler.transform(X_feat[tr_mask])
                    y_tr = y_all[tr_mask]
                    X_te = scaler.transform(X_feat[te_mask])
                    y_te = y_all[te_mask]

                    best_cand = m_cfg["candidates"][0]
                    model = m_cfg["builder"](best_cand)
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_te)
                    f1_val = f1_score(y_te, preds, average="macro", zero_division=0)
                    fold_f1s.append(f1_val)

                gkf_mean = float(np.mean(fold_f1s))
                gkf_std = float(np.std(fold_f1s))

                # 2. Fixed Split (train on train, pick param on val, evaluate on test)
                scaler_fix = StandardScaler().fit(X_feat[train_idx_fixed])
                X_tr_fix = scaler_fix.transform(X_feat[train_idx_fixed])
                y_tr_fix = y_all[train_idx_fixed]
                X_val_fix = scaler_fix.transform(X_feat[val_idx_fixed])
                y_val_fix = y_all[val_idx_fixed]
                X_te_fix = scaler_fix.transform(X_feat[test_idx_fixed])
                y_te_fix = y_all[test_idx_fixed]

                best_p = None
                best_val_f1 = -1.0
                best_fix_model = None

                for cand in m_cfg["candidates"]:
                    mdl = m_cfg["builder"](cand)
                    mdl.fit(X_tr_fix, y_tr_fix)
                    v_preds = mdl.predict(X_val_fix)
                    v_f1 = f1_score(y_val_fix, v_preds, average="macro", zero_division=0)
                    if v_f1 > best_val_f1:
                        best_val_f1 = v_f1
                        best_p = cand
                        best_fix_model = mdl

                # Evaluate best model on Test (Level A)
                te_preds_a = best_fix_model.predict(X_te_fix)
                test_acc = accuracy_score(y_te_fix, te_preds_a)
                test_macro_f1 = f1_score(y_te_fix, te_preds_a, average="macro", zero_division=0)
                
                # Macro-F1 on 4 major groups (excluding phụ)
                major4_mask = y_te_fix != "phụ (supplementary)"
                macro_f1_major4 = f1_score(y_te_fix[major4_mask], te_preds_a[major4_mask], average="macro", zero_division=0)

                print(f"  {m_name:<14} | GKF: {gkf_mean:.4f} ± {gkf_std:.4f} | Test Acc: {test_acc*100:.2f}% | Test F1: {test_macro_f1:.4f} (Major4 F1: {macro_f1_major4:.4f})", flush=True)

                results_table.append({
                    "Đặc trưng": f_name,
                    "Số chiều": X_feat.shape[1],
                    "Mô hình": m_name,
                    "GroupKFold Macro-F1": f"{gkf_mean:.4f} ± {gkf_std:.4f}",
                    "GKF_mean": gkf_mean,
                    "GKF_std": gkf_std,
                    "Test Accuracy": test_acc,
                    "Test Macro-F1": test_macro_f1,
                    "Test Major4 F1": macro_f1_major4,
                    "Best Params": best_p,
                })

                detailed_metrics[f_name][m_name] = {
                    "gkf_mean": gkf_mean,
                    "gkf_std": gkf_std,
                    "gkf_folds": fold_f1s,
                    "best_params": best_p,
                    "test_acc": test_acc,
                    "test_macro_f1": test_macro_f1,
                    "test_major4_f1": macro_f1_major4,
                    "classification_report": classification_report(y_te_fix, te_preds_a, labels=canonical_groups, output_dict=True, zero_division=0),
                    "confusion_matrix": confusion_matrix(y_te_fix, te_preds_a, labels=canonical_groups).tolist(),
                    "predictions_a": te_preds_a.tolist(),
                }

                saved_models[(f_name, m_name)] = {
                    "model": best_fix_model,
                    "scaler": scaler_fix,
                }

                if f_name in ["S0 (H36+S8)", "S1 (+w/h)", "S2_2x2 (+grid 2x2)", "S3 (+HOG)"]:
                    chain_plot_data[m_name].append((gkf_mean, gkf_std))

        joblib.dump({
            "results_table": results_table,
            "chain_plot_data": chain_plot_data,
            "detailed_metrics": detailed_metrics,
            "saved_models": saved_models,
        }, eval_cache_file)
        print(f"Đã lưu cache đánh giá đặc trưng vào {eval_cache_file}", flush=True)

    # =========================================================================
    # Vẽ biểu đồ chuỗi: chain.png (Macro-F1 GroupKFold qua S0 -> S1 -> S2 -> S3)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    steps_labels = ["S0\n(H36+S8)", "S1\n(+w/h)", "S2\n(+grid)", "S3\n(+HOG)"]
    x_indices = np.arange(len(steps_labels))

    colors = {"SVM": "tab:blue", "kNN": "tab:orange", "Random Forest": "tab:green"}
    markers = {"SVM": "o-", "kNN": "s--", "Random Forest": "^-."}

    for m_name in ["SVM", "kNN", "Random Forest"]:
        means = [pt[0] for pt in chain_plot_data[m_name]]
        stds = [pt[1] for pt in chain_plot_data[m_name]]
        ax.errorbar(
            x_indices,
            means,
            yerr=stds,
            fmt=markers[m_name],
            color=colors[m_name],
            label=m_name,
            capsize=4,
            capthick=1.5,
            linewidth=2,
            markersize=7,
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(steps_labels, fontsize=10)
    ax.set_ylabel("Macro-F1 (GroupKFold 5-fold Mean ± Std)", fontsize=11)
    ax.set_title("Nâng cấp chuỗi đặc trưng Stage 2: S0 -> S1 -> S2 -> S3", fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "chain.png")
    plt.close(fig)
    print("\nĐã lưu biểu đồ chuỗi: reports/stage2_vn/chain.png", flush=True)

    # =========================================================================
    # Mức (B) - Đánh giá trên YOLO crops & Bảng so sánh A' vs B
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("ĐÁNH GIÁ MỨC (B) - CROP TỪ BBOX YOLO PHÁT HIỆN", flush=True)
    print("=" * 70, flush=True)

    yolo_model = YOLO(str(yolo_weights))
    test_img_dir = yolo_root / "images" / "test"
    test_stems = sorted(manifest[manifest["split"] == "test"]["img_stem"].unique().tolist())

    matched_gt_feats = {fn: [] for fn in feature_sets.keys()}
    matched_yolo_feats = {fn: [] for fn in feature_sets.keys()}
    matched_gt_labels = []
    matched_metas = []

    det_fp_count = 0
    det_ignored_camera = 0
    det_fn_missed = 0
    total_test_gt_excl_cam = int((manifest[(manifest["split"] == "test") & (manifest["label_fine"] != "Camera")]).shape[0])

    print(f"Chạy suy luận YOLO trên {len(test_stems)} ảnh test VN...", flush=True)

    for s in test_stems:
        stem_str = f"{int(s):04d}"
        img_p = test_img_dir / f"{stem_str}.jpg"
        if not img_p.exists():
            img_p = test_img_dir / f"{stem_str}.png"
        if not img_p.exists():
            continue

        img = cv2.imread(str(img_p))
        ih, iw = img.shape[:2]

        gt_rows = manifest[(manifest["split"] == "test") & (manifest["img_stem"] == s)]
        gt_boxes, gt_lbls, gt_fines, gt_crops = [], [], [], []
        for r in gt_rows.itertuples():
            x1 = max(0, min(iw - 1, int(r.x1)))
            y1 = max(0, min(ih - 1, int(r.y1)))
            x2 = max(x1 + 1, min(iw, int(r.x2)))
            y2 = max(y1 + 1, min(ih, int(r.y2)))
            gt_boxes.append([x1, y1, x2, y2])
            gt_lbls.append(map_stage2_group(r.label_fine))  # None if Camera
            gt_fines.append(r.label_fine)
            gt_crops.append(img[y1:y2, x1:x2])

        # YOLO predict
        res = yolo_model.predict(img_p, conf=0.25, imgsz=800, device=0, verbose=False)[0]
        pred_boxes = []
        for b in res.boxes:
            b_xyxy = b.xyxy[0].cpu().numpy().astype(int)
            px1 = max(0, min(iw - 1, int(b_xyxy[0])))
            py1 = max(0, min(ih - 1, int(b_xyxy[1])))
            px2 = max(px1 + 1, min(iw, int(b_xyxy[2])))
            py2 = max(py1 + 1, min(ih, int(b_xyxy[3])))
            pred_boxes.append([px1, py1, px2, py2])

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

                # Check if matched with Camera
                if gt_fines[best_g_idx] == "Camera":
                    det_ignored_camera += 1
                    continue

                # Crop from YOLO predicted box
                ycrop = img[p_box[1] : p_box[3], p_box[0] : p_box[2]]
                gt_c = gt_crops[best_g_idx]

                if ycrop.size > 0 and ycrop.shape[0] >= 2 and ycrop.shape[1] >= 2 and gt_c.size > 0:
                    y_s0, y_s1, y_s2_2, y_s2_3, y_hf = extract_raw_features(ycrop, p_box[2] - p_box[0], p_box[3] - p_box[1])
                    y_s2_c = y_s2_2 if chosen_s2_name == "2x2" else y_s2_3
                    y_s3 = np.concatenate([y_s2_c, y_hf])

                    gt_s0, gt_s1, gt_s2_2, gt_s2_3, gt_hf = extract_raw_features(gt_c, gt_boxes[best_g_idx][2] - gt_boxes[best_g_idx][0], gt_boxes[best_g_idx][3] - gt_boxes[best_g_idx][1])
                    gt_s2_c = gt_s2_2 if chosen_s2_name == "2x2" else gt_s2_3
                    gt_s3 = np.concatenate([gt_s2_c, gt_hf])

                    matched_yolo_feats["S0 (H36+S8)"].append(y_s0.astype(np.float32))
                    matched_yolo_feats["S1 (+w/h)"].append(y_s1.astype(np.float32))
                    matched_yolo_feats["S2_2x2 (+grid 2x2)"].append(y_s2_2.astype(np.float32))
                    matched_yolo_feats["S2_3x3 (+grid 3x3)"].append(y_s2_3.astype(np.float32))
                    matched_yolo_feats["S3 (+HOG)"].append(y_s3.astype(np.float32))

                    matched_gt_feats["S0 (H36+S8)"].append(gt_s0.astype(np.float32))
                    matched_gt_feats["S1 (+w/h)"].append(gt_s1.astype(np.float32))
                    matched_gt_feats["S2_2x2 (+grid 2x2)"].append(gt_s2_2.astype(np.float32))
                    matched_gt_feats["S2_3x3 (+grid 3x3)"].append(gt_s2_3.astype(np.float32))
                    matched_gt_feats["S3 (+HOG)"].append(gt_s3.astype(np.float32))

                    matched_gt_labels.append(gt_lbls[best_g_idx])
                    matched_metas.append({
                        "stem": stem_str,
                        "pred_box": p_box,
                        "gt_box": gt_boxes[best_g_idx],
                        "true_group": gt_lbls[best_g_idx],
                        "label_fine": gt_fines[best_g_idx],
                        "iou": round(best_iou, 3),
                    })

        # FP & FN tracking
        det_fp_count += (len(pred_boxes) - len(matched_pred_set))
        for g_idx, g_lbl in enumerate(gt_lbls):
            if g_idx not in matched_gt_set and gt_fines[g_idx] != "Camera":
                det_fn_missed += 1

    matched_gt_y = np.array(matched_gt_labels)
    n_matched = len(matched_gt_y)

    print(f"\nTổng GT test (không kể Camera): {total_test_gt_excl_cam}", flush=True)
    print(f"Số GT ghép thành công với YOLO box (IoU >= 0.5): {n_matched} ({n_matched/total_test_gt_excl_cam*100:.1f}%)", flush=True)
    print(f"Số GT bị bỏ sót (FN detector): {det_fn_missed}", flush=True)
    print(f"Số box YOLO ghép trúng Camera (ignored): {det_ignored_camera}", flush=True)
    print(f"Số box YOLO không ghép trúng GT nào (FP detector): {det_fp_count}", flush=True)

    # Evaluate best model of each feature step on Mức A' and Mức B + End-to-end Recall
    step_best_models = {
        "S0 (H36+S8)": "kNN", # or best on S0
        "S1 (+w/h)": "SVM",
        "S2_2x2 (+grid 2x2)": "SVM",
        "S3 (+HOG)": "SVM",
    }
    # Pick dynamically best model for each step based on GroupKFold
    for fn in step_best_models.keys():
        best_m = max(["SVM", "kNN", "Random Forest"], key=lambda m: detailed_metrics[fn][m]["gkf_mean"])
        step_best_models[fn] = best_m

    comparison_a_prime_b = []
    e2e_recalls = {}
    level_b_predictions = {}

    for fn, bm in step_best_models.items():
        saved_entry = saved_models[(fn, bm)]
        mdl = saved_entry["model"]
        scaler = saved_entry["scaler"]

        X_yolo_scaled = scaler.transform(np.array(matched_yolo_feats[fn]))
        X_gt_scaled = scaler.transform(np.array(matched_gt_feats[fn]))

        # Predictions
        preds_b = mdl.predict(X_yolo_scaled)
        preds_a_prime = mdl.predict(X_gt_scaled)
        level_b_predictions[(fn, bm)] = preds_b

        acc_b = accuracy_score(matched_gt_y, preds_b)
        f1_b = f1_score(matched_gt_y, preds_b, average="macro", zero_division=0)

        acc_a_prime = accuracy_score(matched_gt_y, preds_a_prime)
        f1_a_prime = f1_score(matched_gt_y, preds_a_prime, average="macro", zero_division=0)

        # End-to-end recall
        correct_signs = int(np.sum(preds_b == matched_gt_y))
        e2e_rec = correct_signs / total_test_gt_excl_cam
        e2e_recalls[fn] = {"model": bm, "correct": correct_signs, "recall": e2e_rec}

        # Level A metrics for this model from earlier
        acc_a = detailed_metrics[fn][bm]["test_acc"]
        f1_a = detailed_metrics[fn][bm]["test_macro_f1"]

        comparison_a_prime_b.append({
            "Bước đặc trưng": fn,
            "Mô hình": bm,
            "A Acc (1821)": acc_a,
            "A F1": f1_a,
            "A' Acc (1538)": acc_a_prime,
            "A' F1": f1_a_prime,
            "B Acc (1538)": acc_b,
            "B F1": f1_b,
            "End-to-End Recall": e2e_rec,
        })

    df_comp_ab = pd.DataFrame(comparison_a_prime_b)
    print("\n--- BẢNG SO SÁNH CÔNG BẰNG A vs A' vs B & END-TO-END RECALL ---", flush=True)
    print(df_comp_ab.to_string(index=False), flush=True)

    # =========================================================================
    # Cặp nhầm lẫn: Hiệu lệnh <-> Chỉ dẫn qua các bước S0..S3
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print("THEO DÕI CẶP NHẦM LẪN: HIỆU LỆNH <-> CHỈ DẪN QUA S0..S3", flush=True)
    print("=" * 70, flush=True)

    # In Level A (Fixed Test 1821)
    conf_tracking_a = []
    test_y_true = y_all[test_idx_fixed]

    for fn in ["S0 (H36+S8)", "S1 (+w/h)", "S2_2x2 (+grid 2x2)", "S3 (+HOG)"]:
        bm = step_best_models[fn]
        preds_a = np.array(detailed_metrics[fn][bm]["predictions_a"])
        # Hiệu lệnh -> Chỉ dẫn
        mand_to_info = int(np.sum((test_y_true == "hiệu lệnh (mandatory)") & (preds_a == "chỉ dẫn (information)")))
        # Chỉ dẫn -> Hiệu lệnh
        info_to_mand = int(np.sum((test_y_true == "chỉ dẫn (information)") & (preds_a == "hiệu lệnh (mandatory)")))
        conf_tracking_a.append({
            "Bước": fn,
            "Mô hình": bm,
            "Mức": "Level A (GT)",
            "Hiệu lệnh -> Chỉ dẫn": mand_to_info,
            "Chỉ dẫn -> Hiệu lệnh": info_to_mand,
            "Tổng nhầm lẫn cặp này": mand_to_info + info_to_mand,
        })

    for fn in ["S0 (H36+S8)", "S1 (+w/h)", "S2_2x2 (+grid 2x2)", "S3 (+HOG)"]:
        bm = step_best_models[fn]
        preds_b = level_b_predictions[(fn, bm)]
        mand_to_info_b = int(np.sum((matched_gt_y == "hiệu lệnh (mandatory)") & (preds_b == "chỉ dẫn (information)")))
        info_to_mand_b = int(np.sum((matched_gt_y == "chỉ dẫn (information)") & (preds_b == "hiệu lệnh (mandatory)")))
        conf_tracking_a.append({
            "Bước": fn,
            "Mô hình": bm,
            "Mức": "Level B (YOLO)",
            "Hiệu lệnh -> Chỉ dẫn": mand_to_info_b,
            "Chỉ dẫn -> Hiệu lệnh": info_to_mand_b,
            "Tổng nhầm lẫn cặp này": mand_to_info_b + info_to_mand_b,
        })

    df_conf = pd.DataFrame(conf_tracking_a)
    print(df_conf.to_string(index=False), flush=True)

    # =========================================================================
    # Ma trận nhầm lẫn so sánh S0 vs S3 (Mức A và Mức B)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
    
    # S0 Best Model
    bm_s0 = step_best_models["S0 (H36+S8)"]
    plot_confusion_matrix(test_y_true, detailed_metrics["S0 (H36+S8)"][bm_s0]["predictions_a"], canonical_groups, f"Mức A - S0 ({bm_s0})", axes[0, 0])
    plot_confusion_matrix(matched_gt_y, level_b_predictions[("S0 (H36+S8)", bm_s0)], canonical_groups, f"Mức B - S0 ({bm_s0})", axes[0, 1])

    # S3 Best Model
    bm_s3 = step_best_models["S3 (+HOG)"]
    plot_confusion_matrix(test_y_true, detailed_metrics["S3 (+HOG)"][bm_s3]["predictions_a"], canonical_groups, f"Mức A - S3 ({bm_s3})", axes[1, 0])
    plot_confusion_matrix(matched_gt_y, level_b_predictions[("S3 (+HOG)", bm_s3)], canonical_groups, f"Mức B - S3 ({bm_s3})", axes[1, 1])

    fig.suptitle("So sánh ma trận nhầm lẫn: Mốc S0 (HSV thuần) vs S3 (+w/h, grid, HOG)", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(out_dir / "cm_compare_S0_vs_S3.png")
    plt.close(fig)
    print("\nĐã lưu ma trận nhầm lẫn so sánh: reports/stage2_vn/cm_compare_S0_vs_S3.png", flush=True)

    # =========================================================================
    # Tìm và lưu 20 ảnh crop phân loại sai ở S3 và các ảnh sửa đúng từ S0 -> S3
    # =========================================================================
    preds_b_s0 = level_b_predictions[("S0 (H36+S8)", bm_s0)]
    preds_b_s3 = level_b_predictions[("S3 (+HOG)", bm_s3)]

    # 1. 20 Misclassified crops at S3 (Level B)
    s3_misc_candidates = []
    for idx, (m, y_true, y_p) in enumerate(zip(matched_metas, matched_gt_y, preds_b_s3)):
        if y_true != y_p:
            s3_misc_candidates.append({
                "index": len(s3_misc_candidates) + 1,
                "stem": m["stem"],
                "true_label": y_true,
                "pred_label": y_p,
                "label_fine": m["label_fine"],
                "box": m["pred_box"],
                "iou": m["iou"],
            })

    # Save 20 representative
    s3_misc_saved = []
    for c in s3_misc_candidates[:20]:
        img_p = test_img_dir / f"{int(c['stem']):04d}.jpg"
        if not img_p.exists():
            img_p = test_img_dir / f"{int(c['stem']):04d}.png"
        full_im = cv2.imread(str(img_p))
        b = c["box"]
        crop = full_im[b[1]:b[3], b[0]:b[2]]
        fname = f"err_s3_{c['index']:02d}_stem{c['stem']}_True_{c['true_label'].split()[0]}_Pred_{c['pred_label'].split()[0]}.jpg"
        cv2.imwrite(str(misc_dir / fname), crop)
        c_copy = dict(c)
        c_copy["file"] = fname
        c_copy["crop_size"] = f"{crop.shape[1]}x{crop.shape[0]}"
        s3_misc_saved.append(c_copy)

    # 2. 6-8 fixed samples (where S0 failed but S3 succeeded on Hiệu lệnh <-> Chỉ dẫn)
    fixed_samples = []
    for m, y_true, p_s0, p_s3 in zip(matched_metas, matched_gt_y, preds_b_s0, preds_b_s3):
        if y_true in ["hiệu lệnh (mandatory)", "chỉ dẫn (information)"]:
            if p_s0 != y_true and p_s3 == y_true:
                fixed_samples.append({
                    "stem": m["stem"],
                    "true_label": y_true,
                    "pred_s0": p_s0,
                    "pred_s3": p_s3,
                    "label_fine": m["label_fine"],
                    "box": m["pred_box"],
                    "iou": m["iou"],
                })

    fixed_saved = []
    for idx, fs in enumerate(fixed_samples[:8]):
        img_p = test_img_dir / f"{int(fs['stem']):04d}.jpg"
        if not img_p.exists():
            img_p = test_img_dir / f"{int(fs['stem']):04d}.png"
        full_im = cv2.imread(str(img_p))
        b = fs["box"]
        crop = full_im[b[1]:b[3], b[0]:b[2]]
        fname = f"fixed_{idx+1:02d}_stem{fs['stem']}_True_{fs['true_label'].split()[0]}_S0_{fs['pred_s0'].split()[0]}_S3_{fs['pred_s3'].split()[0]}.jpg"
        cv2.imwrite(str(fixed_dir / fname), crop)
        fs_copy = dict(fs)
        fs_copy["file"] = fname
        fs_copy["size"] = f"{crop.shape[1]}x{crop.shape[0]}"
        fixed_saved.append(fs_copy)

    print(f"\nĐã lưu {len(s3_misc_saved)} ảnh crop phân loại sai vào reports/stage2_vn/misclassified/", flush=True)
    print(f"Đã lưu {len(fixed_saved)} ảnh crop sửa đúng vào reports/stage2_vn/fixed_samples/", flush=True)

    # =========================================================================
    # Kiểm tra rò rỉ (leakage) nhóm cảnh báo trên nhãn mới
    # =========================================================================
    warning_test_feats = X_s0[test_idx_fixed][y_all[test_idx_fixed] == "cảnh báo (warning)"]
    warning_train_feats = X_s0[train_idx_fixed][y_all[train_idx_fixed] == "cảnh báo (warning)"]
    dists = cdist(warning_test_feats, warning_train_feats, metric="euclidean")
    min_dists = np.min(dists, axis=1)
    near_identical = int(np.sum(min_dists < 0.05))

    leakage_info = {
        "warning_test_count": len(warning_test_feats),
        "near_identical_dist_lt_0_05": near_identical,
        "mean_min_dist": float(np.mean(min_dists)),
        "median_min_dist": float(np.median(min_dists)),
    }
    print(f"\nKiểm tra trùng lặp nhóm cảnh báo: {near_identical} / {len(warning_test_feats)} crops test có khoảng cách < 0.05 tới train", flush=True)

    # =========================================================================
    # Lưu Model tốt nhất (theo GroupKFold Macro-F1)
    # =========================================================================
    best_overall_entry = max(results_table, key=lambda x: x["GKF_mean"])
    best_feat_name = best_overall_entry["Đặc trưng"]
    best_model_name = best_overall_entry["Mô hình"]
    best_gkf_score = best_overall_entry["GKF_mean"]

    final_saved_entry = saved_models[(best_feat_name, best_model_name)]
    final_model_obj = final_saved_entry["model"]
    final_scaler_obj = final_saved_entry["scaler"]

    best_model_path = models_dir / "stage2_best.joblib"
    joblib.dump({
        "model": final_model_obj,
        "scaler": final_scaler_obj,
        "feature_set": best_feat_name,
        "model_name": best_model_name,
        "canonical_groups": canonical_groups,
        "chosen_s2_grid": chosen_s2_name,
        "gkf_macro_f1": best_gkf_score,
        "seed": SEED,
    }, best_model_path)

    best_config_path = models_dir / "stage2_best_config.json"
    best_config_path.write_text(json.dumps({
        "feature_set": best_feat_name,
        "model_name": best_model_name,
        "gkf_macro_f1": f"{best_gkf_score:.4f}",
        "best_params": best_overall_entry["Best Params"],
        "dims": best_overall_entry["Số chiều"],
        "grid": chosen_s2_name,
        "seed": SEED,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nĐã lưu model tốt nhất vào: {best_model_path}", flush=True)

    # =========================================================================
    # Xuất tất cả kết quả chi tiết ra JSON
    # =========================================================================
    final_json = {
        "seed": SEED,
        "relabeling": {
            "before": table_before,
            "after": table_after,
            "small_train_dropped": num_small_dropped,
        },
        "groupkfold_integrity": {
            "total_blocks": int(valid_clf_manifest["group_id"].nunique()),
            "adjacent_pairs_diff_fold": adj_diff_fold,
        },
        "results_table": results_table,
        "s2_grid_comparison": {
            "val_f1_2x2": val_f1_2x2,
            "val_acc_2x2": val_acc_2x2,
            "val_f1_3x3": val_f1_3x3,
            "val_acc_3x3": val_acc_3x3,
            "chosen": chosen_s2_name,
        },
        "comparison_a_prime_b": comparison_a_prime_b,
        "confusion_tracking_mandatory_info": conf_tracking_a,
        "leakage_check_warning": leakage_info,
        "detector_matching": {
            "total_test_gt_excl_cam": total_test_gt_excl_cam,
            "det_matched": n_matched,
            "det_fn_missed": det_fn_missed,
            "det_ignored_camera": det_ignored_camera,
            "det_fp_count": det_fp_count,
        },
        "best_overall": {
            "feature": best_feat_name,
            "model": best_model_name,
            "gkf_macro_f1": best_gkf_score,
            "test_acc": best_overall_entry["Test Accuracy"],
            "test_macro_f1": best_overall_entry["Test Macro-F1"],
        },
        "misclassified_samples": s3_misc_saved,
        "fixed_samples": fixed_saved,
        "detailed_metrics": detailed_metrics,
    }

    metrics_json_path = out_dir / "stage2_metrics.json"
    metrics_json_path.write_text(json.dumps(final_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã lưu kết quả hoàn chỉnh vào: {metrics_json_path}", flush=True)

if __name__ == "__main__":
    main()
