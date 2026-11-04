import os
import sys
import json
import time
import psutil
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import cv2
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
np.random.seed(SEED)

def check_mem(step_name=""):
    vm = psutil.virtual_memory()
    free_gb = vm.available / 1024**3
    print(f"[{step_name}] Free RAM: {free_gb:.2f} GB", flush=True)
    if free_gb < 2.0:
        raise RuntimeError(f"CRITICAL: Free Virtual Memory < 2 GB ({free_gb:.2f} GB)! Aborting as requested.")
    return free_gb

def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    area1 = max(0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - interArea
    return interArea / float(union) if union > 0 else 0.0

def main():
    check_mem("Khởi động")
    
    out_dir = Path("reports/stage2_vn")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    canonical_groups = [
        "cấm (prohibitory)",
        "cảnh báo (warning)",
        "hiệu lệnh (mandatory)",
        "chỉ dẫn (information)",
        "phụ (supplementary)",
    ]
    g2i = {g: i for i, g in enumerate(canonical_groups)}
    i2g = {i: g for i, g in enumerate(canonical_groups)}
    major4_groups = canonical_groups[:4]

    # 1. Load feature cache
    feat_cache_file = Path("data/stage2_feat_cache.joblib")
    if not feat_cache_file.exists():
        raise FileNotFoundError(f"Missing {feat_cache_file}!")
    
    print(f"Đang tải {feat_cache_file}...", flush=True)
    feat_data = joblib.load(feat_cache_file)
    X_s0 = feat_data["X_s0"]
    X_s1 = feat_data["X_s1"]
    X_s2_2x2 = feat_data["X_s2_2x2"]
    X_s2_3x3 = feat_data["X_s2_3x3"]
    X_hog = feat_data["X_hog"]
    y_all_str = feat_data["y_all"]
    splits_all = feat_data["splits_all"]
    folds_all = feat_data["folds_all"]
    meta_list = feat_data["meta_list"]

    y_all_int = np.array([g2i[g] for g in y_all_str], dtype=np.int32)

    X_s3 = np.concatenate([X_s2_2x2, X_hog], axis=1)

    feature_sets = {
        "S0 (H36+S8)": X_s0,
        "S1 (+w/h)": X_s1,
        "S2_2x2 (+grid 2x2)": X_s2_2x2,
        "S2_3x3 (+grid 3x3)": X_s2_3x3,
        "S3 (+HOG)": X_s3,
    }

    train_idx_fixed = np.where(splits_all == "train")[0]
    val_idx_fixed = np.where(splits_all == "val")[0]
    test_idx_fixed = np.where(splits_all == "test")[0]

    y_test_str = y_all_str[test_idx_fixed]
    y_test_int = y_all_int[test_idx_fixed]

    # 2. Load existing eval cache for SVM, kNN, RF
    eval_cache_file = Path("data/stage2_eval_cache.joblib")
    existing_eval = joblib.load(eval_cache_file)
    results_table = existing_eval["results_table"]
    chain_plot_data = existing_eval["chain_plot_data"]
    detailed_metrics = existing_eval["detailed_metrics"]
    saved_models = existing_eval["saved_models"]

    # Fix Major4 F1 for existing models
    print("\n" + "=" * 70, flush=True)
    print("SỬA LỖI SỐ LIỆU: TÍNH LẠI MAJOR4 F1 CHO SVM, KNN, RANDOM FOREST", flush=True)
    print("=" * 70, flush=True)

    for entry in results_table:
        fn = entry["Đặc trưng"]
        mn = entry["Mô hình"]
        preds_str = np.array(detailed_metrics[fn][mn]["predictions_a"])
        
        # Calculate correct Major4 F1 (average F1 of 4 major classes)
        cr = detailed_metrics[fn][mn]["classification_report"]
        f1_major4_vals = [cr[g]["f1-score"] for g in major4_groups]
        correct_major4 = float(np.mean(f1_major4_vals))
        
        entry["Test Major4 F1"] = correct_major4
        detailed_metrics[fn][mn]["test_major4_f1"] = correct_major4
        print(f"  {fn:<20} | {mn:<14} | Macro-F1: {entry['Test Macro-F1']:.4f} | Correct Major4 F1: {correct_major4:.4f}", flush=True)

    # 3. Boosting Grid Definitions
    boosting_grid = [
        {"n_estimators": 200, "max_depth": 4},
        {"n_estimators": 400, "max_depth": 4},
        {"n_estimators": 200, "max_depth": 6},
        {"n_estimators": 400, "max_depth": 6},
    ]

    boosting_models = ["XGBoost", "LightGBM"]
    for bm in boosting_models:
        if bm not in chain_plot_data:
            chain_plot_data[bm] = []

    # 4. Run XGBoost & LightGBM across all feature steps
    print("\n" + "=" * 70, flush=True)
    print("CHẠY ĐÁNH GIÁ XGBOOST & LIGHTGBM TRÊN CÁC NẤC ĐẶC TRƯNG", flush=True)
    print("=" * 70, flush=True)

    update_cache_file = Path("data/stage2_boosting_cache.joblib")
    if update_cache_file.exists():
        print(f"Tải kết quả Boosting từ cache: {update_cache_file}", flush=True)
        boosting_results = joblib.load(update_cache_file)
        boosting_table = boosting_results["boosting_table"]
        boosting_detailed = boosting_results["boosting_detailed"]
        boosting_saved = boosting_results["boosting_saved"]
        chain_plot_data = boosting_results["chain_plot_data"]
    else:
        boosting_table = []
        boosting_detailed = {fn: {} for fn in feature_sets}
        boosting_saved = {}

        for f_name, X_feat in feature_sets.items():
            print(f"\n--- Đặc trưng: {f_name} (Số chiều: {X_feat.shape[1]}) ---", flush=True)
            check_mem(f_name)

            for m_name in boosting_models:
                print(f"  Đang huấn luyện mô hình: {m_name}...", flush=True)
                
                # A. GroupKFold 5-fold evaluation with inner validation
                fold_f1s = []
                for fold in range(5):
                    tr_outer_mask = (folds_all != fold)
                    te_outer_mask = (folds_all == fold)

                    X_tr_out = X_feat[tr_outer_mask]
                    y_tr_out = y_all_int[tr_outer_mask]
                    X_te_out = X_feat[te_outer_mask]
                    y_te_out = y_all_int[te_outer_mask]
                    folds_inner = folds_all[tr_outer_mask]

                    # Inner split: pick one fold from folds_inner as inner val
                    unique_inner_folds = np.unique(folds_inner)
                    inner_val_fold = unique_inner_folds[0]
                    inner_val_mask = (folds_inner == inner_val_fold)
                    inner_tr_mask = (folds_inner != inner_val_fold)

                    X_in_tr = X_tr_out[inner_tr_mask]
                    y_in_tr = y_tr_out[inner_tr_mask]
                    X_in_val = X_tr_out[inner_val_mask]
                    y_in_val = y_tr_out[inner_val_mask]

                    sw_in_tr = compute_sample_weight("balanced", y_in_tr)

                    best_cand = None
                    best_cand_f1 = -1.0

                    # Evaluate depth=4 and depth=6
                    for depth in [4, 6]:
                        if m_name == "XGBoost":
                            m_inner = xgb.XGBClassifier(
                                n_estimators=400,
                                max_depth=depth,
                                learning_rate=0.1,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                tree_method="hist",
                                device="cpu",
                                n_jobs=4,
                                random_state=SEED,
                                objective="multi:softprob",
                                num_class=5,
                            )
                            m_inner.fit(X_in_tr, y_in_tr, sample_weight=sw_in_tr)
                            for n_est in [200, 400]:
                                preds_val = m_inner.predict(X_in_val, iteration_range=(0, n_est))
                                val_f1 = f1_score(y_in_val, preds_val, labels=list(range(5)), average="macro", zero_division=0)
                                if val_f1 > best_cand_f1:
                                    best_cand_f1 = val_f1
                                    best_cand = {"n_estimators": n_est, "max_depth": depth}
                        else: # LightGBM
                            m_inner = lgb.LGBMClassifier(
                                n_estimators=400,
                                max_depth=depth,
                                learning_rate=0.1,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                subsample_freq=1,
                                n_jobs=4,
                                random_state=SEED,
                                objective="multiclass",
                                num_class=5,
                                verbose=-1,
                            )
                            m_inner.fit(X_in_tr, y_in_tr, sample_weight=sw_in_tr)
                            for n_est in [200, 400]:
                                preds_val = m_inner.predict(X_in_val, num_iteration=n_est)
                                val_f1 = f1_score(y_in_val, preds_val, labels=list(range(5)), average="macro", zero_division=0)
                                if val_f1 > best_cand_f1:
                                    best_cand_f1 = val_f1
                                    best_cand = {"n_estimators": n_est, "max_depth": depth}

                    # Fit outer model with best_cand
                    sw_tr_out = compute_sample_weight("balanced", y_tr_out)
                    if m_name == "XGBoost":
                        m_outer = xgb.XGBClassifier(
                            n_estimators=best_cand["n_estimators"],
                            max_depth=best_cand["max_depth"],
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            tree_method="hist",
                            device="cpu",
                            n_jobs=4,
                            random_state=SEED,
                            objective="multi:softprob",
                            num_class=5,
                        )
                        m_outer.fit(X_tr_out, y_tr_out, sample_weight=sw_tr_out)
                    else:
                        m_outer = lgb.LGBMClassifier(
                            n_estimators=best_cand["n_estimators"],
                            max_depth=best_cand["max_depth"],
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            subsample_freq=1,
                            n_jobs=4,
                            random_state=SEED,
                            objective="multiclass",
                            num_class=5,
                            verbose=-1,
                        )
                        m_outer.fit(X_tr_out, y_tr_out, sample_weight=sw_tr_out)

                    preds_te_out = m_outer.predict(X_te_out)
                    fold_f1 = f1_score(y_te_out, preds_te_out, labels=list(range(5)), average="macro", zero_division=0)
                    fold_f1s.append(fold_f1)

                gkf_mean = float(np.mean(fold_f1s))
                gkf_std = float(np.std(fold_f1s))

                # B. Fixed Split evaluation (train on train, select on val, evaluate on test)
                X_tr_fix = X_feat[train_idx_fixed]
                y_tr_fix = y_all_int[train_idx_fixed]
                X_val_fix = X_feat[val_idx_fixed]
                y_val_fix = y_all_int[val_idx_fixed]
                X_te_fix = X_feat[test_idx_fixed]
                y_te_fix = y_all_int[test_idx_fixed]

                sw_tr_fix = compute_sample_weight("balanced", y_tr_fix)

                best_fix_cand = None
                best_fix_val_f1 = -1.0
                best_fix_model = None

                for depth in [4, 6]:
                    if m_name == "XGBoost":
                        m_fixed = xgb.XGBClassifier(
                            n_estimators=400,
                            max_depth=depth,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            tree_method="hist",
                            device="cpu",
                            n_jobs=4,
                            random_state=SEED,
                            objective="multi:softprob",
                            num_class=5,
                        )
                        m_fixed.fit(X_tr_fix, y_tr_fix, sample_weight=sw_tr_fix)
                        for n_est in [200, 400]:
                            v_preds = m_fixed.predict(X_val_fix, iteration_range=(0, n_est))
                            v_f1 = f1_score(y_val_fix, v_preds, labels=list(range(5)), average="macro", zero_division=0)
                            if v_f1 > best_fix_val_f1:
                                best_fix_val_f1 = v_f1
                                best_fix_cand = {"n_estimators": n_est, "max_depth": depth}
                                best_fix_model = m_fixed
                    else:
                        m_fixed = lgb.LGBMClassifier(
                            n_estimators=400,
                            max_depth=depth,
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            subsample_freq=1,
                            n_jobs=4,
                            random_state=SEED,
                            objective="multiclass",
                            num_class=5,
                            verbose=-1,
                        )
                        m_fixed.fit(X_tr_fix, y_tr_fix, sample_weight=sw_tr_fix)
                        for n_est in [200, 400]:
                            v_preds = m_fixed.predict(X_val_fix, num_iteration=n_est)
                            v_f1 = f1_score(y_val_fix, v_preds, labels=list(range(5)), average="macro", zero_division=0)
                            if v_f1 > best_fix_val_f1:
                                best_fix_val_f1 = v_f1
                                best_fix_cand = {"n_estimators": n_est, "max_depth": depth}
                                best_fix_model = m_fixed

                # Retrain or predict test with best_fix_cand
                if m_name == "XGBoost":
                    te_preds_int = best_fix_model.predict(X_te_fix, iteration_range=(0, best_fix_cand["n_estimators"]))
                else:
                    te_preds_int = best_fix_model.predict(X_te_fix, num_iteration=best_fix_cand["n_estimators"])

                te_preds_str = [i2g[i] for i in te_preds_int]
                test_acc = float(accuracy_score(y_test_int, te_preds_int))
                test_macro_f1 = float(f1_score(y_test_int, te_preds_int, labels=list(range(5)), average="macro", zero_division=0))

                cr_dict = classification_report(y_test_str, te_preds_str, labels=canonical_groups, output_dict=True, zero_division=0)
                f1_major4_vals = [cr_dict[g]["f1-score"] for g in major4_groups]
                correct_major4 = float(np.mean(f1_major4_vals))
                cm = confusion_matrix(y_test_str, te_preds_str, labels=canonical_groups).tolist()

                print(f"    {m_name:<10} | GKF: {gkf_mean:.4f} ± {gkf_std:.4f} | Test Acc: {test_acc*100:.2f}% | Test F1: {test_macro_f1:.4f} | Major4 F1: {correct_major4:.4f}", flush=True)

                entry = {
                    "Đặc trưng": f_name,
                    "Số chiều": X_feat.shape[1],
                    "Mô hình": m_name,
                    "GroupKFold Macro-F1": f"{gkf_mean:.4f} ± {gkf_std:.4f}",
                    "GKF_mean": gkf_mean,
                    "GKF_std": gkf_std,
                    "Test Accuracy": test_acc,
                    "Test Macro-F1": test_macro_f1,
                    "Test Major4 F1": correct_major4,
                    "Best Params": best_fix_cand,
                }
                boosting_table.append(entry)
                boosting_detailed[f_name][m_name] = {
                    "gkf_mean": gkf_mean,
                    "gkf_std": gkf_std,
                    "gkf_folds": fold_f1s,
                    "best_params": best_fix_cand,
                    "test_acc": test_acc,
                    "test_macro_f1": test_macro_f1,
                    "test_major4_f1": correct_major4,
                    "classification_report": cr_dict,
                    "confusion_matrix": cm,
                    "predictions_a": te_preds_str,
                }
                boosting_saved[(f_name, m_name)] = {
                    "model": best_fix_model,
                    "best_params": best_fix_cand,
                }

                if f_name in ["S0 (H36+S8)", "S1 (+w/h)", "S2_2x2 (+grid 2x2)", "S3 (+HOG)"]:
                    chain_plot_data[m_name].append((gkf_mean, gkf_std))

        joblib.dump({
            "boosting_table": boosting_table,
            "boosting_detailed": boosting_detailed,
            "boosting_saved": boosting_saved,
            "chain_plot_data": chain_plot_data,
        }, update_cache_file)
        print(f"Đã lưu cache Boosting vào {update_cache_file}", flush=True)

    # Combine tables
    all_results_table = []
    # Merge existing results (SVM, kNN, RF) and boosting results (XGB, LGB)
    for f_name in feature_sets.keys():
        for m_name in ["SVM", "kNN", "Random Forest"]:
            match = [r for r in results_table if r["Đặc trưng"] == f_name and r["Mô hình"] == m_name]
            if match:
                all_results_table.append(match[0])
        for m_name in boosting_models:
            match = [r for r in boosting_table if r["Đặc trưng"] == f_name and r["Mô hình"] == m_name]
            if match:
                all_results_table.append(match[0])

    # 5. Redraw chain.png with 5 curves
    print("\n" + "=" * 70, flush=True)
    print("VẼ LẠI BIỂU ĐỒ CHUỖI CHAIN.PNG VỚI 5 MÔ HÌNH", flush=True)
    print("=" * 70, flush=True)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    steps_labels = ["S0\n(H36+S8)", "S1\n(+w/h)", "S2\n(+grid)", "S3\n(+HOG)"]
    x_indices = np.arange(len(steps_labels))

    colors = {
        "SVM": "tab:blue",
        "kNN": "tab:orange",
        "Random Forest": "tab:green",
        "XGBoost": "tab:red",
        "LightGBM": "tab:purple",
    }
    markers = {
        "SVM": "o-",
        "kNN": "s--",
        "Random Forest": "^-.",
        "XGBoost": "D-",
        "LightGBM": "v--",
    }

    for m_name in ["SVM", "kNN", "Random Forest", "XGBoost", "LightGBM"]:
        means = [pt[0] for pt in chain_plot_data[m_name]]
        stds = [pt[1] for pt in chain_plot_data[m_name]]
        ax.errorbar(
            x_indices,
            means,
            yerr=stds,
            fmt=markers[m_name],
            color=colors[m_name],
            capsize=4,
            capthick=1.5,
            label=f"{m_name} (S3: {means[-1]:.4f})",
            linewidth=2,
            markersize=6,
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels(steps_labels, fontsize=10)
    ax.set_xlabel("Nấc đặc trưng nâng cấp", fontsize=11, fontweight="bold")
    ax.set_ylabel("GroupKFold Macro-F1 (5 folds)", fontsize=11, fontweight="bold")
    ax.set_title("Tiến triển GroupKFold Macro-F1 qua các nấc đặc trưng (5 mô hình)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.55, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    fig.tight_layout()

    chain_plot_path = out_dir / "chain.png"
    fig.savefig(str(chain_plot_path))
    plt.close(fig)
    print(f"Đã lưu biểu đồ chuỗi mới: {chain_plot_path}", flush=True)

    # 6. Evaluate Level B for S3 XGBoost and LightGBM
    print("\n" + "=" * 70, flush=True)
    print("ĐÁNH GIÁ MỨC (B) & END-TO-END CHO S3 XGBOOST & LIGHTGBM", flush=True)
    print("=" * 70, flush=True)

    yolo_cache_file = Path("data/stage2_yolo_matched_cache.joblib")
    if yolo_cache_file.exists():
        print(f"Tải cache ghép cặp YOLO: {yolo_cache_file}", flush=True)
        yolo_cache = joblib.load(yolo_cache_file)
        matched_yolo_feats = yolo_cache["matched_yolo_feats"]
        matched_gt_feats = yolo_cache["matched_gt_feats"]
        matched_gt_labels = yolo_cache["matched_gt_labels"]
        matched_metas = yolo_cache["matched_metas"]
        det_fp_count = yolo_cache["det_fp_count"]
        det_fn_missed = yolo_cache["det_fn_missed"]
        det_ignored_camera = yolo_cache["det_ignored_camera"]
        total_test_gt_excl_cam = yolo_cache["total_test_gt_excl_cam"]
    else:
        # Run YOLO inference
        print("Chạy suy luận YOLO trên 550 ảnh test để ghép cặp...", flush=True)
        yolo_weights = Path("runs/detect/vn_s/weights/best.pt")
        yolo_model = YOLO(str(yolo_weights))
        test_img_dir = Path("data/vn/yolo/images/test")
        manifest_path = Path("data/vn/manifest.csv")
        manifest = pd.read_csv(manifest_path)

        def map_stage2_group(label_fine: str):
            if label_fine == "Camera":
                return None
            if label_fine == "B.8a":
                return "cấm (prohibitory)"
            s1_prefix_to_group = {
                "P": "cấm (prohibitory)",
                "R": "hiệu lệnh (mandatory)",
                "W": "cảnh báo (warning)",
                "I": "chỉ dẫn (information)",
                "S": "phụ (supplementary)",
            }
            return s1_prefix_to_group.get(label_fine.split(".")[0], "chưa xác định")

        def extract_raw_features(crop_bgr, orig_w, orig_h):
            from skimage.feature import hog
            c32 = cv2.resize(crop_bgr, (32, 32), interpolation=cv2.INTER_AREA)
            hsv32 = cv2.cvtColor(c32, cv2.COLOR_BGR2HSV)
            h32 = ((hsv32[:, :, 0].astype(np.int32) + 30) % 180).astype(np.uint8)
            s32 = hsv32[:, :, 1]
            hist_h = cv2.calcHist([h32], [0], None, [36], [0, 180]).flatten()
            hist_s = cv2.calcHist([s32], [0], None, [8], [0, 256]).flatten()
            s0 = np.concatenate([hist_h, hist_s])
            s0 = s0 / (np.sum(s0) + 1e-7)

            ar = float(orig_w) / float(max(orig_h, 1.0))
            s1 = np.append(s0, ar)

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
            s2_2x2 = np.concatenate([s1, np.concatenate(cells_2x2)])

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
            s2_3x3 = np.concatenate([s1, np.concatenate(cells_3x3)])

            c64 = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
            gray64 = cv2.cvtColor(c64, cv2.COLOR_BGR2GRAY)
            hog_feat = hog(
                gray64,
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                feature_vector=True,
            )
            return s0, s1, s2_2x2, s2_3x3, hog_feat

        test_stems = sorted([int(p.stem) for p in test_img_dir.glob("*.jpg")])
        test_gt = manifest[manifest["split"] == "test"].copy()

        matched_yolo_feats = {fn: [] for fn in feature_sets}
        matched_gt_feats = {fn: [] for fn in feature_sets}
        matched_gt_labels = []
        matched_metas = []

        det_fp_count = 0
        det_fn_missed = 0
        det_ignored_camera = 0
        total_test_gt_excl_cam = 0

        for stem in test_stems:
            stem_str = f"{stem:04d}"
            img_p = test_img_dir / f"{stem_str}.jpg"
            if not img_p.exists():
                img_p = test_img_dir / f"{stem_str}.png"
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            sub_gt = test_gt[test_gt["img_stem"] == stem]
            gt_boxes = []
            gt_lbls = []
            gt_fines = []
            gt_crops = []

            for r in sub_gt.itertuples():
                if r.label_fine == "Camera":
                    continue
                total_test_gt_excl_cam += 1
                x1 = max(0, min(iw - 1, int(r.x1)))
                y1 = max(0, min(ih - 1, int(r.y1)))
                x2 = max(x1 + 1, min(iw, int(r.x2)))
                y2 = max(y1 + 1, min(ih, int(r.y2)))
                gt_boxes.append([x1, y1, x2, y2])
                gt_lbls.append(map_stage2_group(r.label_fine))
                gt_fines.append(r.label_fine)
                gt_crops.append(img[y1:y2, x1:x2])

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

                    ycrop = img[p_box[1]:p_box[3], p_box[0]:p_box[2]]
                    gt_c = gt_crops[best_g_idx]

                    if ycrop.size > 0 and ycrop.shape[0] >= 2 and ycrop.shape[1] >= 2 and gt_c.size > 0:
                        y_s0, y_s1, y_s2_2, y_s2_3, y_hf = extract_raw_features(ycrop, p_box[2] - p_box[0], p_box[3] - p_box[1])
                        y_s3 = np.concatenate([y_s2_2, y_hf])

                        gt_s0, gt_s1, gt_s2_2, gt_s2_3, gt_hf = extract_raw_features(gt_c, gt_boxes[best_g_idx][2] - gt_boxes[best_g_idx][0], gt_boxes[best_g_idx][3] - gt_boxes[best_g_idx][1])
                        gt_s3 = np.concatenate([gt_s2_2, gt_hf])

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

            det_fp_count += (len(pred_boxes) - len(matched_pred_set))
            for g_idx, g_lbl in enumerate(gt_lbls):
                if g_idx not in matched_gt_set:
                    det_fn_missed += 1

        joblib.dump({
            "matched_yolo_feats": matched_yolo_feats,
            "matched_gt_feats": matched_gt_feats,
            "matched_gt_labels": matched_gt_labels,
            "matched_metas": matched_metas,
            "det_fp_count": det_fp_count,
            "det_fn_missed": det_fn_missed,
            "det_ignored_camera": det_ignored_camera,
            "total_test_gt_excl_cam": total_test_gt_excl_cam,
        }, yolo_cache_file)
        print(f"Đã lưu cache ghép cặp YOLO: {yolo_cache_file}", flush=True)

    matched_gt_y = np.array(matched_gt_labels)
    n_matched = len(matched_gt_y)
    print(f"Số GT ghép thành công: {n_matched} / {total_test_gt_excl_cam} ({n_matched/total_test_gt_excl_cam*100:.1f}%)", flush=True)

    # Convert matched features to arrays
    m_gt_s3 = np.array(matched_gt_feats["S3 (+HOG)"], dtype=np.float32)
    m_yolo_s3 = np.array(matched_yolo_feats["S3 (+HOG)"], dtype=np.float32)

    # Evaluate Level A, A', B, and E2E Recall for all S3 models
    s3_models_eval = {}
    for m_name in ["SVM", "kNN", "Random Forest"]:
        saved_entry = saved_models[("S3 (+HOG)", m_name)]
        mdl = saved_entry["model"]
        scaler = saved_entry["scaler"]

        p_a_prime = mdl.predict(scaler.transform(m_gt_s3))
        p_b = mdl.predict(scaler.transform(m_yolo_s3))

        a_prime_acc = float(accuracy_score(matched_gt_y, p_a_prime))
        a_prime_f1 = float(f1_score(matched_gt_y, p_a_prime, labels=canonical_groups, average="macro", zero_division=0))
        b_acc = float(accuracy_score(matched_gt_y, p_b))
        b_f1 = float(f1_score(matched_gt_y, p_b, labels=canonical_groups, average="macro", zero_division=0))
        e2e_recall = float((p_b == matched_gt_y).sum() / total_test_gt_excl_cam)

        s3_models_eval[m_name] = {
            "a_prime_acc": a_prime_acc,
            "a_prime_f1": a_prime_f1,
            "b_acc": b_acc,
            "b_f1": b_f1,
            "e2e_recall": e2e_recall,
            "preds_b": p_b.tolist(),
        }

    for m_name in boosting_models:
        saved_entry = boosting_saved[("S3 (+HOG)", m_name)]
        mdl = saved_entry["model"]
        params = saved_entry["best_params"]

        if m_name == "XGBoost":
            p_a_prime_int = mdl.predict(m_gt_s3, iteration_range=(0, params["n_estimators"]))
            p_b_int = mdl.predict(m_yolo_s3, iteration_range=(0, params["n_estimators"]))
        else:
            p_a_prime_int = mdl.predict(m_gt_s3, num_iteration=params["n_estimators"])
            p_b_int = mdl.predict(m_yolo_s3, num_iteration=params["n_estimators"])

        p_a_prime = np.array([i2g[i] for i in p_a_prime_int])
        p_b = np.array([i2g[i] for i in p_b_int])

        a_prime_acc = float(accuracy_score(matched_gt_y, p_a_prime))
        a_prime_f1 = float(f1_score(matched_gt_y, p_a_prime, labels=canonical_groups, average="macro", zero_division=0))
        b_acc = float(accuracy_score(matched_gt_y, p_b))
        b_f1 = float(f1_score(matched_gt_y, p_b, labels=canonical_groups, average="macro", zero_division=0))
        e2e_recall = float((p_b == matched_gt_y).sum() / total_test_gt_excl_cam)

        s3_models_eval[m_name] = {
            "a_prime_acc": a_prime_acc,
            "a_prime_f1": a_prime_f1,
            "b_acc": b_acc,
            "b_f1": b_f1,
            "e2e_recall": e2e_recall,
            "preds_b": p_b.tolist(),
        }

    # 7. Confusion matrix Hiệu lệnh <-> Chỉ dẫn for ALL models at all stages
    print("\n" + "=" * 70, flush=True)
    print("TÍNH TOÁN BẢNG NHẦM LẪN HIỆU LỆNH <-> CHỈ DẪN CHO TẤT CẢ MÔ HÌNH", flush=True)
    print("=" * 70, flush=True)

    confusion_all_models = []
    # Level A: y_test_str (1821)
    for f_name in feature_sets.keys():
        for m_name in ["SVM", "kNN", "Random Forest"]:
            preds_a = np.array(detailed_metrics[f_name][m_name]["predictions_a"])
            h_to_i = int(((y_test_str == "hiệu lệnh (mandatory)") & (preds_a == "chỉ dẫn (information)")).sum())
            i_to_h = int(((y_test_str == "chỉ dẫn (information)") & (preds_a == "hiệu lệnh (mandatory)")).sum())
            confusion_all_models.append({
                "Nấc đặc trưng": f_name,
                "Mô hình": m_name,
                "Mức": "Level A (GT)",
                "Hiệu lệnh -> Chỉ dẫn": h_to_i,
                "Chỉ dẫn -> Hiệu lệnh": i_to_h,
                "Tổng nhầm": h_to_i + i_to_h,
            })
        for m_name in boosting_models:
            preds_a = np.array(boosting_detailed[f_name][m_name]["predictions_a"])
            h_to_i = int(((y_test_str == "hiệu lệnh (mandatory)") & (preds_a == "chỉ dẫn (information)")).sum())
            i_to_h = int(((y_test_str == "chỉ dẫn (information)") & (preds_a == "hiệu lệnh (mandatory)")).sum())
            confusion_all_models.append({
                "Nấc đặc trưng": f_name,
                "Mô hình": m_name,
                "Mức": "Level A (GT)",
                "Hiệu lệnh -> Chỉ dẫn": h_to_i,
                "Chỉ dẫn -> Hiệu lệnh": i_to_h,
                "Tổng nhầm": h_to_i + i_to_h,
            })

    # Level B for S3 models
    for m_name, s3_res in s3_models_eval.items():
        preds_b = np.array(s3_res["preds_b"])
        h_to_i = int(((matched_gt_y == "hiệu lệnh (mandatory)") & (preds_b == "chỉ dẫn (information)")).sum())
        i_to_h = int(((matched_gt_y == "chỉ dẫn (information)") & (preds_b == "hiệu lệnh (mandatory)")).sum())
        confusion_all_models.append({
            "Nấc đặc trưng": "S3 (+HOG)",
            "Mô hình": m_name,
            "Mức": "Level B (YOLO)",
            "Hiệu lệnh -> Chỉ dẫn": h_to_i,
            "Chỉ dẫn -> Hiệu lệnh": i_to_h,
            "Tổng nhầm": h_to_i + i_to_h,
        })

    # 8. CPU Inference Speed Benchmark (1000 crops of S3)
    print("\n" + "=" * 70, flush=True)
    print("ĐO TỐC ĐỘ SUY LUẬN CPU (MS/CROP TRÊN 1000 MẪU S3)", flush=True)
    print("=" * 70, flush=True)

    X_bench = X_s3[test_idx_fixed][:1000]
    speed_benchmark = {}

    # Scaler transform for SVM and RF
    scaler_s3 = saved_models[("S3 (+HOG)", "SVM")]["scaler"]
    X_bench_scaled = scaler_s3.transform(X_bench)

    # SVM
    svm_mdl = saved_models[("S3 (+HOG)", "SVM")]["model"]
    _ = svm_mdl.predict(X_bench_scaled[:50]) # warmup
    t0 = time.perf_counter()
    _ = svm_mdl.predict(X_bench_scaled)
    t_svm = (time.perf_counter() - t0) * 1000.0 / len(X_bench)
    speed_benchmark["SVM"] = round(t_svm, 3)

    # RF
    rf_mdl = saved_models[("S3 (+HOG)", "Random Forest")]["model"]
    _ = rf_mdl.predict(X_bench_scaled[:50])
    t0 = time.perf_counter()
    _ = rf_mdl.predict(X_bench_scaled)
    t_rf = (time.perf_counter() - t0) * 1000.0 / len(X_bench)
    speed_benchmark["Random Forest"] = round(t_rf, 3)

    # XGBoost
    xgb_mdl = boosting_saved[("S3 (+HOG)", "XGBoost")]["model"]
    xgb_p = boosting_saved[("S3 (+HOG)", "XGBoost")]["best_params"]
    _ = xgb_mdl.predict(X_bench[:50], iteration_range=(0, xgb_p["n_estimators"]))
    t0 = time.perf_counter()
    _ = xgb_mdl.predict(X_bench, iteration_range=(0, xgb_p["n_estimators"]))
    t_xgb = (time.perf_counter() - t0) * 1000.0 / len(X_bench)
    speed_benchmark["XGBoost"] = round(t_xgb, 3)

    # LightGBM
    lgb_mdl = boosting_saved[("S3 (+HOG)", "LightGBM")]["model"]
    lgb_p = boosting_saved[("S3 (+HOG)", "LightGBM")]["best_params"]
    _ = lgb_mdl.predict(X_bench[:50], num_iteration=lgb_p["n_estimators"])
    t0 = time.perf_counter()
    _ = lgb_mdl.predict(X_bench, num_iteration=lgb_p["n_estimators"])
    t_lgb = (time.perf_counter() - t0) * 1000.0 / len(X_bench)
    speed_benchmark["LightGBM"] = round(t_lgb, 3)

    for m, s in speed_benchmark.items():
        print(f"  {m:<14}: {s:.3f} ms / crop", flush=True)

    # 9. Statistical comparison at S3
    print("\n" + "=" * 70, flush=True)
    print("SO SÁNH THỐNG KÊ PAIRED THEO FOLD Ở NẤC S3", flush=True)
    print("=" * 70, flush=True)

    s3_all_models = ["SVM", "kNN", "Random Forest", "XGBoost", "LightGBM"]
    s3_folds_dict = {}
    for m in ["SVM", "kNN", "Random Forest"]:
        s3_folds_dict[m] = detailed_metrics["S3 (+HOG)"][m]["gkf_folds"]
    for m in boosting_models:
        s3_folds_dict[m] = boosting_detailed["S3 (+HOG)"][m]["gkf_folds"]

    best_s3_model_name = max(s3_all_models, key=lambda m: np.mean(s3_folds_dict[m]))
    best_s3_mean = np.mean(s3_folds_dict[best_s3_model_name])
    best_s3_std = np.std(s3_folds_dict[best_s3_model_name])
    print(f"Mô hình tốt nhất ở S3 theo GroupKFold: {best_s3_model_name} (Mean={best_s3_mean:.4f} ± {best_s3_std:.4f})", flush=True)

    paired_comparison = []
    for m in s3_all_models:
        diffs = np.array(s3_folds_dict[best_s3_model_name]) - np.array(s3_folds_dict[m])
        d_mean = float(np.mean(diffs))
        d_std = float(np.std(diffs))
        is_indistinguishable = (d_mean < d_std) if m != best_s3_model_name else False
        status = "Baseline (Tốt nhất)" if m == best_s3_model_name else ("chưa phân biệt được" if is_indistinguishable else "khác biệt rõ rệt")
        paired_comparison.append({
            "Mô hình": m,
            "Fold 0": round(s3_folds_dict[m][0], 4),
            "Fold 1": round(s3_folds_dict[m][1], 4),
            "Fold 2": round(s3_folds_dict[m][2], 4),
            "Fold 3": round(s3_folds_dict[m][3], 4),
            "Fold 4": round(s3_folds_dict[m][4], 4),
            "Mean F1": round(float(np.mean(s3_folds_dict[m])), 4),
            "Std F1": round(float(np.std(s3_folds_dict[m])), 4),
            "Diff vs Best (Mean ± Std)": f"{d_mean:+.4f} ± {d_std:.4f}",
            "Kết luận thống kê": status,
        })
        print(f"  {m:<14}: Mean={np.mean(s3_folds_dict[m]):.4f} | Diff={d_mean:+.4f} ± {d_std:.4f} -> {status}", flush=True)

    # If new best model beats SVM, save models/stage2_best_v2.joblib
    if best_s3_model_name != "SVM":
        print(f"\nModel {best_s3_model_name} vượt qua SVM! Lưu models/stage2_best_v2.joblib...", flush=True)
        best_v2_obj = boosting_saved[("S3 (+HOG)", best_s3_model_name)]
        joblib.dump(best_v2_obj, models_dir / "stage2_best_v2.joblib")
        (models_dir / "stage2_best_v2_config.json").write_text(json.dumps({
            "model_name": best_s3_model_name,
            "feature_set": "S3 (+HOG)",
            "gkf_macro_f1": f"{best_s3_mean:.4f}",
            "params": best_v2_obj["best_params"],
            "seed": SEED,
        }, indent=2), encoding="utf-8")
    else:
        print("\nSVM vẫn giữ vị trí số 1 theo GroupKFold Macro-F1 (0.8983). Giữ nguyên models/stage2_best.joblib.", flush=True)

    # 10. Save comprehensive metrics JSON
    final_update_json = {
        "seed": SEED,
        "all_results_table": all_results_table,
        "chain_plot_data": chain_plot_data,
        "speed_benchmark_ms": speed_benchmark,
        "s3_models_eval": s3_models_eval,
        "paired_comparison": paired_comparison,
        "confusion_all_models": confusion_all_models,
        "best_model_s3": best_s3_model_name,
    }

    update_json_path = out_dir / "stage2_update_metrics.json"
    update_json_path.write_text(json.dumps(final_update_json, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu toàn bộ số liệu cập nhật vào: {update_json_path}", flush=True)

if __name__ == "__main__":
    main()

