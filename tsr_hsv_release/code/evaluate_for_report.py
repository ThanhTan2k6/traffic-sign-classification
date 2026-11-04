"""Evaluation and asset generation script for Stage 1 Part 2 Report.
Does not retrain anything. Restores models (loads kNN from joblib, fits SVM with fixed C=10.0),
computes A', FP breakdown, End-to-end recall, generates fig_hist_by_group.png, correct_samples,
and dumps exact numbers to extra_metrics.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from ultralytics import YOLO

def extract_hsv_feature(crop_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop_bgr, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1]
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

def main():
    out_dir = Path("reports/stage1_classify_vn")
    out_dir.mkdir(parents=True, exist_ok=True)
    correct_dir = out_dir / "correct_samples"
    correct_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("data/vn/manifest.csv")
    gmap_path = Path("data/vn/group_map.csv")
    yolo_root = Path("data/vn/yolo")
    knn_model_path = Path("models/stage1_hsv_vn_best.joblib")

    manifest = pd.read_csv(manifest_path)
    gmap = pd.read_csv(gmap_path)

    prefix_to_group = {
        "P": "cấm (prohibitory)",
        "R": "hiệu lệnh (mandatory)",
        "W": "cảnh báo (warning)",
        "I": "chỉ dẫn (information)",
        "S": "phụ (supplementary)",
    }
    cls_to_group = {}
    for r in gmap.itertuples():
        p = r.name.split(".")[0]
        cls_to_group[r.name] = prefix_to_group.get(p, "chưa xác định")

    manifest["group_label"] = manifest["label_fine"].map(cls_to_group)
    manifest["bw"] = manifest["x2"] - manifest["x1"]
    manifest["bh"] = manifest["y2"] - manifest["y1"]

    all_groups = sorted(manifest["group_label"].unique())

    # Filter train < 8 px
    train_mask = (manifest["split"] == "train") & (manifest["bw"] >= 8) & (manifest["bh"] >= 8)
    train_df = manifest[train_mask]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]

    print(f"Loading features for Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)})...")

    def load_crops_and_feats(df_sub):
        X, y, crops, metas = [], [], [], []
        for rel_img_p, rows in df_sub.groupby("yolo_img"):
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
                feat = extract_hsv_feature(crop)
                X.append(feat)
                y.append(r.group_label)
                crops.append(crop)
                metas.append({
                    "stem": f"{int(r.img_stem):04d}",
                    "box": [x1, y1, x2, y2],
                    "label_fine": r.label_fine,
                    "group": r.group_label
                })
        return np.array(X), np.array(y), crops, metas

    X_train, y_train, train_crops, train_metas = load_crops_and_feats(train_df)
    X_val, y_val, val_crops, val_metas = load_crops_and_feats(val_df)
    X_test_gt, y_test_gt, test_crops, test_metas = load_crops_and_feats(test_df)

    # 1. Models
    print("Loading kNN from joblib...")
    knn_payload = joblib.load(knn_model_path)
    knn = knn_payload["model"]

    print("Restoring SVM (C=10.0, gamma='scale', balanced)...")
    svm = SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=42)
    svm.fit(X_train, y_train)

    # 2. Generate fig_hist_by_group.png
    print("Generating fig_hist_by_group.png...")
    fig, axes = plt.subplots(len(all_groups), 2, figsize=(10, 14), dpi=150, gridspec_kw={'width_ratios': [1, 3]})
    sample_crops = {}
    for g_idx, g in enumerate(all_groups):
        # Find representative crop
        g_indices = [i for i, y in enumerate(y_test_gt) if y == g]
        if not g_indices:
            g_indices = [i for i, y in enumerate(y_train) if y == g]
        rep_idx = g_indices[min(len(g_indices)//2, len(g_indices)-1)]
        rep_crop = test_crops[rep_idx] if rep_idx < len(test_crops) and y_test_gt[rep_idx] == g else train_crops[rep_idx]
        sample_crops[g] = rep_crop

        # Display crop
        ax_crop = axes[g_idx, 0]
        rep_rgb = cv2.cvtColor(rep_crop, cv2.COLOR_BGR2RGB)
        ax_crop.imshow(rep_rgb)
        ax_crop.set_title(f"Mẫu: {g}", fontsize=9, pad=3)
        ax_crop.axis("off")

        # Average Hue histogram across train for this group
        g_mask = (y_train == g)
        avg_h_hist = np.mean(X_train[g_mask, :36], axis=0)

        ax_hist = axes[g_idx, 1]
        bins_x = np.arange(36)
        ax_hist.bar(bins_x, avg_h_hist, color="steelblue", width=0.8, edgecolor="black", linewidth=0.5)
        ax_hist.set_xlim(-0.5, 35.5)
        ax_hist.set_ylabel("Tỉ trọng (L1)", fontsize=8)
        if g_idx == len(all_groups) - 1:
            ax_hist.set_xlabel("Bin Hue shifted (+30 mod 180, mỗi bin 5 độ)", fontsize=9)
        ax_hist.set_title(f"Histogram Hue trung bình: {g}", fontsize=9, pad=3)
        ax_hist.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_hist_by_group.png")
    plt.close(fig)

    # 3. Save 6-8 correct crops for illustration
    print("Saving correct sample crops...")
    # Find examples correctly classified by both SVM and kNN in Level A
    pred_a_svm = svm.predict(X_test_gt)
    pred_a_knn = knn.predict(X_test_gt)
    correct_info = []

    for g in all_groups:
        match_idx = [i for i in range(len(y_test_gt)) if y_test_gt[i] == g and pred_a_svm[i] == g and pred_a_knn[i] == g]
        if match_idx:
            # take 1 or 2
            for pick_i, chosen_idx in enumerate(match_idx[:2 if len(match_idx)>1 else 1]):
                c_crop = test_crops[chosen_idx]
                meta = test_metas[chosen_idx]
                fname = f"correct_{g[:4].strip()}_{pick_i+1}_stem{meta['stem']}.jpg"
                cv2.imwrite(str(correct_dir / fname), c_crop)
                correct_info.append({
                    "file": fname,
                    "group": g,
                    "stem": meta["stem"],
                    "label_fine": meta["label_fine"],
                    "size": f"{c_crop.shape[1]}x{c_crop.shape[0]}"
                })

    # 4. YOLO Inference & Level B Matching
    print("Running YOLO inference on test set for A', B, and FP analysis...")
    yolo_weights = Path("runs/detect/vn_s/weights/best.pt")
    yolo_model = YOLO(str(yolo_weights))
    test_img_dir = yolo_root / "images" / "test"
    test_stems = sorted(test_df["img_stem"].unique().tolist())

    matched_gt_feats = []
    matched_gt_labels = []
    matched_yolo_feats = []
    matched_metas = []

    fp_feats = []
    fp_metas = []

    for s in test_stems:
        stem_str = f"{int(s):04d}"
        img_p = test_img_dir / f"{stem_str}.jpg"
        if not img_p.exists():
            img_p = test_img_dir / f"{stem_str}.png"
        if not img_p.exists():
            continue

        img = cv2.imread(str(img_p))
        ih, iw = img.shape[:2]

        # GT boxes
        gt_rows = test_df[test_df["img_stem"] == s]
        gt_boxes, gt_lbls, gt_fines, gt_crops_list = [], [], [], []
        for r in gt_rows.itertuples():
            x1 = max(0, min(iw - 1, int(r.x1)))
            y1 = max(0, min(ih - 1, int(r.y1)))
            x2 = max(x1 + 1, min(iw, int(r.x2)))
            y2 = max(y1 + 1, min(ih, int(r.y2)))
            gt_boxes.append([x1, y1, x2, y2])
            gt_lbls.append(r.group_label)
            gt_fines.append(r.label_fine)
            gt_crop = img[y1:y2, x1:x2]
            gt_crops_list.append(gt_crop)

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

                # YOLO crop
                ycrop = img[p_box[1]:p_box[3], p_box[0]:p_box[2]]
                gt_c = gt_crops_list[best_g_idx]
                if ycrop.size > 0 and ycrop.shape[0] >= 2 and ycrop.shape[1] >= 2 and gt_c.size > 0:
                    y_feat = extract_hsv_feature(ycrop)
                    gt_feat = extract_hsv_feature(gt_c)

                    matched_yolo_feats.append(y_feat)
                    matched_gt_feats.append(gt_feat)
                    matched_gt_labels.append(gt_lbls[best_g_idx])
                    matched_metas.append({
                        "stem": stem_str,
                        "label_fine": gt_fines[best_g_idx],
                        "true_group": gt_lbls[best_g_idx],
                        "iou": round(best_iou, 3)
                    })

        # FP boxes
        for p_idx, p_box in enumerate(pred_boxes):
            if p_idx not in matched_pred_set:
                fp_crop = img[p_box[1]:p_box[3], p_box[0]:p_box[2]]
                if fp_crop.size > 0 and fp_crop.shape[0] >= 2 and fp_crop.shape[1] >= 2:
                    fp_feat = extract_hsv_feature(fp_crop)
                    fp_feats.append(fp_feat)
                    fp_metas.append({"stem": stem_str, "box": p_box})

    X_matched_gt = np.array(matched_gt_feats)
    X_matched_yolo = np.array(matched_yolo_feats)
    y_matched_gt = np.array(matched_gt_labels)
    X_fp = np.array(fp_feats)

    print(f"Matched count: {len(y_matched_gt)}, FP count: {len(X_fp)}")

    # 5. Evaluate Mức A' (GT crop on 1,559 matched signs)
    pred_a_prime_svm = svm.predict(X_matched_gt)
    acc_a_prime_svm = accuracy_score(y_matched_gt, pred_a_prime_svm)
    f1_a_prime_svm = f1_score(y_matched_gt, pred_a_prime_svm, average="macro", zero_division=0)

    pred_a_prime_knn = knn.predict(X_matched_gt)
    acc_a_prime_knn = accuracy_score(y_matched_gt, pred_a_prime_knn)
    f1_a_prime_knn = f1_score(y_matched_gt, pred_a_prime_knn, average="macro", zero_division=0)

    # 6. Evaluate Mức B (YOLO crop on 1,559 matched signs)
    pred_b_svm = svm.predict(X_matched_yolo)
    acc_b_svm = accuracy_score(y_matched_gt, pred_b_svm)
    f1_b_svm = f1_score(y_matched_gt, pred_b_svm, average="macro", zero_division=0)

    pred_b_knn = knn.predict(X_matched_yolo)
    acc_b_knn = accuracy_score(y_matched_gt, pred_b_knn)
    f1_b_knn = f1_score(y_matched_gt, pred_b_knn, average="macro", zero_division=0)

    # 7. End-to-end Recall
    # (GT signs detected with IoU >= 0.5 AND correctly classified) / total GT (1,845)
    total_gt = len(test_df) # 1845
    e2e_correct_svm = int(np.sum(pred_b_svm == y_matched_gt))
    e2e_recall_svm = e2e_correct_svm / total_gt

    e2e_correct_knn = int(np.sum(pred_b_knn == y_matched_gt))
    e2e_recall_knn = e2e_correct_knn / total_gt

    # 8. FP Detector Predictions breakdown
    pred_fp_svm = svm.predict(X_fp)
    pred_fp_knn = knn.predict(X_fp)

    fp_breakdown_svm = pd.Series(pred_fp_svm).value_counts().to_dict()
    fp_breakdown_knn = pd.Series(pred_fp_knn).value_counts().to_dict()

    # 9. Top confusion pairs in Level A and Level B
    def get_top_confusions(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=all_groups)
        conf_pairs = []
        for i, t_grp in enumerate(all_groups):
            for j, p_grp in enumerate(all_groups):
                if i != j and cm[i, j] > 0:
                    conf_pairs.append({
                        "true": t_grp,
                        "pred": p_grp,
                        "count": int(cm[i, j])
                    })
        conf_pairs.sort(key=lambda x: x["count"], reverse=True)
        return conf_pairs

    top_conf_a_svm = get_top_confusions(y_test_gt, pred_a_svm)
    top_conf_a_knn = get_top_confusions(y_test_gt, pred_a_knn)
    top_conf_b_svm = get_top_confusions(y_matched_gt, pred_b_svm)
    top_conf_b_knn = get_top_confusions(y_matched_gt, pred_b_knn)

    # 10. Check warning signs test vs train distance
    warning_test_feats = X_test_gt[y_test_gt == "cảnh báo (warning)"]
    warning_train_feats = X_train[y_train == "cảnh báo (warning)"]
    # Compute min euclidean distance from each test warning to train warning
    from scipy.spatial.distance import cdist
    dists = cdist(warning_test_feats, warning_train_feats, metric="euclidean")
    min_dists = np.min(dists, axis=1)
    near_identical = np.sum(min_dists < 0.05) # very close in HSV hist

    # 11. Enrich misclassified examples metadata
    misc_json_path = out_dir / "misclassified_examples.json"
    with open(misc_json_path, "r", encoding="utf-8") as f:
        misc_list = json.load(f)

    for item in misc_list:
        stem_str = item["stem"]
        # Find corresponding GT row in test_df
        stem_gts = test_df[test_df["img_stem"] == int(stem_str)]
        # Pick best matching bbox
        item_fine = "N/A"
        item_fine_id = "N/A"
        for r in stem_gts.itertuples():
            item_fine = r.label_fine
            item_fine_id = r.id if hasattr(r, 'id') else "N/A"
            break
        item["original_class"] = item_fine

    # Dump all extra metrics to JSON
    results = {
        "mức_so_sánh_công_bằng": {
            "SVM": {
                "Level_A_all_1845": {"acc": float(accuracy_score(y_test_gt, pred_a_svm)), "macro_f1": float(f1_score(y_test_gt, pred_a_svm, average="macro", zero_division=0))},
                "Level_A_prime_matched_1559": {"acc": float(acc_a_prime_svm), "macro_f1": float(f1_a_prime_svm)},
                "Level_B_yolo_1559": {"acc": float(acc_b_svm), "macro_f1": float(f1_b_svm)}
            },
            "kNN": {
                "Level_A_all_1845": {"acc": float(accuracy_score(y_test_gt, pred_a_knn)), "macro_f1": float(f1_score(y_test_gt, pred_a_knn, average="macro", zero_division=0))},
                "Level_A_prime_matched_1559": {"acc": float(acc_a_prime_knn), "macro_f1": float(f1_a_prime_knn)},
                "Level_B_yolo_1559": {"acc": float(acc_b_knn), "macro_f1": float(f1_b_knn)}
            }
        },
        "end_to_end_pipeline": {
            "total_test_gt": total_gt,
            "yolo_matched_gt": len(y_matched_gt),
            "SVM": {
                "correct_signs": e2e_correct_svm,
                "recall_real": float(e2e_recall_svm)
            },
            "kNN": {
                "correct_signs": e2e_correct_knn,
                "recall_real": float(e2e_recall_knn)
            }
        },
        "detector_fp_classification": {
            "total_fp": len(X_fp),
            "SVM_predicted_groups": fp_breakdown_svm,
            "kNN_predicted_groups": fp_breakdown_knn
        },
        "top_confusion_pairs": {
            "level_a_svm": top_conf_a_svm[:5],
            "level_a_knn": top_conf_a_knn[:5],
            "level_b_svm": top_conf_b_svm[:5],
            "level_b_knn": top_conf_b_knn[:5]
        },
        "warning_similarity_check": {
            "test_warning_count": len(warning_test_feats),
            "near_identical_hist_dist_under_0_05": int(near_identical),
            "mean_min_dist": float(np.mean(min_dists)),
            "median_min_dist": float(np.median(min_dists))
        },
        "correct_samples": correct_info,
        "enriched_misclassified": misc_list
    }

    with open(out_dir / "report_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Completed evaluation! Dumped to report_metrics.json.")

if __name__ == "__main__":
    main()
