"""Step 4 - evaluate the full pipeline: image -> YOLO ("sign") -> crop -> classifier -> compare with GT.

It reports three things separately, so you can see where errors come from:
  A. detector only (class-agnostic) : precision / recall / F1 at --match-iou
  B. classifier on matched detections (crops cut from YOLO boxes) vs. classifier on GT boxes (upper bound)
  C. end-to-end per class           : a detection counts only if it overlaps a GT box AND the class is right
and, for CCTSDB, the same numbers split by weather/lighting and by sign size (test_tags.csv).

Usage:
  python eval_pipeline.py --data-dir data/cctsdb --detector runs/cctsdb_s/weights/best.pt \
         --classifier models/cctsdb_group_best.joblib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common import crop_resize, greedy_match, imread, imwrite, load_crops
from features import extract, extract_batch


def detect_all(args, img_paths):
    """Run YOLO on all images -> {path: (boxes_xyxy (n,4), conf (n,))}. Kept separate so it can be mocked."""
    from ultralytics import YOLO

    model = YOLO(args.detector)
    out = {}
    for i in range(0, len(img_paths), args.batch):
        chunk = [str(p) for p in img_paths[i:i + args.batch]]
        for p, r in zip(chunk, model.predict(chunk, imgsz=args.imgsz, conf=args.conf, iou=args.iou_nms,
                                             device=args.device, verbose=False)):
            out[Path(p)] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
        print(f"\r  detecting {min(i + args.batch, len(img_paths))}/{len(img_paths)}", end="")
    print()
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def summarize(records, gt_by_img, n_classes_hint=None):
    """records: list of dict(img, matched_gt(label or None), pred). Returns dict of metrics."""
    df = pd.DataFrame(records)
    n_gt = sum(len(v) for v in gt_by_img.values())
    n_det = len(df)
    matched = df[df.gt_label.notna()] if n_det else df
    tp_det = len(matched)
    p, r, f = prf(tp_det, n_det - tp_det, n_gt - tp_det)
    res = dict(n_gt=n_gt, n_det=n_det, det_tp=tp_det, det_precision=p, det_recall=r, det_f1=f)
    if tp_det:
        res["cls_acc_on_matched"] = float(accuracy_score(matched.gt_label, matched.pred))
        res["cls_macro_f1_on_matched"] = float(f1_score(matched.gt_label, matched.pred,
                                                         labels=np.unique(matched.gt_label), average="macro",
                                                         zero_division=0))
    ok = df[(df.gt_label.notna()) & (df.gt_label == df.pred)] if n_det else df
    p2, r2, f2 = prf(len(ok), n_det - len(ok), n_gt - len(ok))
    res.update(e2e_tp=len(ok), e2e_precision=p2, e2e_recall=r2, e2e_f1=f2)
    return res


def per_class_e2e(records, gt_by_img):
    df = pd.DataFrame(records)
    gt_counts = pd.Series([l for v in gt_by_img.values() for l in v]).value_counts()
    rows = []
    for c, n in gt_counts.items():
        tp = int(((df.gt_label == c) & (df.pred == c)).sum()) if len(df) else 0
        fp = int(((df.pred == c) & ~(df.gt_label == c)).sum()) if len(df) else 0
        p, r, f = prf(tp, fp, n - tp)
        rows.append(dict(label=c, n_gt=int(n), tp=tp, fp=fp, fn=int(n - tp), precision=p, recall=r, f1=f))
    out = pd.DataFrame(rows).sort_values("n_gt", ascending=False)
    return out


def draw(img, dets, gts_unmatched, out_path):
    vis = img.copy()
    for (x1, y1, x2, y2), lab, status in dets:
        col = {"ok": (0, 200, 0), "wrong": (0, 0, 255), "fp": (0, 215, 255)}[status]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 1)
        cv2.putText(vis, str(lab).encode("ascii", "replace").decode()[:12], (int(x1), max(10, int(y1) - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    for x1, y1, x2, y2 in gts_unmatched:  # missed GT in blue
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 120, 0), 1)
    imwrite(out_path, vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-nms", type=float, default=0.6)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--viz", type=int, default=0, help="save N annotated images")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ds = data_dir.name
    bundle = joblib.load(args.classifier)
    pipe, spec, level = bundle["pipeline"], bundle["spec"], bundle["label_level"]
    cfg = json.loads((data_dir / "config.json").read_text())
    pad, size = cfg["pad"], cfg["size"]
    ycol = "label_group" if level == "group" else "label_fine"

    man = pd.read_csv(data_dir / "manifest.csv")
    gt = man[(man.split == args.split) & (man.is_aug == 0)]
    gt_by_img, gt_box_by_img = {}, {}
    for yimg, g in gt.groupby("yolo_img"):
        p = data_dir / "yolo" / yimg
        gt_by_img[p] = g[ycol].astype(str).tolist()
        gt_box_by_img[p] = g[["x1", "y1", "x2", "y2"]].values
    img_paths = sorted((data_dir / "yolo" / "images" / args.split).glob("*"))
    img_paths = [p for p in img_paths if not p.stem.startswith("neg_")]
    for p in img_paths:
        gt_by_img.setdefault(p, [])
        gt_box_by_img.setdefault(p, np.zeros((0, 4)))
    print(f"[{ds}] {len(img_paths)} {args.split} images, {len(gt)} GT boxes, classifier={bundle['level_name']}"
          f"+{bundle['clf']} ({level})")

    dets = detect_all(args, img_paths)

    # ---- crop every detection, classify all at once
    feats, meta = [], []
    for p in img_paths:
        boxes, conf = dets[p]
        if len(boxes) == 0:
            continue
        img = imread(p)
        for b, c in zip(boxes, conf):
            crop = crop_resize(img, b, pad, size)
            feats.append(extract(crop, b[2] - b[0], b[3] - b[1], spec))
            meta.append((p, b, float(c)))
    preds = pipe.predict(np.stack(feats)) if feats else []

    # ---- match to GT per image
    by_img = {}
    for (p, b, c), pr in zip(meta, preds):
        by_img.setdefault(p, []).append((b, c, str(pr)))
    records, viz_items = [], []
    for p in img_paths:
        items = by_img.get(p, [])
        gboxes, glabs = gt_box_by_img[p], gt_by_img[p]
        pairs = greedy_match(np.array([i[0] for i in items]), [i[1] for i in items], gboxes, args.match_iou)
        for di, (b, c, pr) in enumerate(items):
            gl = glabs[pairs[di]] if di in pairs else None
            records.append(dict(img=p.name, stem=p.stem, x1=b[0], y1=b[1], x2=b[2], y2=b[3], conf=c, pred=pr,
                                gt_label=gl))
        matched_gt = set(pairs.values())
        viz_items.append((p, [(b, pr, "fp" if di not in pairs else ("ok" if glabs[pairs[di]] == pr else "wrong"))
                              for di, (b, c, pr) in enumerate(items)],
                          [gboxes[i] for i in range(len(gboxes)) if i not in matched_gt]))

    res = summarize(records, gt_by_img)

    # ---- classifier on GT boxes (upper bound, no detector noise)
    crops = load_crops(data_dir, len(man), size)
    ids = gt.row_id.values
    whs = np.stack([(gt.x2 - gt.x1).values, (gt.y2 - gt.y1).values], 1)
    Xg = extract_batch(crops[ids], whs, spec, n_jobs=-1)
    pg = pipe.predict(Xg)
    res["cls_acc_on_gt_boxes"] = float(accuracy_score(gt[ycol].astype(str), pg))
    res["cls_macro_f1_on_gt_boxes"] = float(f1_score(gt[ycol].astype(str), pg, labels=np.unique(gt[ycol].astype(str)),
                                                      average="macro", zero_division=0))

    out_dir = Path(args.out) / f"pipeline_{ds}_{level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_dir / "detections.csv", index=False)
    per_class = per_class_e2e(records, gt_by_img)
    per_class.to_csv(out_dir / "per_class_e2e.csv", index=False)
    res["e2e_macro_f1"] = float(per_class.f1.mean()) if len(per_class) else 0.0

    # ---- by weather / size (CCTSDB)
    tag_file = data_dir / "test_tags.csv"
    if tag_file.exists() and args.split == "test":
        tags = pd.read_csv(tag_file, dtype=str).set_index("stem")
        rec = pd.DataFrame(records)
        rows = []
        for col in ("weather", "size"):
            if col not in tags or tags[col].dropna().empty:
                continue
            stem_tag = tags[col].to_dict()
            for tag in sorted(tags[col].dropna().unique()):
                imgs = [p for p in img_paths if stem_tag.get(p.stem) == tag]
                sub_gt = {p: gt_by_img[p] for p in imgs}
                sub_rec = rec[rec.stem.isin({p.stem for p in imgs})].to_dict("records") if len(rec) else []
                s = summarize(sub_rec, sub_gt)
                rows.append(dict(by=col, tag=tag, n_images=len(imgs), n_gt=s["n_gt"], det_recall=s["det_recall"],
                                 det_precision=s["det_precision"], cls_acc_on_matched=s.get("cls_acc_on_matched"),
                                 e2e_recall=s["e2e_recall"], e2e_precision=s["e2e_precision"], e2e_f1=s["e2e_f1"]))
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "by_tag.csv", index=False)
            print("\nBy tag:\n" + pd.DataFrame(rows).round(3).to_string(index=False))

    if args.viz:
        (out_dir / "viz").mkdir(exist_ok=True)
        for p, d, gu in viz_items[: args.viz]:
            draw(imread(p), d, gu, out_dir / "viz" / f"{p.stem}.jpg")

    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))
    print("\nSummary:")
    for k, v in res.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
    print(f"\nresults in {out_dir}/")


if __name__ == "__main__":
    main()
