"""Step 1 - prepare data.

For each dataset it produces, under <out>/<name>/:
  crops.u8, manifest.csv     64x64 crops cut from GT boxes (+ jittered copies for train) and their metadata
  yolo/                      single-class ("sign") YOLO dataset with a leak-free split, + data.yaml
  class_gallery.png          a few crops per class, for eyeballing
  group_map.csv              (Vietnam only) class id -> group, REVIEW IT
  test_tags.csv              (CCTSDB only) weather / size tag of every test image

Splits
  CCTSDB   : official test kept as-is. val = contiguous blocks cut from train (with a gap of dropped frames).
  Vietnam  : frames are video-ordered, so we split by contiguous blocks of frames and drop `gap` frames
             at every boundary between two different splits. No random frame-level split!

Usage:
  python prepare_data.py --cctsdb-root archive --vn-root archive_new/archive --out data
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (CCTSDB_NAMES, block_split, crop_resize, imread, imwrite, jitter_box, link_or_copy,
                    list_images, read_yolo_label, to_common3, vn_code, vn_group, xyxy_to_yolo_line,
                    yolo_to_xyxy)


# ---------------------------------------------------------------- generic builder
def build(name, items, out_root, label_fn, args, tags=None, negatives=()):
    """items: list of dict(stem, img, label, split). label_fn(cls_id) -> (fine, group) or None."""
    out = Path(out_root) / name
    ydir = out / "yolo"
    for sp in ("train", "val", "test"):
        (ydir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (ydir / "labels" / sp).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows, bad_cls, degenerate, unreadable = [], 0, 0, 0
    raw = open(out / "crops.u8", "wb")

    for it in tqdm(items, desc=f"[{name}] images"):
        sp = it["split"]
        if sp == "gap":
            continue
        img = imread(it["img"])
        if img is None:
            unreadable += 1
            continue
        H, W = img.shape[:2]
        lab = read_yolo_label(it["label"])
        boxes = []
        for r in lab:
            res = label_fn(int(r[0]))
            if res is None:
                bad_cls += 1
                continue
            box = yolo_to_xyxy(r, W, H)
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                degenerate += 1
                continue
            boxes.append((int(r[0]), box, res))

        # --- YOLO export (single class)
        dst_img = ydir / "images" / sp / Path(it["img"]).name
        link_or_copy(Path(it["img"]), dst_img)
        (ydir / "labels" / sp / f"{it['stem']}.txt").write_text(
            "\n".join(xyxy_to_yolo_line(b, W, H, 0) for _, b, _ in boxes) + ("\n" if boxes else ""),
            encoding="utf-8")

        # --- crops
        for cid, box, (fine, group) in boxes:
            variants = [(box, 0)]
            if sp == "train" and args.jitter > 0:
                variants += [(jitter_box(box, rng, args.jitter_amount), 1) for _ in range(args.jitter)]
            for b, aug in variants:
                raw.write(crop_resize(img, b, args.pad, args.size).tobytes())
                rows.append(dict(
                    img_stem=it["stem"], split=sp, x1=b[0], y1=b[1], x2=b[2], y2=b[3], img_w=W, img_h=H,
                    cls_id=cid, label_fine=fine, label_group=group, label_common3=to_common3(group),
                    is_aug=aug, yolo_img=str(dst_img.relative_to(ydir)).replace("\\", "/"),
                    weather=(tags or {}).get("weather", {}).get(it["stem"]) if sp == "test" else None,
                    size=(tags or {}).get("size", {}).get(it["stem"]) if sp == "test" else None))
    raw.close()

    # --- negatives (train only, renamed so they never overwrite real frames)
    n_neg = 0
    for p in negatives:
        img = imread(p)
        if img is None:
            continue
        stem = f"neg_{Path(p).stem}"
        imwrite(ydir / "images" / "train" / f"{stem}.jpg", img)
        (ydir / "labels" / "train" / f"{stem}.txt").write_text("", encoding="utf-8")
        n_neg += 1

    man = pd.DataFrame(rows)
    man.insert(0, "row_id", np.arange(len(man)))
    man.to_csv(out / "manifest.csv", index=False)
    (ydir / "data.yaml").write_text(
        f"path: {ydir.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: sign\n", encoding="utf-8")

    # --- report
    print(f"\n===== {name} =====")
    cnt = Counter(it["split"] for it in items)
    print("frames per split:", dict(cnt))
    print(f"negatives added to train: {n_neg} | unreadable images: {unreadable} | "
          f"degenerate boxes dropped: {degenerate} | unknown class ids: {bad_cls}")
    real = man[man.is_aug == 0]
    print("GT boxes per split:", real.split.value_counts().to_dict(), "| train crops incl. jitter:",
          int((man.split == "train").sum()))
    print(pd.crosstab(real.label_group, real.split).to_string())
    save_gallery(out, man, args.size)
    (out / "config.json").write_text(json.dumps(dict(size=args.size, pad=args.pad, jitter=args.jitter,
                                                     jitter_amount=args.jitter_amount, seed=args.seed)), encoding="utf-8")
    return man


def save_gallery(out: Path, man: pd.DataFrame, size: int, per_class: int = 8, max_rows: int = 60):
    from common import load_crops
    crops = load_crops(out, len(man), size)
    real = man[(man.is_aug == 0) & (man.split == "train")]
    labels = real.label_fine.value_counts().index.tolist()[:max_rows]
    canvas = []
    for lab in labels:
        ids = real[real.label_fine == lab].row_id.values
        ids = np.random.default_rng(0).choice(ids, size=min(per_class, len(ids)), replace=False)
        row = [np.asarray(crops[i]) for i in ids] + [np.zeros((size, size, 3), np.uint8)] * (per_class - len(ids))
        strip = np.hstack(row)
        cv2.putText(strip, str(lab)[:14], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        canvas.append(strip)
    if canvas:
        imwrite(out / "class_gallery.png", np.vstack(canvas))


# ---------------------------------------------------------------- CCTSDB2021
def find_split_dirs(root: Path, split: str):
    """Locate <root>/<split>/images and <root>/<split>/labels."""
    for cand in (root / split, *[p for p in root.rglob(split) if p.is_dir()]):
        if (cand / "images").is_dir() and (cand / "labels").is_dir():
            return cand / "images", cand / "labels"
    raise SystemExit(f"[error] cannot find '{split}/images' and '{split}/labels' under {root}. "
                     f"Children: {[p.name for p in root.iterdir()]}")


def scan_tag_dir(folder: Path) -> dict:
    """stem -> tag, where tag is the first sub-folder under `folder` that contains the file."""
    tags = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".xml", ".txt", ".jpg", ".png"):
            rel = p.relative_to(folder).parts
            tags[p.stem] = rel[0] if len(rel) > 1 else "unknown"
    return tags


def prepare_cctsdb(root: Path, out_root: Path, args):
    if (root / "CCTSDB2021").is_dir():
        root = root / "CCTSDB2021"
    train_img, train_lab = find_split_dirs(root, "train")
    test_img, test_lab = find_split_dirs(root, "test")
    tr = list_images(train_img)
    te = list_images(test_img)
    sp = block_split([p.stem for p in tr], {"train": 1 - args.val_ratio, "val": args.val_ratio},
                     block=args.block_cctsdb, gap=args.gap, seed=args.seed)
    items = [dict(stem=p.stem, img=p, label=train_lab / f"{p.stem}.txt", split=sp[p.stem]) for p in tr]
    items += [dict(stem=p.stem, img=p, label=test_lab / f"{p.stem}.txt", split="test") for p in te]

    tags = {"weather": {}, "size": {}}
    negatives = []
    for d in root.iterdir():  # folder names are Chinese; match on a key word
        if not d.is_dir():
            continue
        if "天气" in d.name:
            tags["weather"] = scan_tag_dir(d)
        elif "尺寸" in d.name:
            tags["size"] = scan_tag_dir(d)
        elif "负样本" in d.name:
            negatives = list_images(d)
    print(f"[cctsdb] weather tags: {len(tags['weather'])} | size tags: {len(tags['size'])} | negatives: {len(negatives)}")
    print("[cctsdb] weather tag names:", sorted(set(tags["weather"].values())),
          "| size tag names:", sorted(set(tags["size"].values())))

    def label_fn(cid):
        return None if cid not in CCTSDB_NAMES else (CCTSDB_NAMES[cid], CCTSDB_NAMES[cid])

    man = build("cctsdb", items, out_root, label_fn, args, tags=tags, negatives=negatives)
    te_stems = sorted({it["stem"] for it in items if it["split"] == "test"})
    pd.DataFrame(dict(stem=te_stems, weather=[tags["weather"].get(s) for s in te_stems],
                      size=[tags["size"].get(s) for s in te_stems])).to_csv(
        Path(out_root) / "cctsdb" / "test_tags.csv", index=False)
    return man


# ---------------------------------------------------------------- Vietnam dataset
def prepare_vn(root: Path, out_root: Path, args):
    if (root / "archive").is_dir() and (root / "archive" / "images").is_dir():
        root = root / "archive"
    img_dir = root / "images"
    lab_dir = root / "labels"
    if not img_dir.is_dir() or not lab_dir.is_dir():
        raise SystemExit(f"[error] expected {img_dir} and {lab_dir}. Children: {[p.name for p in root.iterdir()]}")
    names = [l.strip() for l in (root / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    vie = root / "classes_vie.txt"
    vie_names = [l.strip() for l in vie.read_text(encoding="utf-8").splitlines() if l.strip()] if vie.exists() else names
    groups = [vn_group(n) for n in names]

    out = Path(out_root) / "vn"
    out.mkdir(parents=True, exist_ok=True)
    gm = pd.DataFrame(dict(id=range(len(names)), name=names,
                           vie_name=(vie_names + [""] * len(names))[:len(names)],
                           code=[vn_code(n) for n in names], group=groups))
    if args.group_map and Path(args.group_map).exists():  # manual override: CSV with columns id,group
        ov = pd.read_csv(args.group_map).set_index("id")["group"].to_dict()
        gm["group"] = [ov.get(i, g) for i, g in zip(gm.id, gm.group)]
        groups = gm.group.tolist()
    gm["common3"] = [to_common3(g) for g in gm.group]
    gm.to_csv(out / "group_map.csv", index=False, encoding="utf-8-sig")
    print(f"[vn] {len(names)} classes -> groups: {dict(Counter(groups))}  (REVIEW {out / 'group_map.csv'})")
    if "unknown" in groups:
        print("[vn][warn] some classes are 'unknown' -> fix them in group_map.csv and rerun with --group-map")

    imgs = list_images(img_dir)
    sp = block_split([p.stem for p in imgs], {"train": 1 - args.val_ratio - args.test_ratio,
                                              "val": args.val_ratio, "test": args.test_ratio},
                     block=args.block_vn, gap=args.gap, seed=args.seed)
    items = [dict(stem=p.stem, img=p, label=lab_dir / f"{p.stem}.txt", split=sp[p.stem]) for p in imgs]

    def label_fn(cid):
        return None if not (0 <= cid < len(names)) else (names[cid], groups[cid])

    return build("vn", items, out_root, label_fn, args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cctsdb-root", default="archive")
    ap.add_argument("--vn-root", default="archive_new/archive")
    ap.add_argument("--out", default="data")
    ap.add_argument("--only", choices=["cctsdb", "vn"], default=None)
    ap.add_argument("--size", type=int, default=64, help="crop size")
    ap.add_argument("--pad", type=float, default=0.05, help="relative padding around the box")
    ap.add_argument("--jitter", type=int, default=2, help="extra jittered crops per train box")
    ap.add_argument("--jitter-amount", type=float, default=0.10)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.20, help="Vietnam only (CCTSDB keeps its own test)")
    ap.add_argument("--block-vn", type=int, default=100)
    ap.add_argument("--block-cctsdb", type=int, default=200)
    ap.add_argument("--gap", type=int, default=5, help="frames dropped at boundaries between different splits")
    ap.add_argument("--group-map", default=None, help="CSV (id,group) overriding the automatic Vietnam grouping")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "cctsdb") and Path(args.cctsdb_root).exists():
        prepare_cctsdb(Path(args.cctsdb_root), out, args)
    if args.only in (None, "vn") and Path(args.vn_root).exists():
        prepare_vn(Path(args.vn_root), out, args)


if __name__ == "__main__":
    main()
