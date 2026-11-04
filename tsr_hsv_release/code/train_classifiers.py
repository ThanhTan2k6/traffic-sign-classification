"""Step 2 - classical classifiers on GT crops, along the feature chain
L1 H-only -> L2 H+S -> L3 H+S grid (2x2 / 3x3) -> L4 +aspect ratio -> L5 +HOG.

Hyper-parameters are chosen on val (macro-F1), the model is then refit on train+val and scored on test.
Outputs (results/<dataset>_<level>/): results.csv, per_class.csv, confusion.png, chain.png, tags.csv
Best model (by val macro-F1) -> models/<dataset>_<level>_best.joblib  (used by eval_pipeline.py)

Usage:
  python train_classifiers.py --data-dir data/cctsdb --level group
  python train_classifiers.py --data-dir data/vn --level group
  python train_classifiers.py --data-dir data/vn --level fine
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import load_crops

warnings.filterwarnings("ignore", message=".*within_class_std_dev_.*")
from features import extract_batch, level_specs

GRIDS = {
    "centroid": [{}],
    "knn": [{"k": 1}, {"k": 3}, {"k": 5}, {"k": 9}],
    "svm": [{"C": 1}, {"C": 10}, {"C": 100}],
    "rf": [{}],
}


def make_model(name: str, params: dict, n_jobs: int):
    if name == "centroid":  # compare histograms with class-mean histogram (Hellinger distance)
        return Pipeline([("clf", NearestCentroid())])
    if name == "knn":
        clf = KNeighborsClassifier(n_neighbors=params["k"], weights="distance", n_jobs=n_jobs)
    elif name == "svm":
        clf = SVC(C=params["C"], kernel="rbf", gamma="scale", class_weight="balanced")
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=n_jobs,
                                     random_state=0)
    else:
        raise ValueError(name)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def macro_f1(y, p):
    return f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0)


def cap_per_class(idx: np.ndarray, y: np.ndarray, cap: int, rng) -> np.ndarray:
    if cap <= 0:
        return idx
    keep = []
    for c in np.unique(y[idx]):
        ids = idx[y[idx] == c]
        keep.append(ids if len(ids) <= cap else rng.choice(ids, size=cap, replace=False))
    return np.sort(np.concatenate(keep))


def get_features(name, spec, crops, whs, cache_dir: Path, n_jobs):
    f = cache_dir / f"{name}.npy"
    if f.exists():
        return np.load(f)
    t = time.time()
    X = extract_batch(crops, whs, spec, n_jobs=n_jobs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(f, X)
    print(f"    features {name}: {X.shape} in {time.time() - t:.0f}s")
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--level", choices=["group", "fine"], default="group")
    ap.add_argument("--classifiers", nargs="+", default=["centroid", "knn", "svm", "rf"])
    ap.add_argument("--max-per-class", type=int, default=4000, help="cap train samples per class (keeps SVM fast)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ds = data_dir.name
    man = pd.read_csv(data_dir / "manifest.csv")
    n = len(man)
    size = int(round(np.sqrt((data_dir / "crops.u8").stat().st_size / (n * 3))))
    crops = load_crops(data_dir, n, size)
    whs = np.stack([(man.x2 - man.x1).values, (man.y2 - man.y1).values], 1)
    ycol = "label_group" if args.level == "group" else "label_fine"
    y = man[ycol].astype(str).values
    rng = np.random.default_rng(args.seed)

    tr = np.where(man.split == "train")[0]  # includes jittered copies
    va = np.where((man.split == "val") & (man.is_aug == 0))[0]
    te = np.where((man.split == "test") & (man.is_aug == 0))[0]
    print(f"[{ds}/{args.level}] classes={len(np.unique(y))} train={len(tr)} val={len(va)} test={len(te)}")
    tr_fit = cap_per_class(tr, y, args.max_per_class, rng)
    trva_fit = cap_per_class(np.concatenate([tr, va]), y, args.max_per_class, rng)

    out_dir = Path(args.out) / f"{ds}_{args.level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = data_dir / "features"
    rows, best = [], None

    def run_level(name, spec):
        nonlocal best
        print(f"  level {name}")
        X = get_features(name, spec, crops, whs, cache_dir, args.n_jobs)
        hist_only = not spec.get("ar") and not spec.get("hog")
        for cname in args.classifiers:
            if cname == "centroid" and not hist_only:
                continue
            t = time.time()
            val_scores = []
            for params in GRIDS[cname]:
                m = make_model(cname, params, args.n_jobs).fit(X[tr_fit], y[tr_fit])
                val_scores.append(macro_f1(y[va], m.predict(X[va])) if len(va) else np.nan)
            bi = int(np.nanargmax(val_scores)) if len(va) else 0
            params = GRIDS[cname][bi]
            m = make_model(cname, params, args.n_jobs).fit(X[trva_fit], y[trva_fit])
            pred = m.predict(X[te])
            r = dict(level=name, classifier=cname, params=str(params), val_macro_f1=val_scores[bi],
                     test_acc=accuracy_score(y[te], pred), test_macro_f1=macro_f1(y[te], pred),
                     dim=X.shape[1], seconds=round(time.time() - t, 1))
            rows.append(r)
            print(f"    {cname:8s} {params} val_F1={r['val_macro_f1']:.3f} test_acc={r['test_acc']:.3f} "
                  f"test_F1={r['test_macro_f1']:.3f} ({r['seconds']}s)")
            if best is None or (r["val_macro_f1"] > best["val"]):
                best = dict(val=r["val_macro_f1"], level=name, spec=spec, clf=cname, params=params,
                            model=m, pred=pred)

    specs = level_specs(3)
    for name in ("L1_H", "L2_HS", "L3a_HS_grid2", "L3b_HS_grid3"):
        run_level(name, specs[name])
    df = pd.DataFrame(rows)
    g_scores = {g: float(df[df.level == f"L3{'a' if g == 2 else 'b'}_HS_grid{g}"].val_macro_f1.max()) for g in (2, 3)}
    best_grid = max(g_scores, key=g_scores.get)
    print(f"  -> best grid on val: {best_grid}x{best_grid} (val macro-F1 {({k: round(v, 3) for k, v in g_scores.items()})})")
    specs = level_specs(best_grid)
    for name in (f"L4_grid{best_grid}+AR", f"L5_grid{best_grid}+AR+HOG"):
        run_level(name, specs[name])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results.csv", index=False)

    # ---- chain plot (test macro-F1 per level, per classifier)
    piv = df.pivot_table(index="level", columns="classifier", values="test_macro_f1", sort=False)
    ax = piv.plot(kind="bar", figsize=(10, 4.5))
    ax.set_ylabel("test macro-F1")
    ax.set_title(f"{ds} / {args.level}: feature chain")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "chain.png", dpi=130)
    plt.close()

    # ---- best model: per-class report, confusion matrix, weather/size breakdown
    pred = best["pred"]
    rep = pd.DataFrame(classification_report(y[te], pred, zero_division=0, output_dict=True)).T
    rep.to_csv(out_dir / "per_class.csv")
    labels = sorted(np.unique(np.concatenate([y[te], pred])))
    cm = confusion_matrix(y[te], pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(max(5, 0.28 * len(labels) + 3),) * 2)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6 if len(labels) > 12 else 9)
    ax.set_yticklabels(labels, fontsize=6 if len(labels) > 12 else 9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"best: {best['level']} + {best['clf']} (row-normalised)")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion.png", dpi=130)
    plt.close()

    tag_rows = []
    for col in ("weather", "size"):
        if col in man and man.loc[te, col].notna().any():
            for tag, ids in man.iloc[te].groupby(col).groups.items():
                pos = np.searchsorted(te, np.asarray(list(ids)))
                tag_rows.append(dict(by=col, tag=tag, n=len(pos), acc=accuracy_score(y[te][pos], pred[pos]),
                                     macro_f1=macro_f1(y[te][pos], pred[pos])))
    if tag_rows:
        pd.DataFrame(tag_rows).to_csv(out_dir / "tags.csv", index=False)
        print(pd.DataFrame(tag_rows).round(3).to_string(index=False))

    Path(args.models).mkdir(parents=True, exist_ok=True)
    mpath = Path(args.models) / f"{ds}_{args.level}_best.joblib"
    joblib.dump(dict(pipeline=best["model"], spec=best["spec"], level_name=best["level"], clf=best["clf"],
                     params=best["params"], classes=list(best["model"].classes_), label_level=args.level,
                     dataset=ds, size=size), mpath)
    print(f"\nBEST by val: {best['level']} + {best['clf']} {best['params']} -> {mpath}")
    print(f"results in {out_dir}/")


if __name__ == "__main__":
    main()
