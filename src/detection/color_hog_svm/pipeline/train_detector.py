"""Step 3 - train the single-class ("sign") YOLO detector.

Signs are tiny (CCTSDB median ~24x25 px on 1000x350 images), so default imgsz=800 or 960.
Best weights: runs/detect/<name>/weights/best.pt
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

# Reduce CUDA memory fragmentation on Windows
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_free_virtual_memory_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPageFile / (1024**3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml created by prepare_data.py")
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default="yolo11s.pt", help="pretrained .pt (COCO) or your own best.pt")
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0, help="0 recommended on Windows to avoid multiprocessing MemoryError")
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--project", default=None, help="output directory (default: runs/detect)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=True, help="Enable AMP (default True)")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    ap.add_argument("--fraction", type=float, default=1.0, help="dataset fraction to use")
    ap.add_argument("--val-conf", type=float, default=0.25, help="confidence threshold for val/test evaluation")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    epoch_start_time = 0.0

    def on_train_epoch_start(trainer):
        nonlocal epoch_start_time
        epoch_start_time = time.time()

    def on_train_epoch_end(trainer):
        elapsed = time.time() - epoch_start_time
        free_vm = get_free_virtual_memory_gb()
        current_ep = trainer.epoch + 1
        total_ep = trainer.epochs
        print(f"\n[Monitor] Epoch {current_ep}/{total_ep} finished in {elapsed:.1f}s | FreeVirtualMemory: {free_vm:.2f} GB")
        if current_ep == 1:
            remaining_sec = elapsed * (total_ep - 1)
            print(f"[Monitor] Estimated time per epoch: {elapsed:.1f}s | Est. total remaining: {remaining_sec/60:.1f} min")
        if free_vm < 2.0:
            print(f"\nCRITICAL WARNING: FreeVirtualMemory has dropped to {free_vm:.2f} GB (< 2 GB)!\n")

    if args.resume:
        base_dir = Path(args.project) if args.project else Path("runs/detect")
        model = YOLO(str(base_dir / args.name / "weights" / "last.pt"))
    else:
        model = YOLO(args.weights)

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        amp=args.amp,
        fraction=args.fraction,
        cache=False,
        single_cls=True,
        close_mosaic=10,
        plots=True,
        exist_ok=True,
        resume=args.resume
    )

    save_dir = Path(getattr(model.trainer, "save_dir", Path("runs/detect") / args.name)).resolve()
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        for cand in [
            Path("runs/detect") / args.name / "weights" / "best.pt",
            save_dir.parent / args.name / "weights" / "best.pt",
        ]:
            if cand.resolve().exists():
                best = cand.resolve()
                break

    print(f"\n[detector] Best weights saved at: {best.resolve()}")

    # Evaluate on VAL split
    print(f"\n[detector] Evaluating {best} on the VAL split (conf={args.val_conf})...")
    val_res = YOLO(str(best)).val(
        data=args.data, split="val", imgsz=args.imgsz, batch=args.batch, conf=args.val_conf,
        iou=0.6, device=args.device, plots=True,
        project=args.project, name=f"{args.name}_val", exist_ok=True
    )
    val_metrics = {
        "precision": float(val_res.box.mp),
        "recall": float(val_res.box.mr),
        "map50": float(val_res.box.map50),
        "map50_95": float(val_res.box.map)
    }
    print("[detector] VAL Metrics:")
    print(json.dumps(val_metrics, indent=2))
    (save_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))

    # Evaluate on TEST split
    print(f"\n[detector] Evaluating {best} on the TEST split (conf={args.val_conf})...")
    test_res = YOLO(str(best)).val(
        data=args.data, split="test", imgsz=args.imgsz, batch=args.batch, conf=args.val_conf,
        iou=0.6, device=args.device, plots=True,
        project=args.project, name=f"{args.name}_test", exist_ok=True
    )
    test_metrics = {
        "precision": float(test_res.box.mp),
        "recall": float(test_res.box.mr),
        "map50": float(test_res.box.map50),
        "map50_95": float(test_res.box.map)
    }
    print("[detector] TEST Metrics:")
    print(json.dumps(test_metrics, indent=2))
    (save_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    print(f"\n[detector] ALL DONE. Final best model: {best.resolve()}")


if __name__ == "__main__":
    main()
