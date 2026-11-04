import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

with open("reports/stage2_vn/stage2_update_metrics.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("=== 1. ALL RESULTS TABLE ===")
for r in d["all_results_table"]:
    fn = r["Đặc trưng"]
    mn = r["Mô hình"]
    dims = r["Số chiều"]
    gkf = r["GroupKFold Macro-F1"]
    acc = r["Test Accuracy"] * 100
    f1 = r["Test Macro-F1"]
    m4 = r["Test Major4 F1"]
    print(f"{fn:<20} | {dims:3d} dims | {mn:<14} | GKF: {gkf} | Test Acc: {acc:.2f}% | Test F1: {f1:.4f} | Major4: {m4:.4f}")

print("\n=== 2. S3 MODELS EVAL (A / A' / B / E2E) ===")
for mn, res in d["s3_models_eval"].items():
    a_p_acc = res["a_prime_acc"] * 100
    a_p_f1 = res["a_prime_f1"]
    b_acc = res["b_acc"] * 100
    b_f1 = res["b_f1"]
    e2e = res["e2e_recall"] * 100
    print(f"{mn:<14} | A' Acc: {a_p_acc:.2f}% | A' F1: {a_p_f1:.4f} | B Acc: {b_acc:.2f}% | B F1: {b_f1:.4f} | E2E Recall: {e2e:.2f}%")

print("\n=== 3. SPEED BENCHMARK (ms/crop) ===")
for mn, ms in d["speed_benchmark_ms"].items():
    print(f"{mn:<14}: {ms:.3f} ms/crop")

print("\n=== 4. PAIRED COMPARISON (S3) ===")
for r in d["paired_comparison"]:
    print(r)

print("\n=== 5. CONFUSION ALL MODELS (Hiệu lệnh <-> Chỉ dẫn) ===")
for r in d["confusion_all_models"]:
    fn = r["Nấc đặc trưng"]
    mn = r["Mô hình"]
    muc = r["Mức"]
    h2i = r["Hiệu lệnh -> Chỉ dẫn"]
    i2h = r["Chỉ dẫn -> Hiệu lệnh"]
    tot = r["Tổng nhầm"]
    print(f"{fn:<20} | {mn:<14} | {muc:<15} | H->I: {h2i:2d} | I->H: {i2h:2d} | Tổng: {tot:2d}")

