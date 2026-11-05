# HƯỚNG DẪN KỸ THUẬT & DANH MỤC MÃ NGUỒN (`code/`)

Thư mục này chứa toàn bộ mã nguồn thực thi độc lập của hệ thống nhận diện biển báo giao thông Việt Nam 2 chặng (Detector YOLO11s 1-class + Bộ phân loại đặc trưng truyền thống đa tầng).

---

## 1. Danh mục các Script & Vai trò chi tiết

| Tên tập tin | Vai trò / Nhiệm vụ trong Pipeline | Đầu vào kỳ vọng (Inputs) | Đầu ra chính (Outputs) |
|---|---|---|---|
| `common.py` | Thư viện hàm dùng chung: trích xuất đặc trưng, IOU, padding crop, phân chia block chống rò rỉ (`block_split`). | Dữ liệu ảnh và mảng tọa độ. | Các hàm tiện ích hỗ trợ toàn bộ pipeline. |
| `prepare_data.py` | Chuẩn bị dữ liệu: đọc dataset gốc, chạy `block_split` (block 100 frames, gap 5 frames) để tạo tập train/val/test không rò rỉ, xuất `manifest.csv` và format YOLO. | `archive_new/archive` (ảnh & XML/TXT gán nhãn). | `data/vn/manifest.csv`, `data/vn/yolo/data.yaml`. |
| `train_detector.py` | Huấn luyện detector YOLO11s 1 lớp: pretrain CCTSDB hoặc finetune trên dataset Việt Nam (`data/vn/yolo/data.yaml`). | File cấu hình `data/vn/yolo/data.yaml`. | Checkpoints `runs/detect/vn_s/weights/best.pt`. |
| `stage1_classify.py` | Baseline Stage 1: Cắt crop biển báo, trích xuất HSV toàn cục 44 chiều, huấn luyện SVM RBF và kNN phân loại vào 6 nhóm ban đầu. | `data/vn/manifest.csv`, ảnh gốc. | `models/stage1_hsv_vn_best.joblib`, báo cáo Stage 1. |
| `evaluate_for_report.py`| Đo đạc bổ sung cho báo cáo Stage 1: đánh giá Mức A', False Positives detector, xuất biểu đồ histogram màu theo nhóm. | Checkpoint Stage 1, weights YOLO. | `reports/stage1_classify_vn/fig_hist_by_group.png`. |
| `audit_counts.py` | Kiểm toán phân bố số lượng mẫu thực tế trong manifest, kiểm tra chi tiết các lớp đặc biệt (`Camera`, `B.8a`). | `data/vn/manifest.csv`. | Thống kê số lượng frame, bbox, và dải stem liên tục. |
| `stage2_pipeline.py` | Pipeline chính Stage 2: Sửa nhãn (gộp B.8a, loại Camera), xây dựng chuỗi đặc trưng $S0 \rightarrow S3$ (tối đa 457 dims), GroupKFold 5-fold, đánh giá A/A'/B và End-to-End Recall. | `data/vn/manifest.csv`, weights detector YOLO. | `models/stage2_best.joblib`, `reports/stage2_vn/stage2_metrics.json`. |
| `run_stage2_update.py` | Benchmark nâng cao: Huấn luyện và so sánh thêm **XGBoost** và **LightGBM**, tính paired difference từng fold, đo tốc độ suy luận CPU trên 1.000 crop. | Cache đặc trưng Stage 2. | `reports/stage2_vn/stage2_update_metrics.json`, `reports/stage2_vn/chain.png`. |
| `train_classifiers.py` | Script huấn luyện tổng quát nhiều biến thể phân loại (SVM, RF, kNN, Boosting) theo tham số CLI. | Tệp cache đặc trưng hoặc manifest. | Checkpoints phân loại lưu trong `models/`. |
| `eval_pipeline.py` | Đánh giá tổng hợp cuối kỳ trên toàn bộ tập Test theo các kịch bản kích thước và điều kiện môi trường. | Weights detector và model classifiers. | Bảng tổng hợp hiệu năng end-to-end. |

---

## 2. Thứ tự thực thi chuẩn (Execution Workflow)

Để tái hiện lại từ đầu toàn bộ hệ thống:

```powershell
# Bước 1: Chuẩn bị dữ liệu và chia split chống rò rỉ
python code/prepare_data.py --vn-root <đường_dẫn_dataset_gốc> --out data

# Bước 2: Huấn luyện detector YOLO11s 1 lớp
python code/train_detector.py --data data/vn/yolo/data.yaml --epochs 30 --imgsz 800

# Bước 3: Huấn luyện Baseline Stage 1 (HSV 44 dims)
python code/stage1_classify.py

# Bước 4: Chạy toàn bộ pipeline nâng cấp Stage 2 (Chuỗi S0->S3, GroupKFold, Đánh giá Mức A/A'/B)
python code/stage2_pipeline.py

# Bước 5: Chạy benchmark mở rộng XGBoost & LightGBM và đo tốc độ CPU
python code/run_stage2_update.py
```

---

## 3. Rà soát các đường dẫn mặc định cần cấu hình khi chạy ở máy khác

Khi sao chép mã nguồn sang máy tính mới, cần lưu ý các đường dẫn mặc định sau (nếu dữ liệu không đặt ở vị trí mặc định):

1. **`code/prepare_data.py`**:
   - Dòng 243: `ap.add_argument("--cctsdb-root", default="archive")` $\rightarrow$ Đổi thành đường dẫn thư mục chứa dataset CCTSDB nếu muốn pretrain.
   - Dòng 244: `ap.add_argument("--vn-root", default="archive_new/archive")` $\rightarrow$ Đổi thành đường dẫn thư mục chứa dataset Việt Nam gốc (`images/` và `labels/`).
2. **`code/stage2_pipeline.py`**:
   - Dòng 173: `manifest_path = Path("data/vn/manifest.csv")` $\rightarrow$ Yêu cầu file manifest đã được sinh từ Bước 1.
   - Dòng 617: `yolo_weights = Path("runs/detect/vn_s/weights/best.pt")` $\rightarrow$ Cần tải trọng số YOLO về thư mục này theo hướng dẫn trong `WEIGHTS.md`.
3. **`code/run_stage2_update.py`**:
   - Dòng 69: `feat_cache_file = Path("data/stage2_feat_cache.joblib")` $\rightarrow$ Cần chạy `stage2_pipeline.py` trước để sinh cache hoặc trích xuất trực tiếp.

---

## 4. Yêu cầu môi trường & Phiên bản thư viện đã kiểm chứng

Toàn bộ mã nguồn đã được thực thi và kiểm chứng tính tương thích trên môi trường:
- **Hệ điều hành**: Microsoft Windows 11 (x64) / Linux Ubuntu 22.04 LTS
- **Python**: 3.11.9
- **Thư viện cốt lõi**:
  - `torch==2.6.0+cu124`
  - `ultralytics==8.4.198` (YOLO)
  - `scikit-learn==1.6.1`
  - `scikit-image==0.25.2`
  - `opencv-python==4.11.0.86`
  - `xgboost==3.2.0`
  - `lightgbm==4.7.0`
  - `pandas==2.2.3`
  - `numpy==2.2.3`
  - `matplotlib==3.10.1`
  - `seaborn==0.13.2`
  - `joblib==1.4.2`
---

## Ghi chú kỹ thuật về nhãn "1538" trong mã nguồn

Trong file `stage2_pipeline.py` (dòng 771, 773) và một số key trong file kết quả JSON ban đầu (`stage2_metrics.json`), bạn có thể bắt gặp chuỗi ký tự `"A' Acc (1538)"` hoặc `"B Acc (1538)"`.
- **Bản chất kỹ thuật**: Con số `1538` là **nhãn chuỗi văn bản cũ (legacy dictionary key string)** được người viết code gõ tay vào tiêu đề cột trước khi chạy thực nghiệm.
- **Mẫu số đánh giá thực tế**: Mảng nhãn thực thi `matched_gt_y` và tập crop kiểm thử của detector ở Mức ($A'$) và Mức ($B$) chứa đúng **1.559** biển báo (đã loại trừ 24 biển Camera khỏi GT; YOLO bắt trúng 0/24 Camera và 1.559/1.821 biển GT 5 nhóm). Số phép dự đoán đúng của SVM tại Mức B là $1.546 / 1.559 = 99.17\%$.
- Để bảo toàn tính toàn vẹn của mã nguồn gốc, các script trong thư mục `code/` giữ nguyên logic và tên biến ban đầu.
