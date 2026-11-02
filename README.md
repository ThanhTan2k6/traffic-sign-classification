# 🚦 Hệ Thống Phân Loại & Đánh Giá Biển Báo Giao Thông (Traffic Sign Benchmarking System)

Dự án nghiên cứu và đánh giá so sánh (**Benchmarking**) các phương pháp nhận dạng biển báo giao thông Việt Nam (theo Quy chuẩn kỹ thuật quốc gia QCVN 41:2019/BGTVT).

Hệ thống được thiết kế theo kiến trúc chuẩn hóa (Framework Base Skeleton): **Mỗi thành viên phụ trách một phương pháp/mô hình riêng**, tất cả cùng chạy trên **cùng một bộ dữ liệu toàn cục cố định (Global Data Split)** để đảm bảo tính khách quan và công bằng tuyệt đối.

---

## 📋 1. DỮ LIỆU & QUY TẮC PHÂN CHIA (DATA CONTEXT & RULES)

### 🎯 4 Nhóm Biển Báo Mục Tiêu (QCVN 41:2019/BGTVT)
Dữ liệu biển báo giao thông Việt Nam gồm 52 lớp chi tiết được gom nhóm tự động về 4 nhóm biển báo chính:

| STT | Tên nhóm | Tiền tố dataset | Đặc trưng Nhận dạng (Hình dáng & Màu sắc) |
| :---: | :--- | :---: | :--- |
| **1** | **`Bien_Cam`** | `P.*` | Hình tròn, viền đỏ, nền trắng hoặc xanh dương |
| **2** | **`Bien_Nguy_Hiem`** | `W.*` | Hình tam giác đều, viền đỏ, nền vàng |
| **3** | **`Bien_Hieu_Lenh`** | `R.*` | Hình tròn, nền xanh dương |
| **4** | **`Bien_Chi_Dan`** | `I.*` | Hình vuông hoặc chữ nhật, nền xanh dương |

> ⚠️ **Lớp loại trừ (Ignored):** Nhóm biển phụ (`S.*`, `B.8a`) và camera giám sát ngoại cảnh (`Camera`) được tự động loại bỏ để tránh gây nhiễu cho mô hình phân loại và phát hiện.

---

### 📊 Thống Kê Bộ Dữ Liệu Mới (Dataset Overview)
- **Tổng số ảnh toàn cảnh:** 3,216 ảnh thực tế đường phố Việt Nam (độ phân giải 960x540 và 1920x1080).
- **Định dạng nhãn gốc:** YOLO format (`<class_id> <x_center> <y_center> <width> <height>`).
- **Tổng số bounding box được gán nhãn:** 8,334 boxes thuộc 52 lớp chuẩn.
- **Tổng số bounding box mục tiêu sau lọc:** ~7,858 boxes biển báo (đạt chuẩn kích thước $\ge 10$px và thuộc 4 nhóm chính).

---

### 🛡️ Cơ Chế Global Data Split (Chống Data Leakage Tuyệt Đối)
Để đảm bảo so sánh công bằng giữa các thành viên:
1. **Phân chia ở cấp độ ẢNH (Image-level split):** Nếu một ảnh chứa nhiều biển báo, toàn bộ các biển báo này cùng nằm trong một tập dữ liệu duy nhất, tuyệt đối không bị rò rỉ (Zero Data Leakage).
2. **Tỉ lệ phân chia toàn cục:**
   - **Tập Train (70%):** Huấn luyện trọng số mô hình.
   - **Tập Validation (15%):** Tinh chỉnh siêu tham số (Hyperparameter tuning / Validation loss).
   - **Tập Test (15%):** **TẬP ĐÁNH GIÁ MÙ (BLIND TEST SET)** — Dùng một lần duy nhất lúc chấm điểm Benchmarking cuối cùng.

3. **Hỗ trợ 2 luồng bài toán song song:**
   - **Luồng Phân loại (Classification - SVM, LeNet, CNN):** Tự động cắt các vùng chứa biển báo (ROI Cropping) lưu tại `data/cropped/{train,val,test}/{category}/` và định danh qua 3 file text trong `data/global_splits/`:
     * `data/global_splits/train_ids.txt`
     * `data/global_splits/val_ids.txt`
     * `data/global_splits/test_ids.txt`
   - **Luồng Phát hiện (Object Detection - YOLO):** Cung cấp danh sách ảnh toàn cảnh và cấu hình YOLO chuẩn:
     * `data/global_splits/yolo_train.txt`
     * `data/global_splits/yolo_val.txt`
     * `data/global_splits/yolo_test.txt`
     * `data/traffic_signs_4cls.yaml`

Khởi tạo toàn bộ dữ liệu split và crop ROI bằng một lệnh duy nhất:
```bash
python global_data_split.py
```

---

## 🏗️ 2. KIẾN TRÚC MÃ NGUỒN (REPOSITORY STRUCTURE)

```text
traffic-sign-classification/
├── checkpoints/                        # Trọng số mô hình (.pkl, .pt, .pth) [ĐÃ IGNORE]
├── data/                               # Dữ liệu xử lý trung gian
│   ├── cropped/                        # [TỰ ĐỘNG SINH] Ảnh ROI biển báo cắt sẵn theo nhóm
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── global_splits/                  # Danh sách split toàn cục (train_ids, val_ids, test_ids, yolo_*.txt)
│   └── traffic_signs_4cls.yaml         # File config huấn luyện YOLO (4 lớp QCVN 41:2019)
├── dataset/                            # Dataset gốc (YOLO Format) [IMAGES & LABELS ĐÃ IGNORE]
│   ├── images/                         # 3,216 ảnh đường phố (.jpg)
│   ├── labels/                         # 3,216 file nhãn YOLO (.txt)
│   ├── classes.txt                     # 52 mã hiệu biển báo
│   ├── classes_vie.txt                 # Tên tiếng Việt của 52 biển báo
│   └── classes_en.txt                  # Tên tiếng Anh của 52 biển báo
├── notebooks/                          # Jupyter Notebooks phân tích dữ liệu EDA
├── src/
│   ├── config.py                       # Cấu hình tập trung (Context, Paths, Classes, Seed)
│   ├── utils/                          # Tiện ích đọc/ghi ảnh Unicode tiếng Việt an toàn
│   │   ├── __init__.py
│   │   └── image_io.py
│   ├── detection/                      # CÁC PHƯƠNG PHÁP NHẬN DẠNG (MỖI NGƯỜI 1 THƯ MỤC)
│   │   ├── __init__.py
│   │   ├── base_detector.py            # [RULE] Lớp trừu tượng chuẩn hóa Interface
│   │   ├── color_hog_svm/              # 👉 Thành viên 1 (Truyền thống: HSV + Morphology + HOG + SVM)
│   │   │   ├── __init__.py
│   │   │   ├── detector.py             # Kế thừa BaseDetector
│   │   │   └── train.py                # Script train theo Global Split
│   │   ├── model_2/                    # 👉 Thành viên 2 (Deep Learning: Custom CNN / LeNet)
│   │   │   ├── __init__.py
│   │   │   ├── detector.py             # Kế thừa BaseDetector
│   │   │   └── train.py                # Script train theo Global Split
│   │   └── model_3/                    # 👉 Thành viên 3 (Object Detection: YOLO / MobileNet)
│   │       ├── __init__.py
│   │       ├── detector.py             # Kế thừa BaseDetector
│   │       └── train.py                # Script train theo Global Split
│   └── evaluation/                     # BỘ ĐÁNH GIÁ CHUNG (BENCHMARKING ENGINE)
│       ├── __init__.py
│       ├── eval_metrics.py             # Tính Accuracy, Precision, Recall, F1, Latency, FPS, Leaderboard
│       └── plotter.py                  # Xuất biểu đồ so sánh & Heatmap Confusion Matrix
├── app.py                              # Controller trung tâm (Evaluate Benchmark & Demo ảnh)
├── global_data_split.py                # Pipeline phân chia dữ liệu toàn cục & Cắt ROI
├── requirements.txt                    # Danh sách thư viện phụ thuộc
├── .gitignore                          # Chặn commit dataset ảnh và checkpoints nặng
└── README.md                           # Tài liệu hướng dẫn & Quy tắc nhóm
```

---

## 📜 3. QUY TẮC LÀM VIỆC NHÓM (TEAM RULES & CONTRIBUTION)

1. **Quy tắc Git Branching:**
   - **Nhánh `main`:** Chỉ lưu trữ **Bộ khung chuẩn (Base Skeleton)** và các phiên bản Release ổn định.
   - **Nhánh `dev`:** Nhánh tích hợp chung.
   - Mỗi thành viên tạo nhánh tính năng riêng từ `dev`:
     * Thành viên 1: `git checkout -b feature/model-1-hog-svm`
     * Thành viên 2: `git checkout -b feature/model-2-cnn`
     * Thành viên 3: `git checkout -b feature/model-3-yolo`
   - Khi hoàn thành, tạo Pull Request (PR) merge vào `dev`.

2. **Quy tắc Interface (Bắt buộc):**
   - Mọi detector tại `src/detection/<tên_model>/detector.py` **bắt buộc phải kế thừa class `BaseDetector`** từ `src/detection/base_detector.py`.
   - Kết quả trả về của hàm `postprocess()` bắt buộc phải tuân theo cấu trúc dict:
     ```python
     {
         "class_name": str,       # Nhãn lớp ('Bien_Cam', 'Bien_Nguy_Hiem', ...)
         "class_id": int,         # ID nhãn (0, 1, 2, 3)
         "confidence": float,     # Độ tin cậy từ 0.0 đến 1.0
         "bbox": tuple | None     # (x1, y1, x2, y2) nếu có
     }
     ```

3. **Quy tắc Dữ liệu & Checkpoints:**
   - **Tuyệt đối không push** `dataset/images/`, `dataset/labels/`, `data/cropped/` và các file trọng số `.pkl`, `.pt`, `.pth` lên GitHub (đã được cấu hình tự động trong `.gitignore`).

---

## ⚙️ 4. HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG

### Bước 1: Khởi tạo môi trường ảo & Cài đặt thư viện
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install -r requirements.txt
```

### Bước 2: Khởi tạo Global Data Split & Cắt ROI Biển Báo
```bash
python global_data_split.py
```
> Lệnh này sẽ tự động phân chia 70/15/15 và cắt các ROI biển báo lưu vào `data/cropped/`, đồng thời xuất các file danh sách split trong `data/global_splits/`.

### Bước 3: Huấn luyện từng mô hình
Mỗi thành viên thực hiện huấn luyện mô hình của mình độc lập:
```bash
# Thành viên 1 (HOG + SVM):
python -m src.detection.color_hog_svm.train

# Thành viên 2 (Custom CNN / LeNet):
python -m src.detection.model_2.train

# Thành viên 3 (YOLO / MobileNet):
python -m src.detection.model_3.train
```

### Bước 4: Chạy Đánh giá So sánh (Benchmarking) & Demo qua `app.py`
```bash
# 1. Chạy Benchmark so sánh toàn bộ các mô hình trên tập Test:
python app.py --mode evaluate --split test

# 2. Chạy đánh giá 1 mô hình cụ thể trên tập Val:
python app.py --mode evaluate --model color_hog_svm --split val

# 3. Chạy thử nghiệm Demo 1 mô hình trên 1 ảnh bất kỳ:
python app.py --mode demo --model color_hog_svm --input dataset/images/0001.jpg
```
