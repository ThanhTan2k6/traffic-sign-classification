# 🚦 Hệ Thống Phân Loại & Đánh Giá Biển Báo Giao Thông (Traffic Sign Benchmarking System)

Dự án nghiên cứu và đánh giá so sánh (**Benchmarking**) các phương pháp nhận dạng biển báo giao thông Việt Nam (theo Quy chuẩn kỹ thuật quốc gia QCVN 41:2019/BGTVT).

Hệ thống được thiết kế theo tiêu chuẩn module hóa độc lập: **Mỗi thành viên phụ trách một phương pháp/mô hình riêng**, tất cả cùng chạy trên **cùng một bộ dữ liệu toàn cục cố định (Global Data Split)** để đảm bảo tính khách quan và công bằng tuyệt đối.

---

## 📋 1. BỐI CẢNH DỰ ÁN & QUY TẮC DỮ LIỆU (DATA CONTEXT & RULES)

### 🎯 4 Nhóm Biển Báo Mục Tiêu (QCVN 41:2019/BGTVT)
Dữ liệu được gom nhóm tự động từ các mã biển chi tiết về 4 nhóm biển báo chính:

| STT | Tên nhóm | Tiền tố dataset | Đặc trưng Nhận dạng (Hình dáng & Màu sắc) |
| :---: | :--- | :---: | :--- |
| **1** | **`Bien_Cam`** | `P.*` | Hình tròn, viền đỏ, nền trắng hoặc xanh dương |
| **2** | **`Bien_Nguy_Hiem`** | `W.*` | Hình tam giác đều, viền đỏ, nền vàng |
| **3** | **`Bien_Hieu_Lenh`** | `R.*` | Hình tròn, nền xanh dương |
| **4** | **`Bien_Chi_Dan`** | `I.*` | Hình vuông hoặc chữ nhật, nền xanh dương |

> ⚠️ **Lớp loại trừ (Ignored):** Thư mục ngoại cảnh `Camera` và biển phụ `B.8a` bị loại bỏ để tránh gây nhiễu cho mô hình.

---

### 🛡️ Quy Tắc Global Data Split (Chống Data Leakage Tuyệt Đối)
Để đảm bảo kết quả so sánh giữa các thành viên có giá trị khoa học, toàn bộ dự án áp dụng **Cơ chế phân chia dữ liệu toàn cục (Global Split)**:
- **Tập Train (70%)**: Dùng để huấn luyện trọng số mô hình.
- **Tập Validation (15%)**: Dùng để tinh chỉnh siêu tham số (Hyperparameter tuning / Validation loss).
- **Tập Test (15%)**: **TẬP ĐÁNH GIÁ MÙ (BLIND TEST SET)** — Tuyệt đối không được dùng trong quá trình huấn luyện. Chỉ dùng một lần duy nhất lúc chấm điểm Benchmarking.

Mọi thành viên **BẮT BUỘC** nạp dữ liệu thông qua 3 tệp định danh cố định trong `data/global_splits/`:
- `data/global_splits/train_ids.txt`
- `data/global_splits/val_ids.txt`
- `data/global_splits/test_ids.txt`

Khởi tạo danh sách split bằng lệnh:
```bash
python global_data_split.py
```

---

## 🏗️ 2. KIẾN TRÚC MÃ NGUỒN (REPOSITORY STRUCTURE)

```text
traffic-sign-classification/
├── checkpoints/                        # Thư mục lưu trọng số model (.pkl, .pt, .pth) [ĐÃ IGNORE]
│   ├── color_hog_svm/best_svm.pkl
│   ├── model_2/
│   └── model_3/
├── data/
│   ├── global_splits/                  # Danh sách chia dữ liệu cố định (train, val, test)
│   └── raw/                            # Dataset gốc (dataset_cropped_for_lenet)
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
│   │   ├── model_2/                    # 👉 Thành viên 2 (Ví dụ: Custom CNN / LeNet)
│   │   │   ├── __init__.py
│   │   │   ├── detector.py             # Kế thừa BaseDetector
│   │   │   └── train.py                # Script train theo Global Split
│   │   └── model_3/                    # 👉 Thành viên 3 (Ví dụ: YOLO / MobileNet / KNN)
│   │       ├── __init__.py
│   │       ├── detector.py             # Kế thừa BaseDetector
│   │       └── train.py                # Script train theo Global Split
│   └── evaluation/                     # BỘ ĐÁNH GIÁ CHUNG (BENCHMARKING ENGINE)
│       ├── __init__.py
│       ├── eval_metrics.py             # Tính Accuracy, Precision, Recall, F1, Latency, FPS
│       └── plotter.py                  # Xuất biểu đồ so sánh & Confusion Matrix
├── app.py                              # File điều khiển trung tâm (Evaluate / Demo)
├── global_data_split.py                # Script phân chia dữ liệu toàn cục
├── requirements.txt                    # Danh sách thư viện phụ thuộc
├── .gitignore                          # Chặn commit dataset và checkpoints nặng
└── README.md                           # Tài liệu hướng dẫn & Quy tắc nhóm
```

---

## 📜 3. QUY TẮC LÀM VIỆC NHÓM (TEAM RULES & CONTRIBUTION GUIDELINES)

Để mã nguồn luôn sạch sẽ, độc lập và tránh xung đột (conflict) khi merge code:

1. **Quy tắc Git Branching:**
   - **Nhánh `main`:** Chỉ lưu trữ **Bộ khung chuẩn (Base Skeleton)** và các phiên bản Release ổn định.
   - **Nhánh `dev`:** Nhánh tích hợp chung.
   - Mỗi thành viên tạo nhánh tính năng riêng từ `dev`:
     * Thành viên 1: `git checkout -b feature/model-1-hog-svm`
     * Thành viên 2: `git checkout -b feature/model-2`
     * Thành viên 3: `git checkout -b feature/model-3`
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
   - **Tuyệt đối không push** thư mục `data/` và các file trọng số mô hình `.pkl`, `.pt`, `.pth`, `.h5` lên GitHub (đã được cấu hình chặn bởi `.gitignore`).
   - Checkpoints huấn luyện lưu cục bộ tại `checkpoints/<tên_model>/` hoặc chia sẻ qua Google Drive.

---

## ⚙️ 4. HƯỚNG DẪN CÀI ĐẶT & SỬ DỤNG

### Bước 1: Khởi tạo môi trường ảo & Cài đặt
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install -r requirements.txt
```

### Bước 2: Khởi tạo Global Data Split
```bash
python global_data_split.py
```

### Bước 3: Huấn luyện từng mô hình
Mỗi thành viên chạy script huấn luyện trong thư mục mô hình của mình:
```bash
# Thành viên 1:
python -m src.detection.color_hog_svm.train

# Thành viên 2:
python -m src.detection.model_2.train

# Thành viên 3:
python -m src.detection.model_3.train
```

### Bước 4: Chạy Đánh giá So sánh (Benchmarking) & Demo qua `app.py`
```bash
# 1. Chạy Benchmark so sánh toàn bộ các mô hình trên tập Test:
python app.py --mode evaluate --split test

# 2. Chạy đánh giá 1 mô hình cụ thể trên tập Val:
python app.py --mode evaluate --model color_hog_svm --split val

# 3. Chạy thử nghiệm Demo 1 mô hình trên 1 ảnh bất kỳ:
python app.py --mode demo --model color_hog_svm --input path/to/image.png
```
