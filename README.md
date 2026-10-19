# 🚦 ĐỒ ÁN THỊ GIÁC MÁY TÍNH: PHÁT HIỆN & PHÂN LOẠI BIỂN BÁO GIAO THÔNG

Dự án xây dựng hệ thống phát hiện (Detection) và phân loại (Classification) biển báo giao thông Việt Nam sử dụng các kỹ thuật Xử lý ảnh truyền thống kết hợp Học máy cổ điển (HSV Color Masking + Morphology + Contours + HOG + SVM/k-NN).

---

## 📌 1. KIẾN TRÚC HỆ THỐNG (SYSTEM ARCHITECTURE)

```
 [ Ảnh đầu vào (Input Image: RGB) ]
                 │
                 ▼
 ┌──────────────────────────────────────────────┐
 │ GIAI ĐOẠN 1: TIỀN XỬ LÝ & LỌC MÀU (COLOR)     │
 │ - Chuyển không gian màu RGB ➔ HSV             │
 │ - Tạo mặt nạ lọc màu (Đỏ, Vàng, Xanh dương)   │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │ GIAI ĐOẠN 2: PHÂN ĐOẠN VÀ TÌM ROI (DETECTION) │
 │ - Khử nhiễu bằng bộ lọc hình thái học         │
 │ - Tìm đường viền (Contours Analysis)          │
 │ - Lọc theo diện tích & tỷ lệ khung hình       │
 │ - Cắt vùng quan tâm (ROI - Region of Interest)│
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │ GIAI ĐOẠN 3: TRÍCH XUẤT ĐẶC TRƯNG (FEATURES)  │
 │ - Chuẩn hóa kích thước vùng ROI (ví dụ 64x64) │
 │ - Trích xuất đặc trưng hình dáng/cấu trúc     │
 │   bằng thuật toán HOG (Histogram of Gradients)│
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │ GIAI ĐOẠN 4: HUẤN LUYỆN & PHÂN LOẠI (MODEL)   │
 │ - Sử dụng mô hình học máy SVM (hoặc k-NN)     │
 │ - Dự đoán nhãn lớp của biển báo               │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 [ Kết quả hiển thị (Visualized Output & Label) ]
```

---

## 👥 2. PHÂN CÔNG NHIỆM VỤ NHÓM

| Thành viên | Trách nhiệm chính | Khối công việc cụ thể |
| :--- | :--- | :--- |
| **Thành viên 1** | **Dữ liệu, Tiền xử lý & Khoanh vùng** | • Làm sạch dataset (loại bỏ ảnh hỏng, loại bỏ folder `Camera`).<br>• Tăng cường dữ liệu (Augmentation) cân bằng lớp.<br>• Triển khai **Giai đoạn 1** (RGB ➔ HSV, Color Mask Đỏ/Xanh/Vàng).<br>• Triển khai **Giai đoạn 2** (Morphology, Contours, Lọc bbox, Cắt & chuẩn hóa ROI 64x64).<br>• Xuất tập dữ liệu ROI chuẩn giao cho TV2. |
| **Thành viên 2** | **Trích xuất đặc trưng & Huấn luyện mô hình** | • Triển khai **Giai đoạn 3** (Trích xuất vector đặc trưng HOG).<br>• Triển khai **Giai đoạn 4** (Huấn luyện SVM / k-NN bằng scikit-learn, chia Train/Test 80/20).<br>• Tinh chỉnh siêu tham số (Hyperparameter Tuning), xuất `traffic_sign_svm.pkl`.<br>• Đo đạc Accuracy, Precision, Recall, F1 và vẽ biểu đồ Confusion Matrix. |
| **Thành viên 3** | **Tích hợp, Giao diện (UI) & Báo cáo** | • Tích hợp luồng End-to-End Pipeline hoàn chỉnh.<br>• Xây dựng giao diện Web UI tương tác bằng **Streamlit** (hỗ trợ upload ảnh & Webcam).<br>• Soạn Slide thuyết trình và viết báo cáo Word/PDF. |

---

## 📁 3. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
traffic-sign-classification/
├── data/                               # Dữ liệu ảnh (không push lên git)
│   ├── cleaned/                        # Dữ liệu sau khi làm sạch
│   ├── augmented/                      # Dữ liệu sau khi cân bằng lớp
│   └── rois/                           # Dữ liệu ROI chuẩn hóa 64x64 giao cho TV2
├── dataset_cropped_for_lenet/          # Thư mục dataset gốc
├── models_saved/                       # Thư mục chứa trọng số mô hình (.pkl)
├── notebooks/                          # Jupyter Notebooks nghiên cứu & phân tích
│   └── eda_dataset.py                  # Script phân tích khám phá dữ liệu (EDA)
├── src/                                # Mã nguồn chính
│   ├── __init__.py
│   ├── config.py                       # Cấu hình tập trung (HSV, ROI size, HOG, Paths)
│   ├── utils.py                        # Hàm tiện ích (đọc ảnh tiếng Việt, visualize)
│   ├── preprocessing/                  # [TV1] Khối làm sạch & tăng cường ảnh
│   │   ├── __init__.py
│   │   ├── cleaner.py                  # Loại bỏ Camera & ảnh hỏng
│   │   └── augmentor.py                # Biến đổi hình học, cân bằng lớp
│   ├── detection/                      # [TV1] Khối Giai đoạn 1 & Giai đoạn 2
│   │   ├── __init__.py
│   │   ├── color_filter.py             # Lọc màu HSV (Đỏ, Vàng, Xanh)
│   │   └── roi_extractor.py            # Morphology, Contours & Cắt ROI
│   ├── features/                       # [TV2] Khối Giai đoạn 3
│   │   ├── __init__.py
│   │   └── hog_extractor.py            # Trích xuất đặc trưng HOG
│   ├── models/                         # [TV2] Khối Giai đoạn 4
│   │   ├── __init__.py
│   │   ├── train_svm.py                # Huấn luyện mô hình SVM / k-NN
│   │   └── evaluate.py                 # Đánh giá và vẽ Confusion Matrix
│   ├── pipeline/                       # [TV3] Khối tích hợp End-to-End
│   │   ├── __init__.py
│   │   └── end_to_end.py               # Kết nối 4 giai đoạn thành luồng duy nhất
│   └── extract_roi_dataset.py          # Script xuất batch ROI chuẩn hóa cho TV2
├── app.py                              # [TV3] Ứng dụng Web UI Streamlit
├── main_tv1.py                         # [TV1] Script thực thi trọn gói nhiệm vụ TV1
├── requirements.txt                    # Danh sách thư viện phụ thuộc
├── .gitignore                          # Cấu hình bỏ qua file nặng/tạm khi commit
└── README.md                           # Tài liệu hướng dẫn dự án
```

---

## 🚀 4. HƯỚNG DẪN CÀI ĐẶT

### Bước 1: Tạo môi trường ảo & Kích hoạt
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt trên Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Hoặc trên Command Prompt (cmd):
.\venv\Scripts\activate.bat
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install -r requirements.txt
```

---

## 🛠️ 5. HƯỚNG DẪN CHẠY TỪNG PHẦN

### 👉 Dành cho Thành viên 1:
```bash
# 1. Chạy demo trực quan hóa từng bước trên 1 ảnh mẫu ngẫu nhiên:
python main_tv1.py --demo

# 2. Làm sạch dataset (loại bỏ thư mục Camera và ảnh hỏng):
python main_tv1.py --clean

# 3. Tăng cường dữ liệu (Augmentation) để cân bằng các lớp:
python main_tv1.py --augment

# 4. Xuất toàn bộ tập ROI chuẩn 64x64 đưa vào data/rois/ giao cho TV2:
python main_tv1.py --export-rois

# Hoặc chạy toàn bộ 3 bước trên bằng 1 lệnh duy nhất:
python main_tv1.py --all
```

### 👉 Dành cho Thành viên 2:
```bash
# 1. Huấn luyện mô hình SVM với đặc trưng HOG:
python -m src.models.train_svm

# 2. Đánh giá mô hình và vẽ ma trận nhầm lẫn (Confusion Matrix):
python -m src.models.evaluate
```

### 👉 Dành cho Thành viên 3:
```bash
# Khởi chạy giao diện Web tương tác:
streamlit run app.py
```

---

## 🤝 6. QUY TẮC LÀM VIỆC VỚI GIT & GITHUB

Để nhóm làm việc nhịp nhàng, không bị ghi đè hay xung đột mã nguồn (conflict):
1. **Tuyệt đối không commit dữ liệu ảnh** (`data/`, `dataset_*/`) và file trọng số lớn lên Git (đã được cấu hình trong `.gitignore`).
2. Mỗi thành viên làm việc trên thư mục/file của mình theo đúng phân công:
   - Thành viên 1: `src/preprocessing/`, `src/detection/`, `main_tv1.py`
   - Thành viên 2: `src/features/`, `src/models/`
   - Thành viên 3: `src/pipeline/`, `app.py`
