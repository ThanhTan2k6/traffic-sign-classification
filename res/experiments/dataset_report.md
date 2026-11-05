# Báo Cáo Khảo Sát Dataset CCTSDB2021 (China City Traffic Sign Database 2021)

## 1. Tóm Tắt Tổng Quan (Executive Summary)
Dataset này là **CCTSDB2021 (China City Traffic Sign Database 2021)**, một bộ dữ liệu chuẩn được thu thập từ camera hành trình trên đường phố đô thị Trung Quốc, thiết kế nguyên bản cho bài toán **Object Detection (Phát hiện biển báo giao thông)** đa đối tượng trên ảnh độ phân giải thực tế. Bộ dữ liệu chứa tổng cộng **18.356 ảnh** (16.356 train, 1.500 test, 500 ảnh nền âm tính) với **30.066 bounding box** thuộc 3 nhóm biển báo chính: `prohibitory` (cấm), `mandatory` (hiệu lệnh) và `warning` (nguy hiểm/cảnh báo). Toàn bộ 17.856 ảnh trong tập train/test đều có chú thích đầy đủ cả ở chuẩn Pascal VOC XML và YOLO TXT, không có file XML nào bị hỏng cú pháp. Điểm cốt lõi cần lưu ý là **71.8% biển báo có kích thước rất nhỏ (<32px)**, có sự mất cân bằng giữa các lớp (tỉ lệ 3.26:1), và script phụ trợ `voc_label.py` đi kèm bị sai lệch nghiêm trọng về tên lớp và thứ tự nhãn.

---

## 2. Cấu Trúc Thư Mục & Kiểm Tra Toàn Vẹn File

### 2.1 Cây thư mục chi tiết (Tối đa 3 cấp)
```text
archive/
└── CCTSDB2021/ (1 file: voc_label.py)
    ├── XML/ (17.856 file .xml)
    ├── test/
    │   ├── images/ (1.500 file .jpg)
    │   └── labels/ (1.500 file .txt)
    ├── train/
    │   ├── images/ (16.356 file .jpg)
    │   └── labels/ (16.356 file .txt)
    ├── 测试集天气光照情况分类/ (Tập test phân loại theo thời tiết & ánh sáng)
    │   ├── cloud/ (300 file .xml)
    │   ├── foggy/ (40 file .xml)
    │   ├── night/ (500 file .xml)
    │   ├── rain/ (160 file .xml)
    │   ├── snow/ (100 file .xml)
    │   └── sunny/ (400 file .xml)
    ├── 测试集尺寸大小分类/ (Tập test phân loại theo kích cỡ biển báo)
    │   ├── XS/ (231 file .xml)
    │   ├── S/ (239 file .xml)
    │   ├── M/ (362 file .xml)
    │   ├── L/ (148 file .xml)
    │   └── XL/ (124 file .xml)
    └── 负样本图像/ (500 file .jpg - Ảnh âm tính không chứa biển báo)
```

### 2.2 Thống kê số lượng file theo định dạng và thư mục
| Thư mục | .xml | .jpg | .txt | .py | Tổng file |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `archive/CCTSDB2021/` (root) | 0 | 0 | 0 | 1 | 1 |
| `archive/CCTSDB2021/XML/` | 17.856 | 0 | 0 | 0 | 17.856 |
| `archive/CCTSDB2021/train/images/` | 0 | 16.356 | 0 | 0 | 16.356 |
| `archive/CCTSDB2021/train/labels/` | 0 | 0 | 16.356 | 0 | 16.356 |
| `archive/CCTSDB2021/test/images/` | 0 | 1.500 | 0 | 0 | 1.500 |
| `archive/CCTSDB2021/test/labels/` | 0 | 0 | 1.500 | 0 | 1.500 |
| `archive/CCTSDB2021/测试集天气光照情况分类/` | 1.500 | 0 | 0 | 0 | 1.500 |
| `archive/CCTSDB2021/测试集尺寸大小分类/` | 1.104 | 0 | 0 | 0 | 1.104 |
| `archive/CCTSDB2021/负样本图像/` | 0 | 500 | 0 | 0 | 500 |
| **Tổng cộng toàn bộ dataset** | **20.460** | **18.356** | **17.856** | **1** | **56.673** |

### 2.3 So sánh đối chiếu với `archive.zip`
- Tổng số file trong `archive.zip`: **56.673 file**.
- Tổng số file trong thư mục đã giải nén `archive/`: **56.673 file**.
- Số file khớp 100% kích thước và tên: **56.673 / 56.673**.
- Sai lệch kích thước (Size mismatch): **0**.
- File thiếu trong bản giải nén: **0**.
- File thừa trong bản giải nén: **0**.
- **Kết luận**: Thư mục `archive/` trùng khớp hoàn toàn 100% với file nén gốc `archive.zip`, **không cần giải nén lại**.

---

## 3. Ý Nghĩa Từng Thư Mục & Kiểm Tra License/README

### 3.1 Dịch nghĩa và mục đích sử dụng các thư mục
1. **`XML`**: Chứa toàn bộ 17.856 file nhãn XML chuẩn Pascal VOC tương ứng cho cả hai tập train (16.356) và test (1.500).
2. **`train`**: Chứa dữ liệu huấn luyện chính, được chia thành:
   - `images/`: 16.356 ảnh RGB (đánh số từ `00000.jpg` đến `18991.jpg`).
   - `labels/`: 16.356 file text chuẩn YOLO tương ứng.
3. **`test`**: Chứa dữ liệu kiểm thử chuẩn, chia thành:
   - `images/`: 1.500 ảnh RGB (đánh số từ `18992.jpg` đến `20491.jpg`).
   - `labels/`: 1.500 file text chuẩn YOLO tương ứng.
4. **`测试集天气光照情况分类` (Phân loại tập test theo điều kiện thời tiết & ánh sáng)**:
   - Mục đích: Dùng để benchmark và đánh giá chuyên sâu độ bền vững (robustness) của mô hình dưới các điều kiện môi trường bất lợi.
   - Gồm 6 nhóm (tổng 1.500 file XML, bao phủ chính xác 100% tập test):
     - `cloud` (Trời nhiều mây / âm u): 300 file XML.
     - `foggy` (Sương mù / tầm nhìn thấp): 40 file XML.
     - `night` (Ban đêm / thiếu sáng trầm trọng): 500 file XML.
     - `rain` (Trời mưa / ướt mặt đường, nhòe kính): 160 file XML.
     - `snow` (Tuyết rơi / tuyết phủ trắng mặt đường): 100 file XML.
     - `sunny` (Trời nắng / ánh sáng ban ngày rõ nét): 400 file XML.
5. **`测试集尺寸大小分类` (Phân loại tập test theo kích cỡ biển báo)**:
   - Mục đích: Đánh giá khả năng phát hiện vật thể nhỏ và đa tỉ lệ (Multi-scale / Small Object Detection).
   - Gồm 5 thư mục chứa 1.104 file XML ứng với 5 ngưỡng diện tích:
     - `XS` (Extra Small - Cực nhỏ, diện tích 28 - 210 px²): 231 file.
     - `S` (Small - Nhỏ, diện tích 216 - 399 px²): 239 file.
     - `M` (Medium - Trung bình, diện tích 400 - 992 px²): 362 file.
     - `L` (Large - Lớn, diện tích 1.020 - 1.980 px²): 148 file.
     - `XL` (Extra Large - Cực lớn, diện tích 2.009 - 16.104 px²): 124 file.
   - *Ghi chú lý giải con số 1.104*: Có đúng **396 ảnh test không nằm trong phân loại này** vì các ảnh đó chứa nhiều biển báo có kích cỡ chéo nhau (ví dụ một biển kích cỡ S và một biển kích cỡ L trong cùng một ảnh) nên tác giả không gán vào một thư mục kích thước đơn nhất.
6. **`负样本图像` (Ảnh mẫu âm tính - Background / Negative Samples)**:
   - Chứa 500 ảnh đường phố/phong cảnh hoàn toàn **không có bất kỳ biển báo nào**.
   - Mục đích trong ML: Cung cấp ảnh nền âm tính để mô hình học không kích hoạt nhận diện nhầm (giảm False Positives).
7. **Kiểm tra file README / License**:
   - Quét toàn bộ kho lưu trữ: **Hoàn toàn không có file `README.md`, `LICENSE`, `info.txt` hay tài liệu nào đi kèm**.

---

## 4. Phân Tích Script `voc_label.py` & Phát Hiện Lỗi Nguy Hiểm

File `archive/CCTSDB2021/voc_label.py` là một script tiện ích thường được dùng trong cộng đồng Darknet/YOLO để chuyển đổi nhãn từ XML Pascal VOC sang nhãn text YOLO:
- **Hành vi code**:
  1. Đọc XML từ thư mục tương đối `./data/Annotations/%s`.
  2. Parse tọa độ `(xmin, xmax, ymin, ymax)` và chuẩn hóa về dạng YOLO: `(x_center, y_center, width, height)` chia cho kích thước ảnh `(w, h)`.
  3. Ghi file text ra `./data/labels/%s.txt`.
  4. Thực hiện chia tập ngẫu nhiên theo tỉ lệ để sinh file `train.txt` và `test.txt`.
- **Đường dẫn kỳ vọng**: Script kỳ vọng cấu trúc thư mục dạng VOC chuẩn: `./data/Annotations`, `./data/labels`, `./data/images`. Trong khi cấu trúc thực tế của dataset này là `XML/`, `train/images`, `train/labels`, v.v.
- **3 LỖI NGHIÊM TRỌNG TRONG SCRIPT**:
  1. **Lệch tên nhãn (Fatal Class Name Typo)**: Script khai báo `classes = ["prohibitory", "mandatory", "danger"]`. Tuy nhiên, trong toàn bộ 17.856 file XML gốc, tên lớp được gán là **`warning`**, hoàn toàn **không có lớp nào tên là `danger`**. Do code có lệnh `if cls not in classes: continue`, nếu chạy script này thì **toàn bộ 4.931 nhãn warning sẽ bị vứt bỏ hoàn toàn**!
  2. **Sai lệch thứ tự ID lớp**: Trong các file `.txt` có sẵn ở `train/labels/` và `test/labels/`, tác giả dataset đã sử dụng ánh xạ:
     - `0`: **mandatory**
     - `1`: **prohibitory**
     - `2`: **warning**
     Trong khi `voc_label.py` lại định nghĩa `classes = ["prohibitory", "mandatory", "danger"]` (tức `0: prohibitory`, `1: mandatory`). Nếu chạy script này sẽ gây đảo ngược nhãn class 0 và class 1!
  3. **Lỗi logic split**: Tại dòng 62-66, code ngẫu nhiên chia `trainval` rồi viết vào `ftest`, phần còn lại viết vào `ftrain` bị nhầm lẫn giữa định nghĩa train và test.
- **Khuyến nghị**: **KHÔNG ĐƯỢC CHẠY `voc_label.py` NGUYÊN BẢN**. Các file nhãn YOLO trong `train/labels/` và `test/labels/` đã được tiền xử lý sẵn chuẩn xác 100% theo ánh xạ `(0: mandatory, 1: prohibitory, 2: warning)`.

---

## 5. Thống Kê Chi Tiết Annotation (Pascal VOC XML)

### 5.1 Danh sách Class và số lượng Instance
Toàn bộ 17.856 file XML được parse thành công 100% không gặp lỗi cú pháp (`xml_parse_errors: 0`).
Dataset chỉ có đúng **3 class duy nhất**, không có tên dị biệt hay lỗi chính tả:

| Tên Class trong XML | Tên Tiếng Việt | ID trong YOLO txt | Số instance Train | Số instance Test | Tổng Instance | Tỉ lệ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`prohibitory`** | Biển báo Cấm (Viền đỏ, tròn) | 1 | 13.876 | 2.177 | **16.053** | 53.39% |
| **`mandatory`** | Biển Hiệu lệnh / Bắt buộc (Xanh, tròn/chữ nhật) | 0 | 8.364 | 718 | **9.082** | 30.21% |
| **`warning`** | Biển Cảnh báo / Nguy hiểm (Vàng, tam giác) | 2 | 4.598 | 333 | **4.931** | 16.40% |
| **Tổng cộng** | | | **26.838** | **3.228** | **30.066** | **100.0%** |

### 5.2 Thống kê số lượng đối tượng trên mỗi ảnh
- Số ảnh có chú thích: **17.856 ảnh** (100% ảnh train & test đều có ít nhất 1 biển báo).
- Số ảnh không có chú thích: **0 ảnh** (trong train/test; chỉ có riêng 500 ảnh negative là không có nhãn).
- **Ảnh chứa nhiều đối tượng (Multi-object images)**: **7.076 ảnh (chiếm 39.63%)**.

| Số lượng biển báo / ảnh | Số lượng ảnh | Tỉ lệ (%) | Chi tiết Train | Chi tiết Test |
| :---: | :---: | :---: | :---: | :---: |
| 1 biển báo | 10.780 | 60.37% | 10.068 | 712 |
| 2 biển báo | 4.191 | 23.47% | 3.834 | 357 |
| 3 biển báo | 1.605 | 8.99% | 1.426 | 179 |
| 4 biển báo | 723 | 4.05% | 616 | 107 |
| 5 biển báo | 339 | 1.90% | 242 | 97 |
| 6 biển báo | 131 | 0.73% | 114 | 17 |
| 7 biển báo | 39 | 0.22% | 26 | 13 |
| 8 biển báo | 16 | 0.09% | 13 | 3 |
| 9 biển báo | 18 | 0.10% | 3 | 15 |
| 10 biển báo | 1 | 0.006% | 1 | 0 |
| 11 biển báo | 13 | 0.07% | 13 | 0 |

### 5.3 Thống kê kích thước Bounding Box (Width, Height, Area)
Tổng số bounding box hợp lệ được thống kê: **30.066 boxes**.

| Chỉ số | Chiều rộng (Width - px) | Chiều cao (Height - px) | Diện tích (Area - px²) |
| :--- | :---: | :---: | :---: |
| **Min** | 4.0 | 5.0 | 28.0 |
| **Mean (Trung bình)** | 30.28 | 32.21 | 1.452.35 |
| **Std (Độ lệch chuẩn)** | 22.15 | 24.74 | 3.735.56 |
| **Median (Trung vị)** | **24.0** | **25.0** | **624.0** |
| **Max** | 380.0 | 491.4 | 143.640.0 |
| **Phân vị 1% (P1)** | 9.0 | 10.0 | 90.0 |
| **Phân vị 5% (P5)** | 11.0 | 13.0 | 144.0 |
| **Phân vị 25% (P25)** | 18.0 | 19.0 | 342.0 |
| **Phân vị 50% (P50)** | 24.0 | 25.0 | 624.0 |
| **Phân vị 75% (P75)** | 35.0 | 36.0 | 1.258.0 |
| **Phân vị 90% (P90)** | 51.0 | 55.0 | 2.728.0 |
| **Phân vị 95% (P95)** | 69.0 | 75.0 | 4.958.0 |
| **Phân vị 99% (P99)** | 120.0 | 141.24 | 15.688.5 |

### 5.4 Đặc điểm vật thể nhỏ (Small Objects Analysis)
- **Tỉ lệ bbox diện tích < 1024 px² (chuẩn MS COCO Small Object, tương đương 32x32 px)**: **20.683 / 30.066 boxes, chiếm 68.79%**.
- **Tỉ lệ bbox có width < 32px hoặc height < 32px**: **21.576 / 30.066 boxes, chiếm 71.76%**.
- **Tỉ lệ bbox cực nhỏ (Tiny Objects: width hoặc height < 16px)**: **5.039 / 30.066 boxes, chiếm 16.76%**.
- *Nhận xét chuyên môn*: Đây là một thách thức kỹ thuật rất lớn cho mô hình Object Detection. Đa phần các biển báo xuất hiện ở hậu cảnh xa của camera hành trình, đòi hỏi kiến trúc mạng phải có feature pyramid nhạy với vật thể nhỏ (P2/P3 high-resolution layers).

### 5.5 Bất thường tọa độ (Anomalies & Coordinate Issues)
- Tọa độ âm (`xmin < 0` hoặc `ymin < 0`): **0**.
- Tọa độ nghịch đảo (`xmax <= xmin` hoặc `ymax <= ymin`): **0**.
- Diện tích bằng 0 (`area <= 0`): **0**.
- Tọa độ vượt ngoài kích thước ảnh (`out_of_bounds`): **Đúng 5 bounding box**. Cả 5 trường hợp này đều nằm trên ảnh có kích thước 1000x350, trong đó tọa độ `xmax = 1001.0` (vượt đúng 1 pixel mép phải ảnh). Cụ thể:
  1. `04492.xml`: class `warning`, bbox `[951.0, 134.0, 1001.0, 177.0]` (ảnh `1000x350`)
  2. `08144.xml`: class `prohibitory`, bbox `[951.0, 12.0, 1001.0, 66.0]` (ảnh `1000x350`)
  3. `11923.xml`: class `warning`, bbox `[922.0, 145.0, 1001.0, 196.0]` (ảnh `1000x350`)
  4. `11953.xml`: class `prohibitory`, bbox `[946.0, 169.0, 1001.0, 217.0]` (ảnh `1000x350`)
  5. `12655.xml`: class `prohibitory`, bbox `[975.0, 79.0, 1001.0, 102.0]` (ảnh `1000x350`)
  *(Cách xử lý đơn giản: clip giá trị về khoảng `[0, width]` khi tiền xử lý)*.

---

## 6. Khảo Sát Tệp Ảnh (Image Integrity & Leakage Check)

### 6.1 Thống kê độ phân giải và định dạng
- Tổng số ảnh trong toàn bộ dataset: **18.356 ảnh**.
- Ảnh hỏng / không đọc được bằng PIL: **0 ảnh** (tất cả 18.356 ảnh đều đọc và verify thành công).
- **Phân bố độ phân giải**:
  - `1000x350`: **10.291 ảnh** (chiếm 56.06% - ảnh cắt ngang tỷ lệ siêu rộng của camera hành trình)
  - `1280x720`: **4.754 ảnh** (chiếm 25.90% - chuẩn HD)
  - `848x480`: **1.163 ảnh** (chiếm 6.34%)
  - `1024x768`: **884 ảnh** (chiếm 4.82%)
  - `641x936`: **366 ảnh** (chiếm 1.99%)
  - `1920x1080`: **322 ảnh** (chiếm 1.75% - Full HD)
  - `513x999`: **295 ảnh** (chiếm 1.61%)
  - `912x684`: **114 ảnh** (chiếm 0.62%)
  - `1920x1076`: **71 ảnh** (chiếm 0.39%)
  - `1080x606`: **58 ảnh** (chiếm 0.32%)
  - `457x889`: **38 ảnh** (chiếm 0.21%)

### 6.2 Kiểm tra rò rỉ dữ liệu (Data Leakage) & Trùng lặp
- **Data Leakage giữa Train và Test**:
  - Giao tập tên file giữa Train và Test: **0 file** (`train` đánh số 00000 - 18991; `test` đánh số 18992 - 20491).
  - Trùng lặp mã hash MD5 giữa ảnh Train và Test: **0 file**.
  - **Kết luận**: **Hoàn toàn không có data leakage giữa train và test**.
- **Trùng lặp nội bộ (Internal Duplicates)**:
  - Trong tập **Train**: Phát hiện **9 cặp ảnh trùng MD5 tuyệt đối** (18 ảnh):
    - `00383.jpg` == `00398.jpg`
    - `03183.jpg` == `03198.jpg`
    - `17797.jpg` == `17807.jpg`
    - `17830.jpg` == `17838.jpg`
    - `17896.jpg` == `17907.jpg`
    - `17934.jpg` == `17944.jpg`
    - `17970.jpg` == `17981.jpg`
    - `18239.jpg` == `18249.jpg`
    - `18275.jpg` == `18286.jpg`
    - *Phát hiện nhãn bất nhất giữa ảnh trùng*: Ví dụ ở cặp `00383.jpg` và `00398.jpg`, file `00383` được gán cả 2 biển (`prohibitory` và `warning`), nhưng file `00398` lại bị gán thiếu biển `prohibitory` (chỉ gán `warning`).
  - Trong tập **Test**: Phát hiện **4 cặp ảnh trùng MD5 tuyệt đối** (8 ảnh):
    - `20112.jpg` == `20118.jpg`
    - `20146.jpg` == `20150.jpg`
    - `20177.jpg` == `20183.jpg`
    - `20209.jpg` == `20217.jpg`

---

## 7. Phân Tích Tập Train / Test & Các Tập Test Phụ

### 7.1 Tỉ lệ phân chia Train / Test
- Tập Train: **16.356 ảnh (91.6%)**, **26.838 bounding boxes**.
- Tập Test: **1.500 ảnh (8.4%)**, **3.228 bounding boxes**.
- Cấu trúc "2 directories": Bên trong cả `train/` và `test/` gồm thư mục `images/` (chứa ảnh .jpg) và `labels/` (chứa nhãn .txt tương ứng theo chuẩn YOLO).

### 7.2 Đánh giá tập test theo thời tiết & ánh sáng (`测试集天气光照情况分类`)
Toàn bộ 1.500 ảnh của tập test được phân bổ đầy đủ vào 6 điều kiện thực tế:

| Tên thư mục | Dịch nghĩa điều kiện | Số ảnh | Prohibitory | Mandatory | Warning | Đặc trưng môi trường |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `night` | Ban đêm / Thiếu sáng | 500 | 925 | 240 | 114 | Độ tương phản cực thấp, lóa đèn pha |
| `sunny` | Trời nắng ban ngày | 400 | 343 | 136 | 100 | Ánh sáng chuẩn, đổ bóng mạnh |
| `cloud` | Trời nhiều mây / âm u | 300 | 354 | 198 | 103 | Ánh sáng khuếch tán, độ tương phản vừa |
| `rain` | Trời mưa | 160 | 358 | 128 | 2 | Nhòe hạt nước trên kính, phản xạ đường |
| `snow` | Trời tuyết rơi | 100 | 152 | 0 | 14 | Nền trắng tuyết phủ, che khuất một phần |
| `foggy` | Sương mù dày | 40 | 45 | 16 | 0 | Giảm tầm nhìn xa, mờ nhạt viền biển báo |
| **Tổng** | | **1.500** | **2.177** | **718** | **333** | |

### 7.3 Đánh giá tập test theo kích thước (`测试集尺寸大小分类`)
Dùng để kiểm thử năng lực nhận diện vật thể theo các thang đo kích thước:

| Nhóm | Phân loại | Số lượng ảnh | Số lượng bbox | Diện tích (Min - Max) | Kích thước lớn nhất (Min - Max) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **XS** | Extra Small (Cực nhỏ) | 231 | 453 | 28 - 210 px² | 6 - 19 px |
| **S** | Small (Nhỏ) | 239 | 395 | 216 - 399 px² | 15 - 25 px |
| **M** | Medium (Trung bình) | 362 | 592 | 400 - 992 px² | 20 - 42 px |
| **L** | Large (Lớn) | 148 | 209 | 1.020 - 1.980 px² | 32 - 61 px |
| **XL** | Extra Large (Rất lớn) | 124 | 161 | 2.009 - 16.104 px² | 45 - 132 px |
| **Uncategorized** | Ảnh đa kích thước hỗn hợp | 396 | 1.418 | Hỗn hợp nhiều biển báo kích thước khác nhau |
| **Tổng** | | **1.500** | **3.228** | |

---

## 8. Khảo Sát Tập Mẫu Âm Tính (`负样本图像`) & Cạm Bẫy Tiềm Ẩn

Thư mục `负样本图像` chứa đúng **500 ảnh**. Qua khảo sát sâu bằng mã nguồn:
1. **Hoàn toàn không có annotation**: Không có file XML hay TXT nào thuộc về 500 ảnh này.
2. **Bản chất định dạng**: Mặc dù mang phần mở rộng `.jpg`, **100% (500/500 ảnh) thực chất là định dạng PNG**.
3. **CẠM BẪY TRÙNG TÊN FILE**:
   - Tên file trong thư mục này được đặt từ `01501.jpg` đến `02000.jpg`.
   - Trong khi đó, tập `train/images/` cũng có các file cùng tên từ `01501.jpg` đến `02000.jpg`!
   - Tuy nhiên, 2 file này hoàn toàn khác nhau về nội dung (ví dụ `train/images/01501.jpg` là JPEG kích thước 1024x768 chứa 4 biển cấm; còn `负样本图像/01501.jpg` là PNG kích thước 1280x720 không chứa biển báo nào).
   - **Cảnh báo cực kỳ quan trọng**: Nếu người làm vô tình copy toàn bộ `负样本图像` vào `train/images/`, hệ thống sẽ ghi đè (overwrite) và phá hủy 500 ảnh dương tính kèm nhãn trong tập huấn luyện!
4. **Mục đích sử dụng**: Khi huấn luyện YOLO/Faster R-CNN, các mô hình thường hưởng lợi từ 500 ảnh background này để học cách phân biệt biển báo với các vật thể hình tròn/tam giác trên đường phố (đèn giao thông, biển số nhà, đèn hậu ô tô), giúp giảm tỉ lệ False Positive. Khi sử dụng, cần đổi tên (ví dụ: `neg_01501.jpg`) và tạo file nhãn rỗng `.txt`.

---

## 9. Đánh Giá Mất Cân Bằng Dữ Liệu (Class Imbalance)

- Tỉ lệ phân bố toàn bộ dataset:
  - `prohibitory`: 53.39%
  - `mandatory`: 30.21%
  - `warning`: 16.40%
  - **Hệ số mất cân bằng (Max/Min)**: $16.053 / 4.931 \approx \mathbf{3.255 : 1}$.
- Tỉ lệ trong tập Train:
  - `prohibitory`: 51.70% (13.876)
  - `mandatory`: 31.17% (8.364)
  - `warning`: 17.13% (4.598)
  - Tỉ lệ Train: $\approx 3.02 : 1.82 : 1$.
- Tỉ lệ trong tập Test:
  - `prohibitory`: 67.44% (2.177)
  - `mandatory`: 22.24% (718)
  - `warning`: 10.32% (333)
  - Tỉ lệ Test: $\approx \mathbf{6.54 : 2.16 : 1}$.
- **Nhận định**: Lớp `warning` bị thiếu hụt đáng kể so với `prohibitory`, đặc biệt trong tập kiểm thử test set (tỉ lệ gấp 6.54 lần). Cần chú ý theo dõi metric mAP riêng cho từng lớp (đặc biệt là AP của `warning`) thay vì chỉ nhìn vào mAP trung bình.

---

## 10. Môi Trường Phần Cứng & Phần Mềm Hiện Tại

- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU.
  - VRAM: **8.188 MiB (~8 GB)**.
  - Driver Version: **591.66**, Hỗ trợ CUDA tối đa: **13.1**.
  - Tình trạng: Trống bộ nhớ (Memory-Usage ~1.6 GB của hệ thống hiển thị WDDM).
- **CPU & RAM**:
  - Tổng RAM: **23.69 GB**.
  - RAM khả dụng: **~7.5 GB**.
- **Ổ đĩa**:
  - Ổ đĩa làm việc (`D:`): Tổng dung lượng 292.97 GB, dung lượng trống khả dụng: **177.38 GB** (rất thoải mái để lưu trữ checkpoint, cache, model weights).
- **Phiên bản Python & Thư viện đã cài đặt**:
  - Python: `3.11.9 (64-bit)`.
  - PyTorch: **Chưa cài đặt** (`ModuleNotFoundError: No module named 'torch'`).
  - Ultralytics: Đã cài bản `8.4.78` (tuy nhiên chưa thể import do thiếu `torch` và `opencv-python`).
  - Thư viện xử lý dữ liệu sẵn có: `Pillow 12.2.0`, `matplotlib 3.11.0`, `numpy 2.4.6`, `pandas 3.0.1`.

---

## 11. Danh Sách Minh Họa & Biểu Đồ Đã Tạo

### 11.1 Biểu đồ trực quan hóa (Lưu tại `reports/figures/`)
1. `reports/figures/class_distribution.png`: Biểu đồ cột thể hiện phân bố 3 lớp trên toàn bộ dataset và so sánh giữa Train vs Test.
2. `reports/figures/bbox_size_distribution.png`: Biểu đồ phân bố kích thước bbox (Width, Height, và Diện tích ở thang log10).
3. `reports/figures/objects_per_image.png`: Biểu đồ tỉ lệ số lượng biển báo trên mỗi ảnh.
4. `reports/figures/weather_distribution.png`: Biểu đồ phân bổ 1.500 ảnh test theo 6 điều kiện thời tiết & ánh sáng.
5. `reports/figures/size_category_distribution.png`: Biểu đồ số lượng ảnh theo các thang kích cỡ biển báo (XS đến XL).

### 11.2 Ảnh mẫu minh họa có vẽ Bounding Box & Label (Lưu tại `reports/samples/`)
1. `reports/samples/sample_01_sunny_multi_prohibitory_mandatory_19001.jpg`: Test set - Trời nắng, đa đối tượng gồm cả biển cấm và hiệu lệnh.
2. `reports/samples/sample_02_night_prohibitory_small_19300.jpg`: Test set - Ban đêm, biển cấm nhỏ độ tương phản thấp.
3. `reports/samples/sample_03_rain_mandatory_medium_19800.jpg`: Test set - Trời mưa, biển hiệu lệnh kích thước trung bình.
4. `reports/samples/sample_04_snow_prohibitory_warning_19965.jpg`: Test set - Trời tuyết rơi, kết hợp biển cấm và cảnh báo.
5. `reports/samples/sample_05_foggy_prohibitory_20066.jpg`: Test set - Trời sương mù mờ ảo.
6. `reports/samples/sample_06_cloud_mandatory_18992.jpg`: Test set - Trời nhiều mây, biển hiệu lệnh.
7. `reports/samples/sample_07_train_prohibitory_single_00000.jpg`: Train set - Biển cấm đơn lẻ.
8. `reports/samples/sample_08_train_warning_single_00100.jpg`: Train set - Biển cảnh báo đơn lẻ.
9. `reports/samples/sample_09_train_multi_class_prohibitory_warning_00383.jpg`: Train set - Ảnh đa nhãn gồm biển cấm và cảnh báo.
10. `reports/samples/sample_10_train_multi_prohibitory_4objs_01501.jpg`: Train set - Ảnh phức tạp chứa tới 4 biển cấm.
11. `reports/samples/sample_11_test_xs_tiny_sign_19050.jpg`: Test set - Biển cực nhỏ (XS) ở khoảng cách rất xa.
12. `reports/samples/sample_12_test_xl_large_sign_19250.jpg`: Test set - Biển rất lớn (XL) cận cảnh.

---

## 12. Kết Luận Bài Toán & Kế Hoạch Tiền Xử Lý (Pre-processing Checklist)

### 12.1 Kết luận bài toán: Detection hay Classification?
Dataset này **chắc chắn 100% được thiết kế cho bài toán Object Detection (Phát hiện vật thể)** chứ **không phải** Image Classification:
- Ảnh đầu vào là toàn cảnh chụp từ xe ô tô (phần lớn kích thước rộng 1000x350 hoặc 1280x720) với nhiều bối cảnh đường phố phức tạp.
- Gần **40% số ảnh chứa từ 2 đến 11 biển báo** nằm rải rác ở các vị trí khác nhau trong một khung hình.
- Bounding box chiếm diện tích rất nhỏ so với khung hình (trung vị diện tích chỉ 624 px² trên khung hình hàng trăm nghìn pixel).
- *Lưu ý*: Nếu muốn giải bài toán Classification, ta buộc phải crop từng bounding box của từng biển báo ra thành các ảnh nhỏ (patch) riêng biệt. Tuy nhiên với 3 lớp tổng quát (`prohibitory`, `mandatory`, `warning`), bài toán chuẩn mực và có giá trị thực tế nhất cho bộ dữ liệu này chính là **Multi-Class Real-Time Object Detection**.

### 12.2 Danh mục các vấn đề cần tiền xử lý (Pre-processing Checklist)
1. **Sửa 5 bounding box vượt mép ảnh**: Clip tọa độ `xmax` từ 1001 về 1000 cho 5 file ảnh có kích thước 1000x350.
2. **Không dùng script `voc_label.py` gốc**: Dùng trực tiếp các nhãn YOLO `.txt` có sẵn ở `train/labels/` và `test/labels/` (đã xác minh khớp 100% với XML theo thứ tự `0: mandatory, 1: prohibitory, 2: warning`).
3. **Xử lý tập Negative Samples (`负样本图像`)**:
   - Đổi tên file (ví dụ thêm tiền tố `neg_01501.jpg` đến `neg_02000.jpg`) để tránh trùng đè tên file với `train/images/`.
   - Chuyển đổi định dạng từ PNG thực tế sang JPEG hoặc để nguyên nhưng cập nhật đúng định dạng.
   - Tạo các file nhãn `.txt` rỗng tương ứng (không có dòng bbox nào) nếu muốn đưa vào tập huấn luyện của YOLO để giảm False Positives.
4. **Xử lý 9 cặp ảnh trùng lặp trong Train**: Loại bỏ ảnh trùng bị gán thiếu nhãn (như `00398.jpg`) hoặc hợp nhất nhãn trước khi train để tránh gây nhiễu gradient loss.
5. **Chiến lược cho vật thể nhỏ (<32px)**:
   - Với 71.8% vật thể nhỏ, khi resize ảnh cần chọn độ phân giải đầu vào thích hợp (ví dụ `imgsz=640` hoặc `1024` hoặc `1280` tùy năng lực GPU/VRAM).
   - Tận dụng data augmentation phù hợp (Mosaic, Copy-Paste, RandomAffine) và các tầng feature map độ phân giải cao (P2 layer trong YOLO).
6. **Cài đặt môi trường bổ sung (khi được duyệt bước tiếp theo)**:
   - Cài đặt PyTorch tương thích CUDA (`torch`, `torchvision`).
   - Cài đặt OpenCV (`opencv-python`) để hoàn thiện môi trường cho `ultralytics`.

