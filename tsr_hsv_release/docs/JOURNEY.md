# BÁO CÁO HÀNH TRÌNH NGHIÊN CỨU & PHÁT TRIỂN HỆ THỐNG TSR (JOURNEY REPORT)

> **Dự án**: CS231 - Nhận diện Biển báo Giao thông Việt Nam (Traffic Sign Recognition - TSR)  
> **Phương pháp cốt lõi**: Pipeline 2 chặng kết hợp Deep Learning Detector (YOLO11s 1-class) và Bộ trích xuất đặc trưng thị giác truyền thống (HSV + Spatial Grid + HOG) phân loại bằng Machine Learning (SVM / Random Forest / Boosting).  
> **Tác giả thực hiện**: Nhóm nghiên cứu TSR  
> **Ngày hoàn thiện**: 04/11/2026  

---

## MỤC LỤC
1. [Bài toán và các ràng buộc kỹ thuật](#1-bài-toán-và-các-ràng-buộc-kỹ-thuật)
2. [Khảo sát dữ liệu: Từ CCTSDB2021 đến Bộ dữ liệu Việt Nam](#2-khảo-sát-dữ-liệu-từ-cctsdb2021-đến-bộ-dữ-liệu-việt-nam)
3. [Quyết định thiết kế kiến trúc: Pipeline 2 chặng](#3-quyết-định-thiết-kế-kiến-trúc-pipeline-2-chặng)
4. [Mô-đun Detector: Pretrain CCTSDB và Finetune Việt Nam](#4-mô-đun-detector-pretrain-cctsdb-và-finetune-việt-nam)
5. [Stage 1: Baseline phân loại màu HSV 44 chiều & Giới hạn đo được](#5-stage-1-baseline-phân-loại-màu-hsv-44-chiều--giới-hạn-đo-được)
6. [Kiểm toán & Chuẩn hóa định danh nhãn (B.8a & Camera)](#6-kiểm-toán--chuẩn-hóa-định-danh-nhãn-b8a--camera)
7. [Stage 2: Tiến hóa chuỗi đặc trưng đa tầng S0 -> S1 -> S2 -> S3](#7-stage-2-tiến-hóa-chuỗi-đặc-trưng-đa-tầng-s0---s1---s2---s3)
8. [Các sai lệch số liệu tự phát hiện & Quá trình khắc phục minh bạch](#8-các-sai-lệch-số-liệu-tự-phát-hiện--quá-trình-khắc-phục-minh-bạch)
9. [Các hạn chế kỹ thuật cốt lõi (Technical Limitations)](#9-các-hạn-chế-kỹ-thuật-cốt-lõi-technical-limitations)
10. [Đề xuất hướng phát triển tiếp theo (Next Steps)](#10-đề-xuất-hướng-phát-triển-tiếp-theo-next-steps)

---

## 1. Bài toán và các ràng buộc kỹ thuật

Hệ thống hướng tới giải quyết bài toán phát hiện và phân loại biển báo giao thông đường bộ Việt Nam theo Quy chuẩn kỹ thuật quốc gia **QCVN 41:2019/BGTVT**. Trong quá trình nghiên cứu, hệ thống chịu các ràng buộc thực tế:
- **Ràng buộc học thuật**: Đề tài phân công thành viên phụ trách **phương pháp thị giác máy tính truyền thống (Traditional Computer Vision)**, khai thác không gian màu (HSV), hình học bounding box, và các bộ mô tả biên dạng kinh điển (HOG) kết hợp các bộ phân loại Machine Learning cổ điển (SVM, kNN, Random Forest, XGBoost, LightGBM), không dùng mạng nơ-ron sâu (Deep Neural Networks) cho pha phân loại.
- **Ràng buộc phần cứng & Bộ nhớ**: Huấn luyện và thực nghiệm trên máy tính cá nhân chạy Windows với GPU đơn (NVIDIA GeForce RTX 4060 Laptop 8GB VRAM) và giới hạn Windows commit limit (~27.7 - 28.8 GB). Ràng buộc này đòi hỏi quy trình huấn luyện phải dùng cơ chế Automatic Mixed Precision (AMP), cấu hình worker luồng phù hợp (`workers=0` trên Windows), chia batch nhỏ (`batch=4`), và chủ động dọn dẹp biến trung gian để tránh lỗi tràn bộ nhớ ảo (`MemoryError` / `bad allocation`).

---

## 2. Khảo sát dữ liệu: Từ CCTSDB2021 đến Bộ dữ liệu Việt Nam

### 2.1 Bộ dữ liệu CCTSDB 2021 (Trung Quốc)
- **Quy mô thực tế**: Gồm **18.356 ảnh** (16.356 train, 1.500 test, 500 ảnh âm tính không chứa biển) với **30.066 bounding box** thuộc 3 nhóm chính: `prohibitory` (cấm), `mandatory` (hiệu lệnh) và `warning` (cảnh báo).
- **Vấn đề phát hiện qua EDA**:
  - **71.8%** biển báo có kích thước siêu nhỏ ($< 32$ pixel).
  - Tệp script `voc_label.py` đi kèm của bộ dữ liệu gốc bị lỗi đảo ngược thứ tự lớp, đòi hỏi phải viết lại bộ chuyển đổi nhãn sang format YOLO.
  - Bộ dữ liệu này được chọn làm tập nguồn để **pretrain** detector học đặc trưng phát hiện vị trí biển báo tổng quát.

### 2.2 Bộ dữ liệu Biển báo Giao thông Việt Nam (52 lớp QCVN 41)
- **Quy mô thực tế**: Gồm **3.216 ảnh** chụp thực tế từ camera hành trình trên đường phố Việt Nam (độ phân giải phổ biến 960x540 và 1920x1080), gắn nhãn **52 lớp chi tiết**.
- **Đặc điểm & Thách thức kỹ thuật**:
  - **Bản chất chuỗi video liên tục**: Toàn bộ 3.216 ảnh là các frame liên tiếp từ xe chạy. Việc chia ngẫu nhiên từng frame sẽ dẫn tới rò rỉ thông tin nghiêm trọng (đã đo đạc: 46.84% cặp frame kề nhau bị xé lẻ giữa train và test nếu chia ngẫu nhiên). Do đó, dự án bắt buộc phải chia theo khối liên tục 100 frame (`block_split`) với khoảng cách đệm `gap=5` frame để đưa số cặp vi phạm về đúng **0.00%**.
  - **Kích thước biển nhỏ**: 61.2% biển báo có chiều cao dưới 32 pixel. Sau khi resize ảnh toàn cảnh về `imgsz=800` để đưa vào YOLO, kích thước biển thu nhỏ xuống chỉ còn $10 - 25$ pixel.
  - **Mất cân bằng cực đoan**: Nhóm Cấm chiếm ưu thế tuyệt đối (>62% dữ liệu), trong khi nhóm Biển phụ (`phụ`) chỉ có 459 mẫu toàn bộ dữ liệu (tập test chỉ có 8 mẫu) và nhóm Chỉ dẫn trên tập Val chỉ có 4 mẫu.

---

## 3. Quyết định thiết kế kiến trúc: Pipeline 2 chặng

Thay vì sử dụng một mô hình Deep Learning duy nhất để vừa tìm vị trí vừa phân loại hàng chục lớp (End-to-End Object Detection), hệ thống lựa chọn thiết kế **Pipeline 2 chặng phân tách module**:
```
[Ảnh toàn cảnh] ---> [Chặng 1: YOLO11s Detector 1-lớp "sign"] ---> [Crop Bounding Box] 
                                                                          │
                                                                          ▼
[5 Nhóm QCVN] <--- [Chặng 2: Classifier HSV+HOG (SVM/Boosting)] <--- [Tiền xử lý & Trích đặc trưng]
```

### Lý do lựa chọn thiết kế này:
1. **Phân định rõ ràng trách nhiệm kỹ thuật**: YOLO11s chỉ tập trung phát hiện sự xuất hiện của biển báo giao thông ("sign") mà không cần phân biệt ngữ nghĩa hay ký hiệu bên trong. Điều này giúp bộ phát hiện đạt độ bao phủ cao (Recall > 91% trên Val, 85.6% trên Test) dù kích thước biển nhỏ.
2. **Tuân thủ đúng bài toán phân loại truyền thống**: Sau khi có bounding box, việc phân loại vào 5 nhóm lớn được giao trọn vẹn cho bộ trích xuất đặc trưng màu sắc HSV, tỉ lệ hình học, lưới không gian và HOG. Bộ phân loại hoạt động cực nhẹ, dễ diễn giải (interpretable), và có thể thay đổi linh hoạt các thuật toán Machine Learning khác nhau (SVM, RF, XGBoost, LightGBM) mà không cần huấn luyện lại mạng nơ-ron nặng nề.

---

## 4. Mô-đun Detector: Pretrain CCTSDB và Finetune Việt Nam

Detector được xây dựng dựa trên kiến trúc tiên tiến **YOLO11s (1-class "sign")**:

| Giai đoạn / Bộ dữ liệu | Tập đánh giá | Precision | Recall | $\text{mAP}_{50}$ | $\text{mAP}_{50-95}$ | Trọng số lưu trữ |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **CCTSDB 2021 (Pretrain)** | CCTSDB Val | **95.15%** | **93.83%** | **97.93%** | **71.88%** | `runs/detect/cctsdb_s/weights/best.pt` |
| **Việt Nam (Finetune - Epoch 14)** | VN Val (Finetuned) | **87.21%** | **91.55%** | **93.30%** | **67.05%** | `runs/detect/vn_s/weights/best.pt` |
| **Việt Nam (Đánh giá cuối kỳ)** | **VN Test (550 ảnh)** | **87.8%** | **85.61%** | **83.60%** | **60.10%** | Test set độc lập (1.821 GT biển sạch) |

### Hiệu quả ghép cặp trên 550 ảnh Test:
- **Số GT bắt được (IoU $\ge 0.5$)**: **1.559 / 1.821 biển** (**85.61%**).
- **Số GT bị bỏ sót (FN detector)**: **262 biển** (**14.39%**).
- **Số box phát hiện dư (FP detector)**: **216 box**.
- Toàn bộ 24 ảnh scene minh họa phân loại theo 4 nhóm tag (`small_*`, `crowded_*`, `single_*`, `rand_*`) đã được lưu trực quan tại `assets/yolo_detect/`.

---

## 5. Stage 1: Baseline phân loại màu HSV 44 chiều & Giới hạn đo được

Trong Stage 1, hệ thống xây dựng mô-đun phân loại cơ sở:
- Crop từ bounding box, resize $32 \times 32$.
- Trích xuất histogram kênh Hue (36 bin, có dịch góc $+30^\circ \pmod{180}$ để gom dải màu đỏ) và kênh Saturation (8 bin), chuẩn hóa $L1$ $\rightarrow$ **Vector 44 chiều**.
- Phân loại bằng **SVM RBF** và **kNN ($k=11$)**.

### Kết quả đo được tại Stage 1:
- Trên Test Mức A (GT crops): SVM đạt Macro-F1 = **0.6877**; kNN đạt **0.7008**.
- Khi ghép với YOLO ở Mức B: End-to-End Recall đạt **81.63%**.

### Ba điểm yếu cốt lõi được phát hiện qua số liệu:
1. **Trùng màu hoàn toàn giữa biển Hiệu lệnh và Chỉ dẫn**: Cả hai nhóm đều mang màu nền xanh lam chủ đạo (Hue $\approx 106^\circ$). Bộ mô tả HSV thuần túy không có thông tin hình dạng để phân biệt hình tròn (hiệu lệnh) với hình chữ nhật (chỉ dẫn).
2. **Nhóm nhân tạo "chưa xác định"**: Việc phân cụm thuần túy theo tiền tố làm rơi 2 lớp `Camera` và `B.8a` vào nhóm "chưa xác định", kéo tụt điểm Macro-F1 của toàn hệ thống.
3. **Mất cân bằng dữ liệu**: Nhóm Cấm chiếm 57.8% tập test, trong khi nhóm Biển phụ chỉ có 8 mẫu.

---

## 6. Kiểm toán & Chuẩn hóa định danh nhãn (B.8a & Camera)

Để khắc phục nhóm rác "chưa xác định", dự án thực hiện đợt kiểm toán nhãn độc lập trên toàn bộ dữ liệu:

1. **Lớp `Camera` (ID 31)**:
   - Toàn bộ dữ liệu chỉ có **40 mẫu** (6 train, 10 val, 24 test).
   - Kiểm tra chuỗi stem cho thấy 40 mẫu này thực chất chỉ là **~3 biển vật lý duy nhất** được camera quay liên tục khi xe chạy ngang qua. Cỡ mẫu quá nhỏ và tính đa dạng quá thấp để huấn luyện một bộ phân loại tổng quát.
   - **Quyết định**: Loại bỏ hoàn toàn khỏi tập train/val/test của classifier. Giữ nguyên trong detector và gán nhãn `ignored` khi đánh giá Mức B để không phạt sai.
2. **Lớp `B.8a` (ID 34)**:
   - Có **179 mẫu** (150 train, trong đó 148 mẫu $\ge 8$px hợp lệ; 0 val; 29 test).
   - Mã hiệu `B.8a` không có định danh chính thức trong văn bản QCVN 41:2019 (chưa kiểm chứng nguồn gốc gán nhãn của dataset gốc). Tuy nhiên, quan sát trực quan hình ảnh thực tế chứng minh biển có hình tròn viền đỏ nền trắng với biểu tượng xe sơ-mi rơ-moóc mang đặc trưng rõ rệt của biển cấm.
   - **Quyết định**: Gộp lớp `B.8a` vào **nhóm Cấm (`cấm (prohibitory)`)**.
3. **Mốc đối chứng S0 sau sửa nhãn**:
   - Ngay sau khi sửa nhãn, cùng trên bộ đặc trưng HSV 44 chiều cũ, Macro-F1 của SVM tăng vọt từ **0.6877** lên **0.8315** (+14.38% F1) và Random Forest đạt **0.8789**.

---

## 7. Stage 2: Tiến hóa chuỗi đặc trưng đa tầng S0 -> S1 -> S2 -> S3

Tại Stage 2, hệ thống nâng cấp đặc trưng qua chuỗi 4 nấc tiến hóa và mở rộng benchmark trên **5 mô hình học máy**:

```
[Crop biển báo] ──► S0: HSV toàn cục (44 dims)
                ──► S1: S0 + Tỉ lệ w/h (45 dims)
                ──► S2: S1 + Lưới màu không gian 2x2 (133 dims)
                ──► S3: S2 + HOG 64x64 cell 16x16 (457 dims)
```

### Bảng tổng hợp GroupKFold 5-fold (toàn bộ 18.496 mẫu) và Fixed Split Test (1.821 mẫu Test Mức A, đã bỏ Camera):

| Nấc đặc trưng | Số chiều | Mô hình | GroupKFold Macro-F1 | Test Accuracy | Test Macro-F1 | Test Major4 F1 |
|---|:---:|---|:---:|:---:|:---:|:---:|
| **S0 (H36+S8)** | 44 | SVM | $0.6521 \pm 0.0570$ | 94.78% | 0.8315 | 0.8449 |
| | | kNN ($k=11$) | $0.6309 \pm 0.0494$ | 95.39% | 0.8328 | 0.8592 |
| | | Random Forest | $0.6820 \pm 0.0656$ | 98.46% | 0.8789 | 0.9597 |
| | | XGBoost | $0.6828 \pm 0.0499$ | 97.53% | 0.8627 | 0.9468 |
| | | LightGBM | $0.6782 \pm 0.0415$ | 98.02% | 0.8798 | 0.9498 |
| **S1 (+w/h)** | 45 | SVM | $0.6608 \pm 0.0644$ | 95.44% | 0.8665 | 0.8643 |
| | | Random Forest | $0.6898 \pm 0.0666$ | 98.30% | 0.8669 | 0.9586 |
| | | XGBoost | $0.6948 \pm 0.0607$ | 98.02% | 0.8872 | 0.9527 |
| | | LightGBM | $0.7022 \pm 0.0497$ | 98.19% | 0.9199 | 0.9499 |
| **S2_2x2 (+grid)** | 133 | SVM | $0.7130 \pm 0.1048$ | 97.47% | 0.9279 | 0.9377 |
| | | Random Forest | $0.7304 \pm 0.0814$ | **99.29%** | 0.9556 | **0.9839** |
| | | XGBoost | $0.7566 \pm 0.0826$ | 98.68% | 0.9476 | 0.9623 |
| | | LightGBM | $0.7573 \pm 0.0875$ | 98.74% | 0.9394 | 0.9638 |
| **S3 (+HOG)** | **457** | **SVM (Best)** | **$\mathbf{0.8983 \pm 0.0746}$** | **98.57%** | **0.9640** | **0.9551** |
| | | kNN ($k=11$) | $0.8455 \pm 0.1113$ | 98.74% | **0.9687** | **0.9609** |
| | | Random Forest | $0.8753 \pm 0.0981$ | 98.68% | 0.9624 | 0.9530 |
| | | XGBoost | $0.8838 \pm 0.0851$ | 98.68% | 0.9421 | 0.9554 |
| | | LightGBM | $0.8885 \pm 0.0882$ | 98.74% | 0.9517 | 0.9544 |

![Biểu đồ chuỗi đặc trưng](../assets/figures/chain.png)

### Kiểm định Paired Difference tại nấc S3:
- So sánh bắt cặp từng fold giữa SVM (mô hình tốt nhất) và các mô hình khác:
  - **LightGBM vs SVM**: Chênh lệch $\mu = +0.0098$, $\sigma = 0.0928$ (do $\mu < \sigma$ nên **chưa phân biệt được** về mặt ý nghĩa thống kê).
  - **XGBoost vs SVM**: Chênh lệch $\mu = +0.0145$, $\sigma = 0.0931$ (**chưa phân biệt được**).
  - **Random Forest vs SVM**: Chênh lệch $\mu = +0.0230$, $\sigma = 0.0845$ (**chưa phân biệt được**).
- **Đo tốc độ suy luận CPU (ms/crop trên 1.000 crop S3)**:
  - **XGBoost**: **0.018 ms/crop** (~55.500 crops/giây) - Nhanh nhất.
  - **LightGBM**: **0.021 ms/crop** (~47.600 crops/giây).
  - **Random Forest**: **0.091 ms/crop** (~11.000 crops/giây).
  - **SVM (RBF)**: **0.612 ms/crop** (~1.630 crops/giây).


> **Lưu ý quan trọng về giá trị 0.9687**: Con số `0.9687` xuất hiện tại 2 vị trí độc lập là sự trùng hợp ngẫu nhiên về mặt toán học giữa 2 ngữ cảnh hoàn toàn khác nhau:
> 1. `0.9687` (chính xác: `0.968727`): Test Macro-F1 của mô hình **kNN ($k=11$)** tại $S3$ khi đánh giá ở **Mức (A)** trên 1.821 crop Ground-Truth sạch.
> 2. `0.9687` (chính xác: `0.968731`): Macro-F1 của mô hình **SVM (RBF)** tại $S3$ khi đánh giá ở **Mức (B)** trên 1.559 crop thực tế do YOLO phát hiện.

### Đánh giá Mức (B) do YOLO phát hiện & End-to-End Recall:
- Khi chuyển sang nhận diện trên các crop thực tế do YOLO phát hiện (Mức B, 1.559 box ghép trúng sau khi bỏ Camera), mô hình SVM $S3$ đạt độ chính xác **99.17%** (1.546 / 1.559), Macro-F1 đạt **0.9687** (chỉ suy giảm nhẹ 0.0051 F1 so với Mức A' = 0.9738 trên Ground Truth).
- **End-to-End Recall toàn hệ thống đạt 84.90%**. Nút thắt chính nằm ở tỷ lệ bỏ sót 14.39% của YOLO detector.

---

## 8. Các sai lệch số liệu tự phát hiện & Quá trình khắc phục minh bạch

Trong quá trình thực nghiệm, hệ thống tuân thủ nguyên tắc trung thực khoa học, chủ động phát hiện và đính chính các sai lệch:
1. **Lỗi thuật toán tính Major4 F1**:
   - *Hiện tượng*: Báo cáo ban đầu ghi nhận S0 Random Forest có Major4 F1 = 0.7713 (thấp hơn nhiều so với Macro-F1 0.8789).
   - *Nguyên nhân*: Lệnh `f1_score` khi lọc bỏ nhãn phụ không truyền tham số `labels`, khiến các ca mô hình đoán nhầm sang "phụ" bị scikit-learn coi là nhãn thứ 5 có $F1=0.0$, làm tổng điểm bị chia cho 5 thay vì chia cho 4.
   - *Khắc phục*: Truyền tường minh `labels=canonical_groups[:4]`. Major4 F1 của S0 RF thực tế đạt **0.9597** (đã kiểm chứng tính tay khớp 100%).
2. **Sai lệch thống kê số lượng mẫu Camera và B.8a**:
   - *Hiện tượng*: Chú thích Bảng 2.2 từng ghi Camera có 74 train và B.8a có 82 train.
   - *Nguyên nhân*: Tác giả cũ lấy mức tăng ròng của nhóm Cấm ($10.135 - 10.053 = +82$) gán nhầm cho B.8a, rồi lấy $156 - 82 = 74$ gán nhầm cho Camera.
   - *Khắc phục*: Đếm trực tiếp từ manifest gốc: Camera có 6 train (40 tổng), B.8a có 150 train (148 hợp lệ sau lọc $<8$px, 179 tổng).
3. **Chuẩn hóa đối chiếu công bằng Mức A vs Mức B**:
   - Thiết lập thêm mốc **Mức ($A'$)** (đánh giá trên chính 1.538 box Ground-Truth mà YOLO tìm thấy) để tách bạch phần suy giảm do crop lệch tâm của detector ($A' \rightarrow B$) với phần suy giảm do bỏ sót đối tượng ($A \rightarrow A'$).

---

## 9. Các hạn chế kỹ thuật cốt lõi (Technical Limitations)

Nhìn nhận trung thực các hạn chế hiện tại của công trình:
1. **Hạn chế cỡ mẫu biển vật lý thực tế**: Dù tập test có 1.821 bounding box, do bản chất video camera hành trình ghi lại cùng một biển báo qua nhiều frame kề nhau, số lượng "biển vật lý độc lập" trong thực tế nhỏ hơn rất nhiều.
2. **Độ bất định thống kê ở nhóm thiểu số**: Nhóm Biển phụ chỉ có vỏn vẹn **8 mẫu test**. Nhóm Chỉ dẫn trên tập Val chỉ có **4 mẫu**. Do đó, chỉ số F1 đo riêng trên các nhóm này có phương sai lớn.
3. **Phạm vi hạt ngẫu nhiên (Random Seed)**: Các kết quả thực nghiệm hiện được cố định ở `seed=42` trên một lần phân chia. Dù GroupKFold 5-fold cho kết quả ổn định, các biến thiên nhỏ có thể xảy ra khi đổi seed.
4. **Đo thời gian suy luận (Latency)**: Các con số $0.018 - 0.612$ ms/crop mới chỉ đo thời gian dự đoán mô hình (Model Inference Time), chưa bao gồm thời gian trích xuất đặc trưng hình ảnh (Resize, HSV, Spatial Grid, HOG) vốn tốn khoảng $1.5 - 2.5$ ms/crop trên CPU.

---

## 10. Đề xuất hướng phát triển tiếp theo (Next Steps)

1. **Khắc phục nút thắt cổ chai ở Detector**: Cải thiện tỷ lệ phát hiện của YOLO đối với các biển báo xa/nhỏ ($< 20$px) bằng cách thử nghiệm kích thước `imgsz=1024` hoặc bổ sung tầng phát hiện P2 (ngắt pooling sớm để giữ chi tiết).
2. **Mở rộng phân loại chi tiết 52 lớp**: Từ 5 nhóm lớn hiện tại, xây dựng bộ phân loại phân cấp (Hierarchical Classifier) hoặc mạng CNN nhỏ để nhận diện chính xác 52 mã biển cụ thể bên trong từng nhóm.
3. **Thực nghiệm đối đầu công bằng**: Chạy chéo mô hình truyền thống (Color+HOG+SVM) trên tập split của đồng nghiệp và ngược lại để so sánh khách quan trên cùng một mặt bằng dữ liệu.