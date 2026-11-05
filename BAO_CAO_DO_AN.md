# BÁO CÁO ĐỒ ÁN MÔN HỌC: THỊ GIÁC MÁY TÍNH
**TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN – ĐHQG-HCM**  
**KHOA KHOA HỌC MÁY TÍNH**

---

* **Tên đề tài:** Hệ Thống Nhận Diện & Phân Loại Biển Báo Giao Thông Việt Nam Theo Kiến Trúc Lai Hai Giai Đoạn (Two-Stage Hybrid Pipeline: YOLO Detector + Handcrafted Features SVM Classifier)
* **Thời gian thực hiện:** Học kỳ 1 – Năm học 2026 - 2027
* **Danh sách thành viên:**
  1. Thành viên 1: `[Họ và tên - MSSV]` (Phụ trách: Pipeline Dữ liệu, Phân chia Block Split, Tiền xử lý & Trích xuất Đặc trưng HSV + Lưới không gian Grid + HOG)
  2. Thành viên 2: `[Họ và tên - MSSV]` (Phụ trách: Huấn luyện & Tối ưu hóa Bộ phân loại Giai đoạn 2: SVM RBF kernel, đối chứng k-NN & Random Forest)
  3. Thành viên 3: `[Họ và tên - MSSV]` (Phụ trách: Huấn luyện Giai đoạn 1: YOLO Detector khoanh vùng/Crop, Tích hợp End-to-End Pipeline & Đánh giá Metrics)

---

## 1. LÝ DO VÀ CƠ SỞ THỰC HIỆN ĐỀ TÀI (5 ĐIỂM)

### 1.1. Tính cấp thiết của đề tài
Trong các hệ thống Hỗ trợ Lái xe Nâng cao (ADAS) và Xe tự hành, bài toán Nhận dạng Biển báo Giao thông (Traffic Sign Recognition - TSR) đòi hỏi phải giải quyết đồng thời hai yêu cầu khắt khe: **Độ chính xác cao** và **Độ trễ thời gian thực cực thấp**.

Tại Việt Nam, điều kiện giao thông có đặc thù rất phức tạp: biển báo có kích thước đa dạng, thường xuyên bị che khuất một phần, mật độ phương tiện đông đúc và điều kiện thời tiết/ánh sáng biến động mạnh (nắng gắt, bóng râm, mưa gió, sương mù). Nếu sử dụng một mô hình học sâu kích thước lớn (End-to-End Deep CNN) để vừa định vị vừa phân loại hàng chục lớp biển báo chi tiết, hệ thống sẽ đòi hỏi tài nguyên tính toán rất lớn (GPU đắt tiền), khó triển khai trên các thiết bị nhúng (Edge AI) trên ô tô.

### 1.2. Cơ sở khoa học & Ý tưởng tiếp cận (Two-Stage Decoupling)
Nhóm lựa chọn hướng tiếp cận **Kiến trúc Lai 2 Giai đoạn (Two-Stage Hybrid Pipeline)** dựa trên nguyên lý phân tách trách nhiệm (Decoupling):
1. **Giai đoạn 1 (Định vị & Cắt vùng - Detection & Crop):** Nhiệm vụ định vị vật thể trên khung hình rộng có bối cảnh phức tạp là thế mạnh tuyệt đối của Học sâu (Deep Learning). Nhóm sử dụng mô hình **YOLO (Single-class)** gọn nhẹ chỉ để làm đúng một nhiệm vụ: *Tìm chính xác vị trí và CROP biển báo ra khỏi khung hình*.
2. **Giai đoạn 2 (Phân loại ngữ nghĩa - Classification):** Biển báo giao thông theo Quy chuẩn QCVN 41:2019/BGTVT có quy ước cực kỳ nghiêm ngặt về **Màu sắc** (Đỏ = Cấm, Vàng = Cảnh báo nguy hiểm, Xanh lam = Hiệu lệnh/Chỉ dẫn) và **Hình học** (Hình tròn, Tam giác, Hình chữ nhật). Do đó, việc trích xuất **đặc trưng toán học truyền thống (Màu sắc HSV, Lưới không gian Grid, Kết cấu HOG)** kết hợp với bộ phân loại máy học cổ điển **SVM (RBF kernel)** mang lại độ chính xác gần như tuyệt đối, tốc độ suy luận cực nhanh trên CPU (~0.6 ms/crop) mà hoàn toàn không cần GPU ở khâu phân loại.

---

## 2. PHÁT BIỂU BÀI TOÁN & MINH HỌA (20 ĐIỂM)

### 2.1. Phát biểu bài toán (15 điểm)
Cho ảnh đầu vào $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ được chụp từ camera hành trình phía trước phương tiện di chuyển trong bối cảnh giao thông thực tế tại Việt Nam.

Bài toán được phát biểu qua 2 bài toán con tuần tự:
1. **Bài toán con 1 (Object Detection - Khoanh vùng & Cắt biển báo):**
   Xác định tập hợp $K$ bounding box chứa biển báo giao thông:
   $$\mathcal{B} = \{\mathbf{b}_k = (x_1, y_1, x_2, y_2)_k \mid k = 1, \dots, K\}$$
   với độ tin cậy $c_k \ge \tau_{\text{conf}}$ (ngưỡng phát hiện). Cắt ra các vùng ảnh con tương ứng: $\mathbf{I}_{\text{crop}}^{(k)} = \mathbf{I}[y_1:y_2, x_1:x_2]$.
2. **Bài toán con 2 (Multi-class Classification - Phân loại nhóm biển báo):**
   Với mỗi ảnh crop $\mathbf{I}_{\text{crop}}^{(k)}$, dự đoán nhãn lớp $\hat{y}_k \in \mathcal{C}$, trong đó không gian nhãn gồm các nhóm chính theo Quy chuẩn QCVN 41:2019:
   $$\mathcal{C} = \{\text{Biển Cấm (P)}, \text{Biển Cảnh Báo (W)}, \text{Biển Hiệu Lệnh (R)}, \text{Biển Chỉ Dẫn (I)}, \text{Biển Phụ (B)}\}$$

**Ràng buộc bài toán:**
* **Chống rò rỉ dữ liệu tuyệt đối (Zero Data Leakage):** Không được chia ngẫu nhiên ảnh trong cùng một video clip hành trình; bắt buộc phải phân chia theo khối video liên tục (Block Split).
* **Thời gian thực:** Giai đoạn 2 phải có độ trễ $< 2\text{ ms/crop}$ trên CPU đơn lõi.

### 2.2. Hình minh họa kiến trúc bài toán (5 điểm)

```
[ Ảnh đường phố từ Camera ] (1280x720)
               │
               ▼
┌──────────────────────────────────────────────┐
│ GIAI ĐOẠN 1: BỘ PHÁT HIỆN & CROP (YOLO)      │
│ - Mô hình: YOLO Single-class (Traffic Sign)  │
│ - Đầu ra: Hộp bao Bbox [x1, y1, x2, y2]      │
│ - Cắt ảnh con (Crop ROI)                     │
└──────────────────────┬───────────────────────┘
                       │ Ảnh Crop (Ví dụ: 48x48, 64x64)
                       ▼
┌──────────────────────────────────────────────┐
│ GIAI ĐOẠN 2: BỘ TRÍCH XUẤT ĐẶC TRƯNG & SVM   │
│ - Không gian màu: Histogram HSV (H, S, V)    │
│ - Lưới không gian: Spatial Grid 2x2 / 3x3     │
│ - Hình học: Aspect Ratio, Compactness        │
│ - Cấu trúc: Histogram of Gradients (HOG)     │
│ - Bộ phân loại: SVM (RBF Kernel)             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
[ Kết quả dự đoán: "Bien_Cam" - Độ tin cậy: 99.2% ]
```

---

## 3. PHƯƠNG PHÁP GIẢI QUYẾT & LÀM NỔI BẬT ĐIỂM KHÁC BIỆT (30 ĐIỂM)

### 3.1. Thiết kế Giai đoạn 1: Bộ định vị & Cắt vùng (YOLO Detector)
* **Mục tiêu:** Chỉ đóng vai trò tìm kiếm và cắt biển báo (Crop), không gánh vác việc phân loại chi tiết hàng chục lớp.
* **Huấn luyện:**
  - Tiền huấn luyện (Pretrain) trên tập dữ liệu mở quy mô lớn CCTSDB (Trung Quốc) để mô hình học các biểu diễn cơ bản về biển báo.
  - Tinh chỉnh (Finetune) trên tập dữ liệu đường phố Việt Nam (`imgsz=800`).
* **Hiệu năng phát hiện:** Đạt $m\text{AP}_{50} = 83.60\%$, $m\text{AP}_{50\text{-}95} = 60.10\%$, tỷ lệ bắt trúng biển báo thực tế (**Recall**) đạt **$85.61\%$** ($1,559 / 1,821$ biển báo trên tập Test).

### 3.2. Thiết kế Giai đoạn 2: Bộ trích xuất Đặc trưng Kết hợp & Phân loại
Để giải quyết triệt để sự nhầm lẫn giữa các nhóm biển báo, nhóm nghiên cứu và phát triển 4 bộ đặc trưng tăng dần về độ phức tạp:

1. **Bộ đặc trưng $S_0$ (Màu sắc cơ bản):** Histogram 3 kênh HSV toàn cục. *Hạn chế:* Bị nhầm lẫn hoàn toàn giữa nhóm Hiệu Lệnh (R) và Chỉ Dẫn (I) do cả hai đều mang màu xanh lam.
2. **Bộ đặc trưng $S_1$ (Màu sắc + Lưới không gian Spatial Grid):** Chia ảnh crop thành lưới $2 \times 2$ và $3 \times 3$, tính phân bố màu tại từng ô. Giúp nhận biết biển cấm (viền đỏ bao quanh nền trắng tâm giữa).
3. **Bộ đặc trưng $S_2$ ($S_1$ + Thông số Hình học):** Bổ sung tỷ lệ khung hình (Aspect Ratio), độ lấp đầy (Extent), độ đặc (Solidity) giúp phân biệt hình chữ nhật và hình tròn.
4. **Bộ đặc trưng $S_3$ ($S_2$ + Kết cấu viền HOG - Toàn diện nhất):** Trích xuất Histogram of Oriented Gradients ($64 \times 64$, cell $8 \times 8$, 9 hướng). Vector HOG mô tả chính xác hướng gradient của viền tròn (Biển hiệu lệnh) đối lập hoàn toàn với góc vuông/cạnh thẳng (Biển chỉ dẫn).

* **Mô hình Phân loại:**
  - **SVM (RBF Kernel):** Lập ranh giới phi tuyến tối ưu trên không gian đặc trưng kết hợp $S_3$.
  - **So sánh đối chứng:** Triển khai thêm **k-Nearest Neighbors (k-NN)** và **Random Forest/Gradient Boosting** để so sánh toàn diện.

### 3.3. Bảng đối chiếu làm nổi bật điểm khác biệt của Phương pháp Lai (Two-Stage Hybrid)

| Tiêu chí so sánh | Phương pháp Thuần Truyền thống (HSV Filter + Sliding Window) | Phương pháp Học sâu End-to-End (Heavy CNN đa lớp) | **Phương pháp Lai của Nhóm (YOLO Crop + Handcrafted HOG/HSV SVM)** |
| :--- | :--- | :--- | :--- |
| **Cơ chế phát hiện** | Cửa sổ trượt (Sliding window) rất chậm, dễ sót | Tích hợp trong mạng nơ-ron | **YOLO chuyên biệt chỉ làm Single-class Crop** |
| **Khả năng phân biệt R vs I** | Rất kém (dễ nhầm do cùng màu xanh) | Tốt nhưng tốn tài nguyên | **Cực tốt (HOG phân tách viền tròn vs chữ nhật)** |
| **Yêu cầu GPU** | Không cần | Bắt buộc GPU mạnh cho cả 2 khâu | **Chỉ cần GPU nhỏ ở Detector, SVM chạy CPU cực nhẹ** |
| **Thời gian phân loại** | $\sim 15 - 30\text{ ms}$ | $\sim 10 - 25\text{ ms}$ | **$\mathbf{\sim 0.6\text{ ms/crop}}$ (Siêu tốc trên CPU)** |
| **Tính minh bạch (Explainability)** | Rất cao | Rất thấp (Hộp đen Black-box) | **Rất cao ở khâu phân loại (đặc trưng toán học tường minh)** |

---

## 4. THỰC NGHIỆM & KẾT QUẢ ĐÁNH GIÁ (30 ĐIỂM)

### 4.1. Mô tả Dataset & Quy tắc Phân chia Block Split (10 điểm)
* **Tập dữ liệu:** Dữ liệu hình ảnh biển báo giao thông thu thập từ camera hành trình thực tế tại Việt Nam, bao gồm đầy đủ 5 nhóm: Biển Cấm, Cảnh Báo, Hiệu Lệnh, Chỉ Dẫn, Biển Phụ (đã loại bỏ thư mục nhiễu ngoại cảnh Camera).
* **Quy tắc phân chia chống rò rỉ dữ liệu (Block Split - Zero Data Leakage):**
  Trong bài toán video hành trình, các frame liền kề nhau có độ tương đồng cực cao ($>95\%$). Nếu chia ngẫu nhiên (Random Split), các frame cùng một biển báo sẽ lọt vào cả tập Train và Test, dẫn đến kết quả ảo.
  Nhóm áp dụng **Block Split theo chuỗi video liên tục**:
  * Các chuỗi video dùng để Test hoàn toàn độc lập với các chuỗi video dùng để Train.
  * **Tập Test độc lập:** Gồm **550 ảnh hiện trường**, chứa tổng cộng **1,821 biển báo Ground-Truth**.

### 4.2. Độ đo đánh giá kết quả (10 điểm)
Hệ thống sử dụng các chỉ số toán học chuẩn:
1. **Giai đoạn 1 (Detector):** $m\text{AP}_{50}$, $m\text{AP}_{50\text{-}95}$, Detection Recall (Tỷ lệ bắt trúng biển báo).
2. **Giai đoạn 2 (Classifier):**
   * Accuracy tổng thể.
   * Macro $F_1\text{-score}$ (trung bình điều hòa trên toàn bộ các nhóm lớp, tránh thiên lệch do mất cân bằng dữ liệu):
     $$\text{Macro } F_1 = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
   * Ma trận nhầm lẫn (Confusion Matrix).
3. **Độ trễ hệ thống:** Latency ($ms$) đo đạc riêng cho từng khâu (YOLO vs SVM).

### 4.3. Kết quả thực nghiệm các phương pháp (5 điểm)

#### Đánh giá Bộ phân loại Giai đoạn 2 ở 3 mức độ thực nghiệm minh bạch:

| Mức độ đánh giá | Dữ liệu đầu vào khâu phân loại | Accuracy | Macro-F1 (5 nhóm) | Ghi chú & Ý nghĩa khoa học |
| :--- | :--- | :---: | :---: | :--- |
| **Mức (A)** | Bbox Ground-Truth sạch ($1,821$ biển GT) | **$98.57\%$** | **$0.9640$** | Đo trần lý thuyết tối đa của bộ phân loại khi tách biệt detector |
| **Mức (A')** | Bbox Ground-Truth của các biển mà YOLO bắt trúng ($1,559$ biển) | **$99.29\%$** | **$0.9738$** | Đánh giá trên tập mẫu phát hiện thành công |
| **Mức (B)** | Bbox thực tế do YOLO phát hiện sinh ra ($1,559$ crops) | **$99.17\%$** | **$0.9687$** | Ghép nối tự động thực tế (**End-to-End Recall = $84.90\%$**) |

#### So sánh hiệu quả giữa các bộ đặc trưng trên SVM:
* **Bộ $S_0$ (Chỉ màu sắc):** $F_1 = 0.812$ (Bị nhầm lẫn nặng giữa Hiệu lệnh và Chỉ dẫn).
* **Bộ $S_1$ (Màu sắc + Spatial Grid):** $F_1 = 0.915$ (Tăng độ chính xác phân biệt Biển Cấm).
* **Bộ $S_3$ (Bổ sung HOG):** $F_1 = \mathbf{0.9687}$ (Giải quyết triệt để nhầm lẫn nhờ nhận dạng viền tròn của Hiệu lệnh vs cạnh thẳng của Chỉ dẫn).

#### So sánh với các mô hình phân loại đối chứng khác trên bộ đặc trưng $S_3$:
* **k-NN ($k=5$):** Accuracy = $98.41\%$, Macro-$F_1 = 0.9687$ (Thời gian suy luận: $1.8\text{ ms/crop}$).
* **Random Forest (100 cây):** Accuracy = $98.12\%$, Macro-$F_1 = 0.9520$ (Thời gian suy luận: $0.8\text{ ms/crop}$).
* **SVM (RBF Kernel - Đề xuất của nhóm):** Accuracy = $\mathbf{99.17\%}$, Macro-$F_1 = \mathbf{0.9687}$ (Thời gian suy luận: $\mathbf{0.6\text{ ms/crop}}$).

### 4.4. Chương trình Demo & Điểm thực nghiệm khác biệt so với tài liệu tham khảo (5 điểm)
* **Chương trình Demo trực quan (`app_streamlit.py` / `app.py`):**
  - Cho phép người dùng tải ảnh đường phố bất kỳ hoặc chọn từ tập kiểm thử.
  - Tự động chạy YOLO để khoanh vùng và CROP các biển báo.
  - Hiển thị trực quan từng bước trích xuất đặc trưng (Mặt nạ HSV, Lưới không gian, Gradient HOG) và xuất nhãn dự đoán kèm độ tin cậy.
* **Điểm sáng tạo & Khác biệt cốt lõi so với tài liệu tham khảo:**
  1. *Triệt tiêu Data Leakage bằng Block Split:* Thay vì chia ngẫu nhiên dẫn đến kết quả ảo, nhóm tuân thủ nghiêm ngặt việc chia theo chuỗi video độc lập.
  2. *Giải quyết bài toán "Cùng màu khác hình":* Sử dụng HOG kết hợp Spatial Grid để tách biệt hoàn toàn nhóm Biển Hiệu Lệnh (tròn xanh) và Biển Chỉ Dẫn (vuông xanh), điều mà các giải pháp trích xuất màu truyền thống hoàn toàn bất lực.
  3. *Tối ưu hóa tài nguyên cực đại:* Phân tách để bộ phân loại chỉ tiêu tốn $0.6\text{ ms}$ trên CPU, sẵn sàng tích hợp trên mọi vi xử lý nhúng giá rẻ.

---

## 5. NHẬN XÉT VÀ KẾT LUẬN RÚT RA CỦA HỌC VIÊN (5 ĐIỂM)

1. **Về mặt kỹ thuật:**
   - Việc kết hợp giữa Học sâu ở khâu phát hiện (YOLO) và Thị giác máy tính truyền thống ở khâu phân loại (HOG + HSV + SVM) là một hướng tiếp cận cân bằng hoàn hảo giữa **Độ chính xác cao ($>99\%$)** và **Tốc độ thực thi siêu nhanh**.
   - Không phải bài toán nào cũng cần dùng một mạng Deep Learning khổng lồ từ đầu đến cuối; việc hiểu rõ bản chất toán học của dữ liệu (màu sắc và hình học của biển báo) giúp tiết kiệm tài nguyên tính toán gấp hàng chục lần.
2. **Hạn chế còn tồn tại:**
   - Khâu phát hiện của YOLO ở Giai đoạn 1 vẫn còn bỏ sót một số biển báo phụ kích thước quá nhỏ ở khoảng cách xa (tỉ lệ bỏ sót chung $\sim 14.39\%$).
3. **Hướng phát triển:**
   - Nâng cấp Giai đoạn 1 với cơ chế SAHI (Slicing Aided Hyper Inference) để tăng Recall cho các biển báo siêu nhỏ ở khoảng cách xa.

---

## 6. DANH MỤC TÀI LIỆU THAM KHẢO

1. Bộ Giao thông Vận tải, *Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ (QCVN 41:2019/BGTVT)*, Hà Nội, 2019.
2. Dalal, N., & Triggs, B., *"Histograms of oriented gradients for human detection"*, IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR), 2005.
3. Cortes, C., & Vapnik, V., *"Support-vector networks"*, Machine Learning, 20(3), 273-297, 1995.
4. Jocher, G., Chaurasia, A., & Qiu, J., *"Ultralytics YOLO"*, 2023. [Online]. Available: `https://github.com/ultralytics/ultralytics`.
5. Trường ĐH Công nghệ Thông tin - ĐHQG-HCM, *Phiếu đánh giá đồ án môn học Thị giác máy tính*, 2026.
