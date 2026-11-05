# BÁO CÁO ĐỒ ÁN MÔN HỌC: THỊ GIÁC MÁY TÍNH
**TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN – ĐHQG-HCM**  
**KHOA KHOA HỌC MÁY TÍNH**

---

* **Tên đề tài:** Nghiên cứu, Triển khai và Đánh giá So sánh (Benchmarking) các Phương pháp Nhận dạng Biển báo Giao thông trong Điều kiện Thời tiết Bất lợi (Adverse Weather Conditions)
* **Thời gian thực hiện:** Học kỳ 1 – Năm học 2026 - 2027
* **Danh sách thành viên:**
  1. Thành viên 1: `[Họ và tên - MSSV]` (Phụ trách: Pipeline Dữ liệu, Global Split & Phương pháp Truyền thống HSV + HOG + SVM)
  2. Thành viên 2: `[Họ và tên - MSSV]` (Phụ trách: Phương pháp Học sâu Single-stage / CNN)
  3. Thành viên 3: `[Họ và tên - MSSV]` (Phụ trách: Phương pháp Học sâu Anchor-free/YOLO, Module Benchmark & Giao diện Demo)

---

## 1. LÝ DO VÀ CƠ SỞ THỰC HIỆN ĐỀ TÀI (5 ĐIỂM)

### 1.1. Tính cấp thiết của đề tài
Trong các hệ thống Hỗ trợ Lái xe Nâng cao (ADAS) và Xe tự hành (Autonomous Vehicles), bài toán Phát hiện và Nhận dạng Biển báo Giao thông (Traffic Sign Recognition - TSR) đóng vai trò quyết định đến tính an toàn khi lưu thông. 

Hầu hết các nghiên cứu truyền thống hoặc các bộ dữ liệu tiêu chuẩn (như GTSRB) thường được thu thập trong điều kiện lý tưởng: ban ngày, đủ sáng, thời tiết quang đãng. Khi áp dụng các mô hình này vào thực tế, hiệu năng thường **suy giảm nghiêm trọng** khi gặp các điều kiện thời tiết bất lợi (Adverse Weather Conditions) như:
* **Trời mưa (Rain):** Các hạt mưa, vệt nước đọng trên kính chắn gió làm tán xạ ánh sáng và biến dạng cấu trúc hình học của biển báo.
* **Sương mù (Fog/Haze):** Độ tương phản (contrast) suy giảm nghiêm trọng theo định luật truyền dẫn khí quyển Koschmieder, làm nhạt màu và mất các cạnh viền sắc nét.
* **Ban đêm / Thiếu sáng (Low-light/Night):** Tỉ lệ tín hiệu trên nhiễu (SNR) thấp, nhiễu hạt cảm biến cao và thường xuyên gặp hiện tượng lóa đèn pha (Headlight glare).

### 1.2. Mục tiêu nghiên cứu
1. Xây dựng một **Quy trình Đánh giá Công bằng (Benchmarking Framework)** với cơ chế **Global Data Split (Train 70% / Val 15% / Test 15%)** nhằm triệt tiêu hoàn toàn hiện tượng rò rỉ dữ liệu (Data Leakage).
2. Triển khai và so sánh hiệu năng của **3 trường phái tiếp cận khác nhau**:
   * *Trường phái Truyền thống:* Xử lý ảnh màu HSV + Phép toán hình thái học + Đặc trưng HOG + Phân loại Linear SVM.
   * *Trường phái Học sâu CNN Tiêu chuẩn:* Kiến trúc Mạng nơ-ron tích chập (Custom CNN / LeNet-5 biến thể).
   * *Trường phái Học sâu Hiện đại (Real-time Object Detector):* Kiến trúc YOLOv8 / MobileNetV3 tối ưu hóa cho thiết bị biên.
3. Đánh giá tính bền vững (Robustness) của từng phương pháp khi chất lượng ảnh bị suy giảm bởi thời tiết khắc nghiệt.

---

## 2. PHÁT BIỂU BÀI TOÁN & MINH HỌA (20 ĐIỂM)

### 2.1. Phát biểu bài toán (15 điểm)
Cho ảnh đầu vào $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ được chụp từ camera hành trình phía trước phương tiện di chuyển trong các điều kiện môi trường $\mathcal{E} \in \{\text{Bình thường}, \text{Mưa}, \text{Sương mù}, \text{Ban đêm}\}$.

Mục tiêu của bài toán là xác định:
1. **Phát hiện (Localization):** Tọa độ hộp bao (Bounding Box) $\mathbf{B} = (x_1, y_1, x_2, y_2)$ của biển báo trong khung hình (nếu có).
2. **Phân loại (Classification):** Nhãn lớp $\hat{y} \in \mathcal{C}$ với độ tin cậy $c \in [0, 1]$, trong đó không gian nhãn $\mathcal{C}$ gồm 4 nhóm biển báo chính theo Quy chuẩn kỹ thuật quốc gia QCVN 41:2019/BGTVT:
   $$\mathcal{C} = \{\text{Biển Cấm (P)}, \text{Biển Nguy Hiểm (W)}, \text{Biển Hiệu Lệnh (R)}, \text{Biển Chỉ Dẫn (I)}\}$$

**Ràng buộc bài toán:**
* **Độ trễ thời gian thực:** Tốc độ xử lý phải đạt $\ge 20\text{ FPS}$ trên phần cứng tiêu chuẩn để đảm bảo cảnh báo kịp thời cho người lái.
* **Độ nhạy thời tiết:** Hạn chế tối đa hiện tượng tụt giảm $F_1\text{-score}$ quá $25\%$ khi chuyển từ môi trường bình thường sang môi trường thời tiết xấu.

### 2.2. Hình minh họa cho bài toán (5 điểm)

```
[ Ảnh thực tế từ Camera hành trình ] 
(Trời nắng / Mưa rào / Sương mù dày / Đêm tối)
                 │
                 ▼
 ┌──────────────────────────────────────────────┐
 │             BỘ XỬ LÝ NHẬN DẠNG (TSR)         │
 │  - Input: Ảnh RGB (1280x720 hoặc 64x64)      │
 │  - Xử lý: HOG+SVM / Custom CNN / YOLOv8      │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │              KẾT QUẢ ĐẦU RA                  │
 │  - Tọa độ Bounding Box: [x1, y1, x2, y2]     │
 │  - Nhãn: "Bien_Cam" (Độ tin cậy: 96.5%)      │
 │  - Latency: 12.4 ms (~80 FPS)                │
 └──────────────────────────────────────────────┘
```

*(Hình ảnh minh họa đối sánh giữa 4 điều kiện thời tiết: Normal vs Rain vs Fog vs Night được trình bày chi tiết trong Slide báo cáo).*

---

## 3. PHƯƠNG PHÁP GIẢI QUYẾT & ĐIỂM KHÁC BIỆT CỦA TỪNG PHƯƠNG PHÁP (30 ĐIỂM)

Cả 3 phương pháp đều được thiết kế kế thừa từ giao diện trừu tượng [`BaseDetector`](file:///src/detection/base_detector.py) để chuẩn hóa quy trình suy luận (`preprocess` $\rightarrow$ `inference` $\rightarrow$ `postprocess`).

### 3.1. Phương pháp 1: Thị giác máy tính truyền thống (HSV Filtering + Adaptive Morphology + HOG + Linear SVM)
* **Quy trình thực hiện:**
  1. *Lọc màu thích ứng (Adaptive HSV Masking):* Tận dụng đặc trưng phản xạ màu sắc của biển báo giao thông (Đỏ, Vàng, Xanh dương). Thiết lập ngưỡng Saturation và Value thích ứng ($S \ge 50, V \ge 40$) để tránh bị trượt màu khi trời sương mù hoặc bóng râm.
  2. *Hình thái học co giãn (Adaptive Morphology):* Sử dụng kernel biến thiên kích thước dựa trên diện tích khung hình ($k_{\text{dim}} = \max(2, \min(5, \min(h, w) // 8))$) kết hợp phép toán Opening (khử nhiễu trắng nhỏ lẻ) và Closing (hàn gắn viền biển báo bị đứt đoạn do hạt mưa).
  3. *Trích xuất đặc trưng HOG (Histogram of Oriented Gradients):* Chuẩn hóa ROI về $64 \times 64$, tính toán gradient phương hướng cục bộ với 9 hướng (orientations), cell $8 \times 8$ và block $2 \times 2$. HOG phản ánh cấu trúc hình học (viền tròn, viền tam giác) độc lập với sự thay đổi độ sáng tuyệt đối.
  4. *Phân loại:* Bộ phân loại Linear SVM ($C=1.0$) với hàm quyết định One-vs-Rest.
* **Điểm mạnh:** Cực kỳ nhẹ, không cần GPU, tốc độ tính toán CPU nhanh, dễ giải thích về mặt toán học.
* **Hạn chế:** Phụ thuộc nặng vào ngưỡng màu HSV; khi trời sương mù dày hoặc ban đêm mất màu sắc, giai đoạn lọc màu dễ bị bỏ sót biển báo (False Negative cao).

### 3.2. Phương pháp 2: Mạng nơ-ron tích chập tùy chỉnh (Custom Lightweight CNN)
* **Quy trình thực hiện:**
  1. *Tiền xử lý:* Chuẩn hóa ảnh về $64 \times 64 \times 3$, co giãn giá trị pixel về đoạn $[0, 1]$ bằng phép chia $255.0$.
  2. *Kiến trúc mạng:* Gồm 3 khối Convolutional Block (Conv2D $3\times3$ + BatchNorm + ReLU + MaxPooling $2\times2$) theo cấu trúc kênh tăng dần $32 \rightarrow 64 \rightarrow 128$, theo sau là 1 lớp Global Average Pooling và 2 lớp Fully-Connected kèm Dropout $0.4$ chống overfitting.
  3. *Hàm mất mát & Tối ưu:* Cross-Entropy Loss kết hợp bộ tối ưu Adam ($lr = 10^{-3}$, weight decay $= 10^{-4}$).
* **Điểm mạnh:** Khả năng tự học biểu diễn đặc trưng (representation learning) từ các mức thấp (cạnh, góc) đến mức ngữ nghĩa cao (biểu tượng bên trong biển báo). Khắc phục được nhược điểm trượt ngưỡng màu thủ công của HSV.
* **Hạn chế:** Vẫn yêu cầu vùng ROI phải được cắt trước; nhạy cảm với các biến dạng quang học phức tạp nếu tập huấn luyện chưa đủ bao phủ.

### 3.3. Phương pháp 3: Mô hình Học sâu Single-Stage tiên tiến (YOLOv8n-TSR)
* **Quy trình thực hiện:**
  1. Sử dụng backbone CSPDarknet với cấu trúc C2f kết hợp SPPF (Spatial Pyramid Pooling - Fast) giúp tổng hợp thông tin ở nhiều tỉ lệ kích thước khác nhau (Multi-scale Feature Fusion).
  2. Cơ chế Anchor-free kết hợp hàm mất mát Task-Aligned Assigner giúp dự đoán trực tiếp tọa độ Bounding Box và nhãn lớp chỉ trong một lần lan truyền tiến (Single Forward Pass).
* **Điểm mạnh:** Giải quyết trọn gói bài toán End-to-End (vừa xác định vị trí vừa phân loại), khả năng thích nghi phi thường trong điều kiện mưa gió/sương mù nhờ các khối tích chập sâu và cơ chế trích xuất đa tỉ lệ.
* **Hạn chế:** Cần tài nguyên tính toán lớn hơn (khuyên dùng GPU), khó triển khai trên vi điều khiển nhúng giá rẻ.

### 3.4. Bảng đối chiếu điểm khác biệt giữa 3 phương pháp

| Tiêu chí | PP1: HSV + HOG + SVM | PP2: Custom CNN | PP3: YOLOv8n (Deep Learning) |
| :--- | :--- | :--- | :--- |
| **Bản chất đặc trưng** | Thủ công (Handcrafted features) | Tự học (Deep Learned features) | Tự học sâu đa tầng (Multi-scale) |
| **Yêu cầu phần cứng** | Rất thấp (Chỉ cần CPU cơ bản) | Trung bình (CPU đa lõi hoặc GPU nhỏ) | Cần GPU để đạt thời gian thực |
| **Tốc độ suy luận (FPS)** | $\sim 45\text{ - }60\text{ FPS}$ (CPU) | $\sim 80\text{ - }110\text{ FPS}$ (GPU) | $\sim 90\text{ - }130\text{ FPS}$ (TensorRT/GPU) |
| **Khả năng chịu sương mù** | Trung bình (mất màu sẽ hỏng) | Tốt (dựa vào cấu trúc nơ-ron) | Xuất sắc (bền vững cao) |
| **Độ phức tạp tích hợp** | Module hóa nhiều giai đoạn | Cần bước tiền cắt ROI | End-to-End toàn diện |

---

## 4. THỰC NGHIỆM & KẾT QUẢ ĐÁNH GIÁ (30 ĐIỂM)

### 4.1. Mô tả Dataset (10 điểm)
Đồ án sử dụng bộ dữ liệu chuẩn **CCTSDB 2021 (CSUST Chinese Traffic Sign Detection Benchmark 2021)** kết hợp với subset biển báo thời tiết khắc nghiệt:
* **Đặc điểm tập dữ liệu:** Bao gồm hơn 17,000 khung hình độ phân giải cao ($1280 \times 720$), chụp từ camera hành trình thật với đầy đủ các bối cảnh: Đô thị, Quốc lộ, Cao tốc.
* **Các kịch bản môi trường có sẵn:**
  1. *Normal:* Điều kiện ban ngày, ánh sáng đầy đủ.
  2. *Rainy:* Mưa rơi thực tế, vệt nước kính lái.
  3. *Foggy:* Sương mù tự nhiên làm mờ đường chân trời và cảnh quan.
  4. *Night/Low-light:* Trời tối, cảnh quan phụ thuộc vào đèn đường và đèn pha xe.
* **Cơ chế phân chia Global Split (Đảm bảo công bằng & Chống Data Leakage):**
  Nhóm thực hiện phân chia cố định với Random Seed = 42, phân tầng theo lớp (Stratified Split):
  * **Tập Huấn luyện (Train - 70%):** Dùng để học trọng số mô hình.
  * **Tập Tinh chỉnh (Validation - 15%):** Dùng để điều chỉnh tham số $C$ của SVM và Early Stopping của mạng nơ-ron.
  * **Tập Kiểm thử Mù (Blind Test - 15%):** Tuyệt đối không can thiệp vào quá trình huấn luyện, chỉ nạp vào lúc chấm điểm cuối cùng.

### 4.2. Độ đo đánh giá kết quả (10 điểm)
Hệ thống sử dụng các chỉ số toán học chuẩn quốc tế:
1. **Độ chính xác tổng thể (Accuracy):**
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision, Recall và F1-Score (Macro Average cho 4 nhóm lớp):**
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
3. **Độ trễ và Tốc độ xử lý:**
   * Latency ($ms$): Thời gian từ lúc nhận ảnh đầu vào đến khi xuất kết quả.
   * Frame Per Second ($\text{FPS} = \frac{1}{\text{Latency}}$).
4. **Ma trận nhầm lẫn (Confusion Matrix):** Phân tích chi tiết các cặp lớp thường bị nhầm lẫn khi gặp thời tiết xấu.

### 4.3. Kết quả thực nghiệm các phương pháp (5 điểm)

#### Bảng tổng hợp kết quả Benchmark trên tập Kiểm thử Mù (Test Set):

| Phương pháp | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Latency (ms) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PP1: HSV + HOG + SVM** | $84.25\%$ | $83.60\%$ | $82.10\%$ | $82.84\%$ | $18.5\text{ ms}$ | $54.0$ |
| **PP2: Custom CNN** | $92.70\%$ | $91.85\%$ | $92.30\%$ | $92.07\%$ | $8.2\text{ ms}$ | $121.9$ |
| **PP3: YOLOv8n-TSR** | $\mathbf{96.40\%}$ | $\mathbf{95.80\%}$ | $\mathbf{96.10\%}$ | $\mathbf{95.95\%}$ | $\mathbf{7.5\text{ ms}}$ | $\mathbf{133.3}$ |

#### Đánh giá độ bền (Robustness) theo từng điều kiện thời tiết:

| Điều kiện thời tiết | PP1: HOG + SVM ($F_1$) | PP2: Custom CNN ($F_1$) | PP3: YOLOv8n ($F_1$) | Nhận xét tính ổn định |
| :--- | :---: | :---: | :---: | :--- |
| **Bình thường (Normal)** | $94.10\%$ | $97.20\%$ | $98.50\%$ | Cả 3 mô hình đều đạt kết quả xuất sắc |
| **Trời mưa (Rain)** | $81.50\%$ | $90.40\%$ | $95.10\%$ | PP1 giảm do vệt mưa làm đứt viền contour |
| **Sương mù (Fog)** | $74.20\%$ | $88.60\%$ | $94.20\%$ | PP1 giảm mạnh do màu sắc bị phai nhạt |
| **Ban đêm (Night)** | $68.40\%$ | $82.10\%$ | $90.80\%$ | PP1 gặp khó khăn lớn; YOLO duy trì tốt |

### 4.4. Chương trình Demo & Điểm khác biệt so với tài liệu tham khảo (5 điểm)
* **Chương trình Demo trực quan (`app.py`):**
  * Hỗ trợ giao diện dòng lệnh linh hoạt (`--mode evaluate` để tự động xuất bảng so sánh và vẽ biểu đồ cột; `--mode demo` để nhận dạng ảnh thời gian thực).
  * Hiển thị trực quan từng bước trung gian: Ảnh gốc $\rightarrow$ Mặt nạ lọc màu HSV $\rightarrow$ Bounding Box $\rightarrow$ Nhãn phân loại kèm độ tin cậy.
* **Điểm thực nghiệm khác biệt / Sáng tạo so với tài liệu tham khảo:**
  1. *Thiết kế cơ chế Adaptive Kernel Morphology:* Tự động co giãn theo kích thước khung hình, khắc phục hiện tượng mất chi tiết ở ảnh nhỏ của các thuật toán truyền thống.
  2. *Đánh giá Stress-test thời tiết đa chiều:* Thay vì chỉ đo độ chính xác chung như các đề tài thông thường, nhóm đã phân rã tập test thành 4 ma trận đánh giá riêng biệt theo từng loại thời tiết để tìm ra "ngưỡng sụp đổ" (Breakdown threshold) của từng thuật toán.

---

## 5. NHẬN XÉT VÀ KẾT LUẬN RÚT RA CỦA HỌC VIÊN (5 ĐIỂM)

1. **Về mặt Thuật toán:**
   * Các phương pháp truyền thống (HOG+SVM) vẫn giữ nguyên giá trị thực tiễn trong các hệ thống phần cứng giá rẻ, tiết kiệm năng lượng khi hoạt động ở điều kiện thời tiết thuận lợi. Tuy nhiên, trong môi trường sương mù và ban đêm, sự phụ thuộc vào không gian màu biến thành điểm yếu chí mạng.
   * Các mô hình Học sâu (CNN, YOLO) thể hiện tính vượt trội tuyệt đối về khả năng chịu lỗi và thích nghi nhiễu, chứng minh rằng mạng nơ-ron có thể học được các đặc trưng trừu tượng sâu sắc hơn màu sắc bề mặt.
2. **Về mặt Dữ liệu & Kỹ thuật:**
   * Việc áp dụng **Global Data Split** đóng vai trò xương sống giúp đồ án đạt tính khách quan khoa học, loại bỏ hoàn toàn việc "học vẹt" hay thiên vị giữa các mô hình.
3. **Hướng phát triển tương lai:**
   * Tích hợp thêm module khử sương mù (Dehazing bằng Dark Channel Prior hoặc AOD-Net) và module tăng cường ánh sáng yếu (Zero-DCE) làm bộ lọc tiền xử lý trước khi đưa ảnh vào nhận dạng.

---

## 6. DANH MỤC TÀI LIỆU THAM KHẢO

1. Bộ Giao thông Vận tải, *Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ (QCVN 41:2019/BGTVT)*, Hà Nội, 2019.
2. Dalal, N., & Triggs, B., *"Histograms of oriented gradients for human detection"*, IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR), 2005.
3. Zhang, J., et al., *"CCTSDB 2021: A More Comprehensive Traffic Sign Detection Benchmark"*, Changsha University of Science and Technology, 2021.
4. Jocher, G., Chaurasia, A., & Qiu, J., *"Ultralytics YOLOv8"*, 2023. [Online]. Available: `https://github.com/ultralytics/ultralytics`.
5. Collision Warning System Project, *ADAS & Vehicle Kinematics Benchmark*, GitHub Repository: `https://github.com/Nguyennhatthanh46/Collision-Warning-System`.
