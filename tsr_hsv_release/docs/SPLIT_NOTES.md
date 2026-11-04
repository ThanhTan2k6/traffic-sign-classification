# GHI CHÚ KỸ THUẬT VỀ PHÂN CHIA DỮ LIỆU & KIỂM SOÁT DATA LEAKAGE

> **Mục đích**: Tài liệu này phân tích chi tiết cơ chế phân chia dữ liệu (Train/Val/Test split) trong bài toán nhận diện biển báo giao thông thu thập từ video hành trình (dashcam), đối chiếu định lượng giữa hai cách tiếp cận và đề xuất phương án chuẩn hóa chung để so sánh công bằng.

---

## 1. Bản chất dữ liệu chuỗi khung hình (Sequential Video Frames)

Bộ dữ liệu biển báo giao thông Việt Nam gồm **3.216 ảnh** thực chất được trích xuất từ các đoạn video hành trình liên tục của camera gắn trên xe ô tô:
- Các ảnh có stem kề nhau (ví dụ: `0414.jpg`, `0415.jpg`, `0416.jpg`...) được ghi lại cách nhau chỉ vài phần mười giây.
- Khung cảnh đường sá, điều kiện ánh sáng, góc chụp và chính bản thân biển báo trong các frame kề nhau gần như giống hệt nhau (chỉ khác biệt nhẹ về kích thước pixel do xe di chuyển lại gần).

Do đó, nếu áp dụng phương pháp trộn ngẫu nhiên truyền thống (`random.shuffle` từng ảnh đơn lẻ) thường dùng cho dữ liệu ảnh tĩnh (như ImageNet hay CIFAR), một biển báo sẽ xuất hiện đồng thời ở frame 0414 (tập Train) và frame 0415 (tập Test). Khi đó, mô hình học máy chỉ cần "học vẹt" (memorize) hình ảnh quen thuộc thay vì học khả năng tổng quát hóa (generalization) trên các cung đường mới.

---

## 2. Cách chia của tôi: Phân chia theo khối video liên tục (Block-based Split)

Nhằm đảm bảo tính khách quan và khoa học, tôi áp dụng thuật toán `block_split` trong `code/common.py`:

### Cơ chế kỹ thuật:
1. **Sắp xếp theo thứ tự thời gian**: Sắp xếp 3.216 ảnh theo số thứ tự stem (thứ tự video thực tế).
2. **Gom khối liên tục (Block assignment)**: Chia toàn bộ chuỗi video thành các khối gồm **100 frame liên tục**. Toàn bộ 100 frame trong cùng một khối được phân bổ trọn vẹn vào một tập duy nhất (Train, Val, hoặc Test).
3. **Cắt khoảng đệm an toàn (Gap frames)**: Tại ranh giới giữa 2 khối thuộc 2 tập khác nhau, tự động loại bỏ **5 frame** kề ranh giới (`gap = 5`) để triệt tiêu hoàn toàn sự tương đồng hình ảnh giữa điểm kết thúc của tập này và điểm bắt đầu của tập kia.
4. **Kiểm tra rò rỉ ranh giới**:
   - Đếm số cặp frame kề nhau ($s$ và $s+1$) nằm ở hai tập khác nhau:
     $$\text{Số cặp vi phạm} = \mathbf{0} \ / \ \mathbf{3.215} \quad (\mathbf{0.00\%})$$

### Thống kê mẫu theo split của tôi:
- **Tập Train**: 2.257 ảnh | 16.117 bbox hợp lệ ($\ge 8$px, thuộc 5 nhóm).
- **Tập Val**: 409 ảnh | 558 bbox hợp lệ.
- **Tập Test**: 550 ảnh | 1.821 bbox Ground-Truth (không tính Camera).

---

## 3. Đối chiếu số liệu đo đạc với cách chia của bạn tôi (`data/global_splits`)

Phân tích trực tiếp từ các tệp chỉ mục trong repo của bạn tôi (`data/global_splits/`):

### 3.1. Thuật toán phân chia
Repo của bạn tôi sử dụng hàm `split_dataset_stratified` (trộn ngẫu nhiên `random.shuffle` từng ảnh đơn lẻ theo tỷ lệ 70% Train, 15% Val, 15% Test).

### 3.2. Số liệu đo đạc thực tế
- **Số lượng ảnh**: Train 2.252 ảnh (70.0%), Val 483 ảnh (15.0%), Test 481 ảnh (15.0%). Tổng cộng 3.216 ảnh.
- **Số lượng crop biển báo**: Train 5.545 crop, Val 1.177 crop, Test 1.136 crop. Tổng cộng 7.858 crop (chỉ gồm 4 nhóm chính, đã lọc sạch box $<10$px).
- **Số cặp frame kề nhau ($s, s+1$) nằm ở 2 tập khác nhau**:
  $$\text{Số cặp vi phạm ranh giới} = \mathbf{1.506} \ / \ \mathbf{3.215} \quad (\mathbf{46.84\%})$$
  *(Tức là cứ 2 frame liên tiếp nhau trong video, có tới xấp xỉ 47% khả năng một frame nằm ở tập Train và frame ngay cạnh nằm ở tập Val hoặc Test).*

### 3.3. Khác biệt về không gian nhãn và định nghĩa nhóm
- **Repo của bạn**: Chỉ phân loại **4 nhóm** mục tiêu (`Bien_Cam`, `Bien_Nguy_Hiem`, `Bien_Hieu_Lenh`, `Bien_Chi_Dan`), tương ứng với **47 / 52 lớp**. Loại bỏ hoàn toàn nhóm Biển phụ (`S.*`), lớp `Camera` và lớp `B.8a`.
- **Project của tôi**: Phân loại **5 nhóm** (`cấm`, `cảnh báo`, `hiệu lệnh`, `chỉ dẫn`, `phụ`), tương ứng với **51 / 52 lớp** cho classifier. Gộp `B.8a` vào nhóm Cấm sau khi kiểm toán hình thái học; loại `Camera` khỏi classifier nhưng vẫn giữ trong detector và gán `ignored` khi đánh giá.

### 3.4. Ghi nhận về giả định điểm số
- Trong README của bạn tôi, mô hình Enhanced LeNet-5 đạt độ chính xác **99.74%** và Macro-F1 **99.59%** trên tập Blind Test gồm 1.136 crop cắt sẵn.
- **Nhận định kỹ thuật (giả định, chưa kiểm chứng)**: Hiệu năng cao trên tập test có thể được đóng góp bởi nhiều yếu tố, trong đó độ tương đồng cao giữa các khung hình hành trình liên tiếp giữa tập train và test là một khả năng kỹ thuật đáng lưu ý (*đây là giả định dựa trên đo đạc 46.84% cặp frame kề nhau nằm ở hai tập khác nhau, chưa được kiểm chứng thực nghiệm bằng cách huấn luyện lại mô hình của bạn trên split không rò rỉ*).
- Để kiểm chứng khoa học giả định trên, cần thực hiện chạy chéo:
  1. Chạy mô hình truyền thống (HSV + HOG + SVM/Boosting) của tôi trên đúng tập split của bạn tôi.
  2. Chạy mô hình Enhanced LeNet-5 của bạn tôi trên tập split chia theo khối video (block-based split) của tôi.

---

## 4. Đề xuất chung để so sánh công bằng (Constructive Proposal)

Trên tinh thần hợp tác xây dựng và minh bạch khoa học:
1. **Thống nhất một bộ phân chia chuẩn chung (Unified Benchmark Split)**: Hai bên nên cùng áp dụng cơ chế chia theo khối video liên tục (Block Split) với rò rỉ ranh giới bằng 0. Khi đó, cả phương pháp Deep Learning (LeNet-5/YOLOv8) lẫn phương pháp Truyền thống (Color+HOG+SVM) đều được đo đạc năng lực tổng quát hóa thực sự trên các cung đường chưa từng thấy.
2. **Thống nhất không gian nhãn đánh giá**:
   - Có thể thống nhất đánh giá trên **4 nhóm chính** theo chuẩn của bạn tôi để so sánh độ chính xác đối đầu.
   - Hoặc đánh giá trên **5 nhóm** (bao gồm biển phụ) và báo cáo kèm chỉ số **Major4 F1** (trung bình F1 trên 4 nhóm chính, loại trừ nhóm phụ) như cách làm tại Stage 2 của tôi để không bị biến động thống kê bởi cỡ mẫu nhỏ của nhóm biển phụ.