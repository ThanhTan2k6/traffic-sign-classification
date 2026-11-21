# KỊCH BẢN THUYẾT TRÌNH BẢO VỆ ĐỒ ÁN CS231 & TÀI LIỆU VẤN ĐÁP (Q&A)
**Đề tài:** Nhận dạng Biển báo Giao thông Hai giai đoạn (YOLO11s + SVM / Enhanced LeNet-5)  
**Người phụ trách:** Nguyễn Thanh Tân (Mô-đun M2, M3B Enhanced LeNet-5 & Thực nghiệm Đánh giá)  
**Thời gian trình bày ước tính:** 7 -- 10 phút (trong tổng thời lượng 15 phút của nhóm)

---

## MỤC LỤC
1. [Phân công thuyết trình trong nhóm](#1-phân-công-thuyết-trình-trong-nhóm)
2. [Cấu trúc bài thuyết trình (Khung sư phạm: Lý thuyết $\rightarrow$ Áp dụng thực tế)](#2-cấu-trúc-bài-thuyết-trình)
3. [Kịch bản lời thoại chi tiết theo từng Slide](#3-kịch-bản-lời-thoại-chi-tiết-theo-từng-slide)
   - [Slide 6: Hai giai đoạn chính (Pipeline Tổng thể)](#slide-6-hai-giai-đoạn-chính-pipeline-tổng-thể)
   - [Slide 11: Tổng quan Mạng nơ-ron tích chập (CNN) & LeNet-5](#slide-11-tổng-quan-mạng-nơ-ron-tích-chập-cnn--lenet-5)
   - [Slide 12: Thiết kế kiến trúc Enhanced LeNet-5 tùy biến](#slide-12-thiết-kế-kiến-trúc-enhanced-lenet-5-tùy-biến)
   - [Slide 13: Vòng lặp huấn luyện & Tối ưu hóa siêu tham số](#slide-13-vòng-lặp-huấn-luyện--tối-ưu-hóa-siêu-tham-số)
   - [Slide 14: Bộ dữ liệu áp dụng & Chiến lược phân chia Block Split](#slide-14-bộ-dữ-liệu-áp-dụng--chiến-lược-phân-chia-block-split)
   - [Slide 15: Các độ đo đánh giá mô hình](#slide-15-các-độ-đo-đánh-giá-mô-hình)
   - [Slide 16: Kết quả thực nghiệm đối chuẩn trên 1.813 mẫu ROI Test](#slide-16-kết-quả-thực-nghiệm-đối-chuẩn-trên-1813-mẫu-roi-test)
   - [Slide 17: Phân tích Ma trận nhầm lẫn & Các ca phân loại sai](#slide-17-phân-tích-ma-trận-nhầm-lẫn--các-ca-phân-loại-sai)
   - [Slide 18: Đánh giá toàn trình trên 550 ảnh hiện trường (Blind Test)](#slide-18-đánh-giá-toàn-trình-trên-550-ảnh-hiện-trường-blind-test)
   - [Slide 19: Sản phẩm hoàn thiện: Web App Demo & CLI Tool](#slide-19-sản-phẩm-hoàn-thiện-web-app-demo--cli-tool)
   - [Slide 20: Kết luận & Hướng phát triển](#slide-20-kết-luận--hướng-phát-triển)
4. [Bộ câu hỏi Phản biện & Kịch bản Trả lời Vấn đáp (Q&A Defense Sheet)](#4-bộ-câu-hỏi-phản-biện--kịch-bản-trả-lời-vấn-đáp-qa-defense-sheet)
   - [Câu 1: Tại sao gọi là Enhanced LeNet-5? Nó khác gì LeNet-5 gốc 1998?](#câu-1-tại-sao-gọi-là-enhanced-lenet-5-nó-khác-gì-lenet-5-gốc-1998)
   - [Câu 2: Kiến trúc này đã tồn tại chưa hay tự chế? Có copy trên mạng không?](#câu-2-kiến-trúc-này-đã-tồn-tại-chưa-hay-tự-chế-có-copy-trên-mạng-không)
   - [Câu 3: Sao không dùng ResNet-18 hay MobileNetV3 tiền huấn luyện cho tiện?](#câu-3-sao-không-dùng-resnet-18-hay-mobilenetv3-tiền-huấn-luyện-cho-tiện)
   - [Câu 4: Tại sao chọn kích thước $64\times 64$ mà không phải $28\times 28$ hay $224\times 224$?](#câu-4-tại-sao-chọn-kích-thước-64times-64-mà-không-phải-28times-28-hay-224times-224)
   - [Câu 5: Bài toán này có phân loại được biển bị cây che, nhòe mờ do mưa, nhiễu không?](#câu-5-bài-toán-này-có-phân-loại-được-biển-bị-cây-che-nhòe-mờ-do-mưa-nhiễu-không)
   - [Câu 6: Tại sao cần 7 độ đo đánh giá? Macro-F1 có ý nghĩa gì hơn Accuracy?](#câu-6-tại-sao-cần-7-độ-đo-đánh-giá-macro-f1-có-ý-nghĩa-gì-hơn-accuracy)
   - [Câu 7: Tại sao không dùng YOLO phân loại luôn mà phải tách 2 giai đoạn?](#câu-7-tại-sao-không-dùng-yolo-phân-loại-luôn-mà-phải-tách-2-giai-đoạn)
   - [Câu 8: 12 ca nhầm lẫn ở Level B là gì? Tại sao lại nhầm?](#câu-8-12-ca-nhầm-lẫn-ở-level-b-là-gì-tại-sao-lại-nhầm)

---

## 1. PHÂN CÔNG THUYẾT TRÌNH TRONG NHÓM

* **Từ Lê Tuấn Anh:** Trình bày từ Slide 1 đến Slide 10:
  * Slide 1 -- 5: Dẫn nhập thực tế, Phát biểu bài toán cốt lõi Input/Output, Ví dụ và Mở rộng pipeline toàn trình.
  * Slide 6: Tổng quan 2 giai đoạn (Tuấn Anh giới thiệu pha Offline và pha Online).
  * Slide 7 -- 8: **Phương pháp 1 (YOLO11s):** Slide 7 nói sơ về lý thuyết Single-stage YOLO11 $\rightarrow$ Slide 8 nói về cách áp dụng vào bài toán định vị (1-Class, pretrain CCTSDB, finetune VN, 30 epochs, batch 4, NMS).
  * Slide 9 -- 10: **Phương pháp 2 (Thị giác truyền thống):** Slide 9 nói sơ về lý thuyết HOG và SVM RBF $\rightarrow$ Slide 10 nói về cách áp dụng vào bài toán (chuỗi đặc trưng S0--S3, vector 457 chiều, tham số SVM).
* **Nguyễn Thanh Tân (Bạn):** Bắt đầu nhận lượt nói chính thức từ **Slide 11 đến Slide 21 (Hết bài)**:
  * Slide 11 -- 13: **Phương pháp 3 (Học sâu Hiện đại - Enhanced LeNet-5):**
    * *Slide 11:* Tổng quan lý thuyết Mạng tích chập (CNN) & Mô hình kinh điển LeNet-5 (LeCun 1998).
    * *Slide 12:* Áp dụng vào bài toán: 5 điểm nâng cấp kỹ thuật, bảng cấu trúc tầng chi tiết, tham số $0.56\text{M}$.
    * *Slide 13:* Áp dụng vào bài toán: Cấu hình siêu tham số, vòng lặp huấn luyện 20 epochs, hàm mất mát Cross-Entropy, bộ tối ưu Adam, StepLR, Augmentation và đồ thị hội tụ.
  * Slide 14 -- 18: **Dữ liệu thực nghiệm & Đánh giá kết quả (Sau khi đã trình bày xong toàn bộ phương pháp):**
    * *Slide 14:* Bộ dữ liệu áp dụng (3.216 ảnh, 7.814 bbox) & Chiến lược chia Block Split (70/15/15) chống rò rỉ dữ liệu video.
    * *Slide 15:* Các độ đo đánh giá bài toán phân loại (Accuracy, Precision, Recall, Macro-F1, Confusion Matrix).
    * *Slide 16:* Kết quả đối chuẩn phân loại trên 1.813 mẫu ROI Test (so sánh trực diện với SVM, ResNet-18, MobileNetV3).
    * *Slide 17:* Phân tích chuyên sâu Ma trận nhầm lẫn (Level A vs Level B, 12 ca nhầm lẫn giữa Hiệu lệnh và Chỉ dẫn).
    * *Slide 18:* Đánh giá toàn trình trên 550 ảnh hiện trường (Blind Scene Benchmark).
  * Slide 19 -- 20: Sản phẩm hoàn thiện (Web App Streamlit & CLI Tool), Kết luận & Hướng phát triển tương lai.

---

## 2. CẤU TRÚC BÀI THUYẾT TRÌNH
Cấu trúc chuẩn sư phạm môn CS231:
$$\text{Lý thuyết phương pháp} \longrightarrow \text{Áp dụng vào bài toán đề tài (Thông số, Vòng lặp)} \longrightarrow \text{Tập dữ liệu \& Thực nghiệm}$$

Mỗi phương pháp đều có:
1. **1 slide:** Giới thiệu ngắn gọn lý thuyết cốt lõi (Core theory / Concept).
2. **1 -- 2 slide:** Cách áp dụng phương pháp đó vào bài toán của nhóm (Input size, tiền xử lý, cấu trúc mạng, siêu tham số, vòng lặp huấn luyện, hàm mất mát, bộ tối ưu, độ trễ).

---

## 3. KỊCH BẢN LỜI THOẠI CHI TIẾT THEO TỪNG SLIDE (NGUYỄN THANH TÂN)

### Slide 6: Hai giai đoạn chính (Pipeline Tổng thể)
*(Chuyển giao lượt nói từ Tuấn Anh sang Tân)*
> *"Kính thưa quý Thầy/Cô, tiếp nối phần trình bày của bạn Tuấn Anh về nhánh Thị giác truyền thống, em là **Nguyễn Thanh Tân**. Em xin phép đại diện nhóm trình bày tiếp nhánh **Học sâu Hiện đại (Deep Learning)** cùng toàn bộ quá trình thực nghiệm, đánh giá kết quả và sản phẩm của đề tài.*
> 
> *Nhìn vào sơ đồ trên Slide 6, sau khi hộp bao biển báo được YOLO11s phát hiện và cắt trích (ROI Crop ở Mô-đun M2), ảnh con sẽ được đưa vào Mô-đun M3B với mô hình **Enhanced LeNet-5**. Đây là mạng nơ-ron tích chập tùy biến được nhóm thiết kế chuyên biệt để phân loại 4 nhóm quy chuẩn với độ chính xác cao và tốc độ siêu nhẹ."*

---

### Slide 11: Tổng quan Mạng nơ-ron tích chập (CNN) & LeNet-5
*(Slide 1 của Phương pháp 3: Lý thuyết nền tảng)*
> *"Để giải quyết bài toán phân loại hình ảnh tự động, nhóm em lựa chọn hướng tiếp cận dựa trên **Mạng nơ-ron tích chập (Convolutional Neural Network - CNN)**.*
> 
> *Về mặt nguyên lý, CNN vượt trội so với thị giác truyền thống ở chỗ: thay vì con người phải ngồi thiết kế đặc trưng thủ công như HOG hay HSV, CNN có khả năng **tự động học các bộ lọc (filters) tối ưu** trực tiếp từ dữ liệu qua 4 thành phần:*
> 1. * **Tầng Tích chập (Convolution):** Quét các kernel trích xuất đặc trưng cục bộ (Local Receptive Fields) và chia sẻ trọng số (Weight Sharing), giúp mô hình bất biến với vị trí xuất hiện của hoa văn.*
> 2. * **Hàm kích hoạt phi tuyến:** Đưa tính phi tuyến vào mạng để nắm bắt các hình thái phức tạp.*
> 3. * **Tầng Lấy mẫu (Pooling):** Giảm kích thước không gian, tạo tính bất biến tịnh tiến và nén thông tin.*
> 4. * **Tầng Kết nối đầy đủ (Dense / Fully Connected):** Tổng hợp đặc trưng mức cao và xuất phân phối xác suất qua hàm Softmax.*
> 
> *Kiến trúc kinh điển đặt nền móng cho CNN hiện đại chính là **LeNet-5 của Giáo sư Yann LeCun năm 1998**. Đây là một cấu trúc tuần tự phẳng cực kỳ thanh thoát gồm các khối `Conv -> Pool` xen kẽ rồi đến các tầng Dense, vốn được thiết kế để nhận dạng chữ số viết tay MNIST với ảnh xám $28 \times 28$."*

---

### Slide 12: Thiết kế kiến trúc Enhanced LeNet-5 tùy biến
*(Slide 2 của Phương pháp 3: Áp dụng vào bài toán - Kiến trúc mạng & Cải tiến)*
> *"Tuy nhiên, khi đem LeNet-5 cổ điển vào bài toán phân loại biển báo giao thông thực tế ngoài đường phố, mô hình bộc lộ 5 hạn chế nghiêm trọng. Nhóm em đã **nâng cấp toàn diện 5 yếu tố kỹ thuật hiện đại** để tạo nên kiến trúc **Enhanced LeNet-5**:*
> 
> 1. * **Đầu vào:** Nâng từ ảnh xám 1 kênh $28\times 28$ lên **ảnh màu RGB 3 kênh, kích thước $64 \times 64$**, nhằm bảo toàn trọn vẹn màu sắc quy chuẩn (đỏ, vàng, xanh) và các hoa văn chi tiết như mũi tên, người đi bộ.*
> 2. * **Hàm kích hoạt:** Thay Sigmoid/Tanh bằng **ReLU ($\max(0, x)$)**, triệt tiêu dứt điểm hiện tượng triệt tiêu đạo hàm (Vanishing Gradient) và tính toán cực nhanh.*
> 3. * **Tầng Pooling:** Thay Average Pooling bằng **Max Pooling ($2 \times 2$)**, giữ lại giá trị kích hoạt lớn nhất, tức bảo toàn các đường viền sắc nét của biển báo thay vì bị làm nhòe.*
> 4. * **Chuẩn hóa trung gian:** Tích hợp **Batch Normalization (BatchNorm2d)** sau mỗi tầng Conv, chống trôi dạt phân phối ẩn (Internal Covariate Shift), giúp mạng hội tụ nhanh và mượt mà.*
> 5. * **Chống học vẹt (Regularization):** Bổ sung **2 tầng Dropout ($p=0.3$ và $p=0.2$)** tại khối Classifier để ép mạng không phụ thuộc cục bộ vào các chi tiết nhiễu nền hay góc chụp.*
> 
> *Bảng cấu trúc chi tiết trên màn hình cho thấy mạng gồm **3 khối ConvBlock** mở rộng từ $3 \to 16 \to 32 \to 64$ kênh, duỗi phẳng 4.096 chiều qua tầng ẩn Dense 128 nơ-ron và xuất ra 4 logits tương ứng 4 nhóm quy chuẩn.*
> *Tổng số tham số của mô hình chỉ là **557.700 tham số ($\approx 0.56\text{M}$)**, dung lượng checkpoint chỉ **2.24 MB** và tốn **13.1 MFLOPs**. Cấu trúc này được 'may đo' vừa vặn hoàn hảo cho bài toán, tránh được sự cồng kềnh, lãng phí tài nguyên của các mạng lớn như ResNet-18 ($11.69\text{M}$ tham số)."*

---

### Slide 13: Vòng lặp huấn luyện & Tối ưu hóa siêu tham số
*(Slide 3 của Phương pháp 3: Áp dụng vào bài toán - Quá trình huấn luyện, Vòng lặp, Siêu tham số)*
> *"Để huấn luyện mô hình Enhanced LeNet-5 trên tập dữ liệu đề tài, nhóm thiết lập cấu hình kỹ thuật chặt chẽ:*
> * * **Dữ liệu đầu vào:** Ảnh ROI $64 \times 64 \times 3$, chuẩn hóa kênh theo phân phối chuẩn của ImageNet.*
> * * **Vòng lặp huấn luyện (Training Loop):** Mô hình được huấn luyện trong **20 vòng lặp (Epochs)**, kích thước lô **batch size = 64 mẫu/lô**, tương đương khoảng 85 bước lặp (iterations) cập nhật trọng số trong mỗi epoch trên 5.401 mẫu huấn luyện.*
> * * **Hàm mất mát (Loss function):** Sử dụng **Cross-Entropy Loss** cho bài toán phân loại đa lớp:*
>   $$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^4 y_c \log \hat{y}_c$$
> * * **Bộ tối ưu hóa (Optimizer):** Sử dụng thuật toán **Adam** với tốc độ học ban đầu $\text{lr}_0 = 10^{-3}$, hệ số suy giảm trọng số weight decay $= 10^{-4}$ để kiểm soát độ lớn trọng số.*
> * * **Chiến lược lập lịch bước (StepLR):** Tự động giảm một nửa tốc độ học sau mỗi 7 epochs ($\text{lr} \times 0.5$) để giúp mạng len lỏi vào đáy cực tiểu địa phương.*
> * * **Pipeline Tăng cường dữ liệu (Data Augmentation):** Áp dụng phép xoay ngẫu nhiên $\pm 10^\circ$ (mô phỏng xe rung lắc) và biến thiên độ sáng/độ tương phản $\pm 20\%$ (mô phỏng trời chói nắng hoặc bóng râm).*
> * *Quan sát đồ thị huấn luyện bên phải, hàm mất mát Train Loss giảm dốc mượt mà từ $0.65$ xuống dưới $0.01$, và độ chính xác kiểm chứng (Val Accuracy) đã chạm ngưỡng **99.82\%** ngay từ epoch thứ 13 mà không hề có dấu hiệu quá khớp (Overfitting)."*

---

### Slide 14: Bộ dữ liệu áp dụng & Chiến lược phân chia Block Split
*(Chuyển sang Section 4: Dữ liệu thực nghiệm & Đánh giá kết quả)*
> *"Sau khi đã xây dựng xong các phương pháp, nhóm tiến hành thực nghiệm trên tập dữ liệu thực tế:*
> * *Bộ dữ liệu đường phố Việt Nam do nhóm thu thập gồm **3.216 ảnh** chất lượng cao, chứa **7.814 bounding box biển báo hợp lệ**.*
> * *Để đánh giá khách quan và khoa học, nhóm áp dụng chiến lược **Block Split (Zero Data Leakage)**: Cắt chuỗi video theo các khối **100 khung hình liên tục** kết hợp **khoảng đệm an toàn 5 frames**.*
> * *Kỹ thuật này ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (Data Leakage) khi hai khung hình kề nhau trong video cùng xuất hiện ở cả tập train và tập test.*
> * *Dữ liệu được chia theo tỷ lệ chuẩn 70/15/15: Tập Train gồm 5.401 mẫu ROI, tập Val gồm 542 mẫu ROI, và tập Blind Test gồm **1.813 mẫu ROI độc lập**."*

---

### Slide 15: Các độ đo đánh giá mô hình
> *"Về các độ đo đánh giá hiệu năng bài toán phân loại:*
> * *Do dữ liệu thực tế có sự chênh lệch lớn giữa các nhóm (Biển Cấm chiếm số đông, Biển Chỉ Dẫn chiếm số ít), nhóm không chỉ nhìn vào **Accuracy** mà sử dụng **Macro-F1 Score** làm thước đo trung tâm.*
> * *Macro-F1 lấy trung bình cộng bình đẳng F1 của 4 lớp, đảm bảo mô hình không thể 'ăn gian' điểm bằng cách đoán lệch về lớp đa số.*
> * *Kết hợp cùng **Precision** (đo lường độ tin cậy, không báo sai) và **Recall** (đo lường khả năng không bỏ sót biển báo nguy hiểm).*
> * *Đặc biệt, nhóm đối chiếu trực quan bằng **Ma trận nhầm lẫn (Confusion Matrix)** và các thông số phần cứng như **FLOPs, Số tham số, Độ trễ (ms) và Tốc độ (FPS)** trên CPU."*

---

### Slide 16: Kết quả thực nghiệm đối chuẩn trên 1.813 mẫu ROI Test
> *"Bảng số liệu trên màn hình thể hiện kết quả đối chuẩn khách quan giữa các phương pháp trên tập 1.813 mẫu ROI Test độc lập:*
> * *Mô hình **Enhanced LeNet-5** của nhóm đạt kết quả cao nhất với độ chính xác **Accuracy 98.73\%**, **Recall 97.43\%** và **Macro-F1 0.9566**.*
> * *So sánh trực diện với **ResNet-18 tiền huấn luyện**, Enhanced LeNet-5 đạt độ chính xác tương đương thậm chí nhỉnh hơn một chút ($98.73\%$ vs $98.62\%$), nhưng ưu thế áp đảo nằm ở tốc độ: mô hình của nhóm đạt **911.1 FPS** (độ trễ chỉ **1.10 ms/ảnh**), nhanh hơn **gấp 11 lần** so với ResNet-18 (80 FPS) trên cùng vi xử lý CPU đơn lõi.*
> * *Đồng thời, nhánh Thị giác truyền thống **Color + HOG + SVM** do bạn Tuấn Anh xây dựng cũng đạt Accuracy rất cao là **98.57\%** với tốc độ 379 FPS, chứng minh bộ đặc trưng 457 chiều kết hợp HOG bắt viền và HSV bắt màu đã trích xuất xuất sắc hình thái biển báo."*

---

### Slide 17: Phân tích Ma trận nhầm lẫn & Các ca phân loại sai
> *"Đi sâu vào phân tích Ma trận nhầm lẫn ở Slide 17 để giải thích các ca lỗi thực tế:*
> * *Ở mức thử nghiệm **Level A** (ảnh crop lý tưởng đúng bounding box), cả hai mô hình đều phân loại chính xác tuyệt đối $100\%$ giữa Biển Cấm và Biển Nguy Hiểm.*
> * *Ở mức thử nghiệm **Level B** (ảnh crop từ kết quả định vị thực tế của YOLO11s):*
>   * *Trong tổng số 1.813 mẫu thử, Enhanced LeNet-5 chỉ có **12 ca dự đoán nhầm lẫn** (11 ca nhầm từ Hiệu lệnh sang Chỉ dẫn, và 1 ca nhầm từ Chỉ dẫn sang Hiệu lệnh).*
>   * *Nguyên nhân kỹ thuật: Cả 2 nhóm này đều có nền màu **xanh lam**. Khi biển ở cự ly xa ($< 25\text{ px}$) hoặc góc chụp nghiêng gắt của camera hành trình, hình tròn (Hiệu lệnh) bị méo nhẹ thành hình bầu dục, khiến mạng CNN nhầm lẫn với hình vuông (Chỉ dẫn).*
>   * *Tuy nhiên, hoàn toàn **không có bất kỳ ca nhầm lẫn nào giữa nhóm Cấm (đỏ) và Nguy hiểm (vàng)**, đảm bảo an toàn tuyệt đối cho các tình huống cảnh báo giao thông khẩn cấp."*

---

### Slide 18: Đánh giá toàn trình trên 550 ảnh hiện trường (Blind Test)
> *"Khi ghép nối toàn trình hệ thống trên 550 ảnh hiện trường độc lập (Scene Benchmark):*
> * *Hệ thống hoàn chỉnh **YOLO11s + Enhanced LeNet-5** đạt **Precision toàn trình 78.89\%** và **F1 toàn trình 82.12\%**.*
> * *Về độ trễ toàn trình (End-to-End Latency): Khâu phát hiện YOLO11s mất $10.5\text{ ms}$, khâu phân loại LeNet-5 chỉ mất $1.10\text{ ms}$, đưa tổng thời gian xử lý toàn trình về mức **14.23 ms/ảnh (đạt 70.3 FPS)**.*
> * *Tốc độ này vượt xa yêu cầu thời gian thực của camera hành trình thông thường (chỉ cần 30 FPS), khẳng định tính khả thi cao để triển khai vào sản phẩm thực tế."*

---

### Slide 19: Sản phẩm hoàn thiện: Web App Demo & CLI Tool
> *"Nhóm đã đóng gói toàn bộ mô hình thành sản phẩm thực tế:*
> * * **Web App tương tác bằng Streamlit:** Cho phép tải ảnh hoặc video hiện trường, tùy chỉnh ngưỡng tin cậy Confidence và NMS IoU, hiển thị trực quan bounding box gắn mã màu 4 nhóm theo quy chuẩn và bảng thống kê độ trễ chi tiết.*
> * * **CLI Tool:** Cung cấp giao diện dòng lệnh cho phép xử lý hàng loạt ảnh (Batch Processing) phục vụ tích hợp vào camera hành trình hoặc hệ thống biên.*
> * *Toàn bộ mã nguồn, checkpoint và tài liệu đã được công khai trên GitHub của nhóm."*

---

### Slide 20: Kết luận & Hướng phát triển
> *"Để tổng kết bài báo cáo, đề tài của nhóm đã hoàn thành xuất sắc các mục tiêu:*
> 1. *Xây dựng thành công hệ thống nhận dạng biển báo hai giai đoạn cân bằng hoàn hảo giữa độ chính xác ($98.73\%$) và độ trễ siêu tốc ($1.10\text{ ms}$ trên CPU).*
> 2. *Chứng minh được tính hiệu quả của triết lý **thiết kế mạng chuyên biệt, vừa vặn bài toán (Enhanced LeNet-5)**: siêu nhẹ ($0.56\text{M}$ params), chạy mượt trên chip nhúng mà không cần GPU.*
> 3. *Đóng gói hoàn chỉnh ứng dụng Web App và CLI Tool trực quan.*
> 
> *Về hướng phát triển tiếp theo, nhóm dự kiến mở rộng mô hình phân loại sâu xuống 43 phân lớp chi tiết theo QCVN 41:2019 và tích hợp thêm thuật toán ByteTrack để theo dõi biển báo mượt mà trên video liên tục.*
> 
> *Em xin chân thành cảm ơn quý Thầy/Cô đã chú ý lắng nghe. Nhóm em rất mong nhận được những góp ý và câu hỏi của Thầy/Cô ạ!"*

---

## 4. BỘ CÂU HỎI PHẢN BIỆN & KỊCH BẢN TRẢ LỜI VẤN ĐÁP (Q&A DEFENSE SHEET)

---

### Câu 1: Tại sao gọi là Enhanced LeNet-5? Nó khác gì LeNet-5 gốc 1998?
* **Kịch bản trả lời:**
  > *"Dạ thưa Thầy/Cô, mô hình nguyên bản của Giáo sư Yann LeCun năm 1998 chỉ nhận ảnh xám 1 kênh $28\times 28$, dùng hàm Sigmoid/Tanh và Average Pooling cho tập chữ số MNIST. Nhóm gọi là **Enhanced LeNet-5** vì đã nâng cấp 5 yếu tố kỹ thuật hiện đại:*
  > 1. *Nâng đầu vào lên **ảnh màu RGB $64\times 64$** để giữ màu sắc và chi tiết biển báo.*
  > 2. *Thay Sigmoid bằng **ReLU** để triệt tiêu hiện tượng triệt tiêu đạo hàm (vanishing gradient).*
  > 3. *Thay Average Pooling bằng **Max Pooling ($2\times 2$)** để giữ độ sắc nét của viền biển.*
  > 4. *Thêm **BatchNorm2d** sau mỗi tầng Conv giúp mạng ổn định phân phối và hội tụ nhanh.*
  > 5. *Bổ sung **2 tầng Dropout ($0.3$ và $0.2$)** tại Classifier để chống Overfitting.*
  > *Nhóm giữ tên LeNet-5 vì mô hình vẫn giữ trọn triết lý: mạng tích chập tuần tự phẳng, tinh gọn, không cần các khối skip-connection cồng kềnh."*

---

### Câu 2: Kiến trúc này đã tồn tại chưa hay tự chế? Có copy trên mạng không?
* **Kịch bản trả lời:**
  > *"Dạ thưa Thầy/Cô:*
  > * *Về mặt học thuật: Hướng nghiên cứu hiện đại hóa LeNet (thêm ReLU, BatchNorm, Dropout) cho bài toán Traffic Sign Recognition trên thiết bị nhúng là một **mẫu hình thiết kế đã được kiểm chứng trên các tạp chí và hội nghị quốc tế (IEEE, Springer)**.*
  > * *Tuy nhiên, về mặt kỹ thuật triển khai: **Nhóm hoàn toàn không copy mã nguồn có sẵn**. Cấu hình cụ thể gồm 3 khối Conv ($3 \to 16 \to 32 \to 64$), chuẩn hóa $64\times 64$, tầng ẩn Dense(128) và đầu ra 4 nhóm quy chuẩn là do **nhóm trực tiếp lập trình trên PyTorch và huấn luyện từ đầu (train from scratch)** trên tập dữ liệu của đề tài.*
  > * *Kiến trúc này được 'may đo' vừa vặn cho 4 nhóm biển báo, tránh được hiện tượng thừa tham số (Overparameterization) so với các mạng lớn."*

---

### Câu 3: Sao không dùng ResNet-18 hay MobileNetV3 tiền huấn luyện cho tiện?
* **Kịch bản trả lời:**
  > *"Dạ thưa Thầy/Cô, trong bài báo cáo nhóm có đối chuẩn thực nghiệm trực tiếp với ResNet-18 và MobileNetV3:*
  > * *Về độ chính xác: Enhanced LeNet-5 đạt $98.73\%$, tương đương thậm chí nhỉnh hơn một chút so với ResNet-18 ($98.62\%$). Lý do là bài toán phân loại 4 nhóm từ ảnh crop ROI có không gian đặc trưng khá chuẩn tắc, không cần mô hình 11 triệu tham số của ResNet-18.*
  > * *Về tài nguyên và tốc độ: ResNet-18 nặng tới $11.69\text{M}$ tham số và tốn $1.82\text{ GFLOPs}$, tốc độ trên CPU chỉ được 80 FPS. Trong khi Enhanced LeNet-5 chỉ có **$0.56\text{M}$ tham số (nhẹ hơn 21 lần)**, dung lượng **$2.24\text{ MB}$**, chạy tới **911 FPS (nhanh hơn gấp 11 lần)**.*
  > * *Đối với một hệ thống hỗ trợ lái xe ADAS chạy trên chip nhúng của camera hành trình (vốn không có GPU rời), việc tiết kiệm tài nguyên và độ trễ siêu thấp của Enhanced LeNet-5 mang ý nghĩa thực tiễn vượt trội."*

---

### Câu 4: Tại sao chọn kích thước $64\times 64$ mà không phải $28\times 28$ hay $224\times 224$?
* **Kịch bản trả lời:**
  > *"Dạ thưa Thầy/Cô:*
  > * *Nếu chọn $28\times 28$ (chuẩn MNIST): Quá nhỏ đối với biển báo thực tế. Các chi tiết biểu tượng nội hàm như mũi tên rẽ hay người đi bộ bị nén lại chỉ còn vài pixel nhòe, khiến mô hình rất dễ nhầm lẫn giữa Biển Hiệu Lệnh tròn và Biển Chỉ Dẫn vuông.*
  > * *Nếu chọn $224\times 224$ (chuẩn ImageNet): Khối lượng tính toán FLOPs bùng nổ gấp hơn 12 lần ($> 150\text{ MFLOPs}$), khiến độ trễ trên CPU tăng lên $> 12\text{ ms}$, không đảm bảo được tần số quét cao của hệ thống.*
  > * *Kích thước **$64 \times 64$ chính là 'điểm cân bằng vàng'**: vừa bảo toàn trọn vẹn đặc trưng hoa văn hình học, vừa giữ FLOPs ở mức siêu nhẹ ($13.1\text{ MFLOPs}$) để đạt tốc độ $1.10\text{ ms/ảnh}$."*

---

### Câu 5: Bài toán này có phân loại được biển bị cây che, nhòe mờ do mưa, nhiễu không?
* **Kịch bản trả lời:**
  > *"Dạ thưa Thầy/Cô, hệ thống hoàn toàn có khả năng phân loại tốt các trường hợp biển bị che khuất một phần, nhòe hoặc nhiễu thời tiết nhờ 3 cơ chế:*
  > 1. * **Trong quá trình huấn luyện:** Nhóm đã áp dụng Color Jitter ($\pm 20\%$ độ sáng, tương phản) và xoay nghiêng ngẫu nhiên $\pm 10^\circ$. Đồng thời, tập dữ liệu thực tế trích xuất từ camera hành trình vốn đã chứa sẵn nhiều khung hình trời mưa, ngược sáng và rung lắc.*
  > 2. * **Bản chất của tích chập Conv:** Các bộ lọc học sâu có tính chất bất biến tịnh tiến (translation invariance) và trích xuất đặc trưng phân cấp từ cạnh viền đến hoa văn trung tâm. Do đó, nếu biển bị cây che khuất một góc nhỏ, mạng vẫn tổng hợp được hình dạng tổng thể và màu sắc chủ đạo để phân loại chính xác.*
  > 3. * **Cơ chế 2 giai đoạn:** Nhóm thiết lập ngưỡng tự tin Confidence threshold. Nếu ảnh biển bị nhòe nát đến mức không thể đọc nổi (Confidence $< 0.25$), hệ thống sẽ từ chối đưa ra kết luận sai nhằm đảm bảo an toàn giao thông."*

---

### Câu 6: Tại sao cần 7 độ đo đánh giá? Macro-F1 có ý nghĩa gì hơn Accuracy?
* **Kịch bản trả lời:**
  > *"Dạ thưa Thầy/Cô:*
  > * *Về độ chính xác phân loại: Dữ liệu thực tế bị mất cân bằng mẫu nghiêm trọng (Biển Cấm chiếm số lượng lớn, Biển Chỉ Dẫn chiếm ít). Nếu chỉ nhìn vào **Accuracy**, mô hình chỉ cần dự đoán đúng lớp Cấm là điểm số đã rất cao. Do đó, **Macro-F1** là độ đo bắt buộc vì nó tính trung bình công bằng giữa cả 4 nhóm lớp. Kết hợp **Precision và Recall** giúp nhóm đo lường được cả rủi ro báo động giả (FP) và bỏ sót cảnh báo (FN).*
  > * *Về tính khả thi công nghiệp: Nhóm đo lường thêm **Params, FLOPs, Latency (ms) và Tốc độ (FPS)** để chứng minh mô hình không chỉ chạy được trên lý thuyết mà hoàn toàn khả thi khi nạp vào vi điều khiển/chip nhúng thực tế."*

---

### Câu 7: Tại sao không dùng YOLO phân loại luôn mà phải tách 2 giai đoạn?
* **Kịch bản trả lời:**
  > *"Dạ thưa Thầy/Cô:*
  > * *Trong ảnh camera hành trình, biển báo rất nhỏ (thường chỉ chiếm $< 0.1\%$ diện tích ảnh $1080p$). Nếu bắt YOLO vừa tìm vị trí vừa phân loại ngay 4 nhóm hoặc 43 phân lớp, hàm mất mát sẽ bị cạnh tranh gradient nghiêm trọng giữa nhánh định vị và nhánh phân loại, dẫn đến tỉ lệ sót biển (Miss rate) cao.*
  > * *Bằng cách ép YOLO11s chỉ làm duy nhất một nhiệm vụ là **1-Class Detector ("sign")**, YOLO đạt Recall cực cao lên tới **85.61\%**.*
  > * *Sau đó, ảnh con được cắt ra và nạp vào Enhanced LeNet-5 để phân loại chuyên biệt. Mô hình phân loại lúc này nhận đầu vào là ảnh vuông $64\times 64$, tập trung $100\%$ dung lượng tham số để phân biệt hoa văn và màu sắc, mang lại độ chính xác vượt trội $98.73\%$."*

---

### Câu 8: 12 ca nhầm lẫn ở Level B là gì? Tại sao lại nhầm?
* **Kịch bản trả lời:**
  > *"Dạ thưa Thầy/Cô, qua phân tích ma trận nhầm lẫn ở Level B trên 1.813 mẫu thử nghiệm:*
  > * *Nhóm ghi nhận có 12 ca dự đoán sai, cụ thể là **11 ca nhầm từ Hiệu lệnh sang Chỉ dẫn**, và **1 ca nhầm từ Chỉ dẫn sang Hiệu lệnh**.*
  > * *Lý do: Cả Biển Hiệu lệnh (Nhóm R) và Biển Chỉ dẫn (Nhóm I) đều có tông màu chủ đạo là **xanh lam**. Sự khác biệt cốt lõi là hình học: Hiệu lệnh hình tròn, Chỉ dẫn hình vuông.*
  > * *Khi xe ở cự ly xa, ảnh crop chỉ đạt kích thước nhỏ ($< 25\text{ px}$) hoặc bị góc chụp nghiêng của camera, hình tròn bị biến dạng quang học thành hình bầu dục/hình chữ nhật, khiến lớp tích chập bị nhầm lẫn giữa viền tròn và viền vuông.*
  > * *Tuy nhiên, giữa 2 nhóm quan trọng nhất là **Cấm (đỏ) và Nguy hiểm (vàng)**, mô hình đạt độ chính xác $100\%$, không có ca nhầm lẫn nào."*
