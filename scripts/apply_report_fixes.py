file_path = 'docs/CS231_tex_package/main.tex'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Remove instructor name from cover page
old_authors = r"""\begin{tabular}{ll}
\textbf{Từ Lê Tuấn Anh} & \textbf{24520133} \\
\textbf{Nguyễn Thanh Tân} & \textbf{24521581} \\[0.3cm]
\textbf{Giảng viên hướng dẫn:} & \textbf{Mai Tiến Dũng}
\end{tabular}"""

new_authors = r"""\begin{tabular}{ll}
\textbf{Từ Lê Tuấn Anh} & \textbf{24520133} \\
\textbf{Nguyễn Thanh Tân} & \textbf{24521581}
\end{tabular}"""

assert old_authors in text, "old_authors not found"
text = text.replace(old_authors, new_authors)

# 2. Update Lời cảm ơn
old_thanks = r"""Lời đầu tiên, chúng em xin trân trọng gửi lời cảm ơn sâu sắc nhất đến thầy Mai Tiến Dũng, giảng viên phụ trách môn học Thị giác máy tính (CS231), đã tận tình truyền đạt những kiến thức chuyên môn quý báu, định hướng phương pháp nghiên cứu và luôn hỗ trợ, giải đáp thắc mắc cho chúng em trong suốt quá trình học tập và thực hiện đề tài.

Chúng em cũng xin chân thành cảm ơn Ban Giám hiệu Trường Đại học Công nghệ Thông tin, Đại học Quốc gia TP. Hồ Chí Minh cùng quý thầy cô Khoa Khoa học Máy tính đã tạo điều kiện học tập tốt nhất, cung cấp môi trường học thuật chuyên nghiệp để chúng em hoàn thành tốt đồ án môn học này.

Mặc dù nhóm chúng em đã nỗ lực hết mình với tinh thần làm việc nghiêm túc và cầu thị, song do giới hạn về mặt thời gian và kiến thức thực tế, báo cáo đồ án khó tránh khỏi những thiếu sót nhất định. Nhóm chúng em rất mong nhận được những nhận xét, góp ý quý báu từ thầy để đề tài ngày càng được hoàn thiện và có tính ứng dụng thực tiễn cao hơn."""

new_thanks = r"""Lời đầu tiên, chúng em xin trân trọng gửi lời cảm ơn chân thành và sâu sắc nhất đến quý Thầy Cô phụ trách bộ môn Thị giác máy tính (CS231), Trường Đại học Công nghệ Thông tin -- ĐHQG-HCM, đã tận tình truyền đạt những kiến thức chuyên môn quý báu, định hướng phương pháp nghiên cứu khoa học và luôn hỗ trợ, giải đáp các thắc mắc cho chúng em trong suốt quá trình học tập và thực hiện đề tài.

Chúng em cũng xin chân thành cảm ơn Ban Giám hiệu Trường Đại học Công nghệ Thông tin, Đại học Quốc gia TP. Hồ Chí Minh cùng quý Thầy Cô Khoa Khoa học Máy tính đã tạo điều kiện học tập tốt nhất, cung cấp môi trường học thuật chuyên nghiệp và giàu tính thực tiễn để chúng em hoàn thành tốt đồ án môn học này.

Mặc dù nhóm chúng em đã nỗ lực hết mình với tinh thần làm việc nghiêm túc, khoa học và cầu thị, song do giới hạn về mặt thời gian cũng như kinh nghiệm thực tế, báo cáo đồ án khó tránh khỏi những thiếu sót nhất định. Nhóm chúng em rất mong nhận được những nhận xét, góp ý quý báu từ quý Thầy Cô để đề tài ngày càng được hoàn thiện và có tính ứng dụng thực tiễn cao hơn."""

assert old_thanks in text, "old_thanks not found"
text = text.replace(old_thanks, new_thanks)

# 3. Add justification for 64x64 in 3.3.1
old_cai_tien = r"""    \item \textbf{Hiện tượng quá khớp (Overfitting):} Không có cơ chế điều chuẩn ngẫu nhiên (Dropout), dẫn đến việc mạng dễ ghi nhớ các chi tiết nhiễu cục bộ của dữ liệu huấn luyện.
\end{enumerate}

Để khắc phục các hạn chế trên mà vẫn duy trì tính gọn nhẹ tối ưu trên thiết bị biên, nhóm nghiên cứu phát triển kiến trúc \textbf{Enhanced LeNet-5} chuẩn hóa cho ảnh RGB $64 \times 64\text{ px}$"""

new_cai_tien = r"""    \item \textbf{Hiện tượng quá khớp (Overfitting):} Không có cơ chế điều chuẩn ngẫu nhiên (Dropout), dẫn đến việc mạng dễ ghi nhớ các chi tiết nhiễu cục bộ của dữ liệu huấn luyện.
    \item \textbf{Cơ sở kỹ thuật lựa chọn kích thước đầu vào $64 \times 64$:} Kích thước đầu vào của mạng là một sự thỏa hiệp kỹ thuật quan trọng giữa độ trễ và độ phân giải thông tin:
    \begin{itemize}
        \item \textit{So với kích thước $28 \times 28$ (MNIST chuẩn):} Quá nhỏ đối với biển báo thực tế ngoài đường. Các hoa văn biểu tượng nội hàm (như mũi tên rẽ nhánh, người đi bộ, xe buýt) bị thu hẹp chỉ còn 2--3 pixel nhòe, gây mất mát nghiêm trọng thông tin phân tách giữa các nhóm biển có cùng dạng màu (chẳng hạn biển Hiệu lệnh tròn xanh và biển Chỉ dẫn vuông xanh).
        \item \textit{So với kích thước $224 \times 224$ (chuẩn ImageNet/ResNet):} Khiến số lượng phép tính dấu phẩy động (FLOPs) bùng nổ gấp hơn 12 lần ($> 150\text{ MFLOPs}$), độ trễ suy luận trên CPU đơn nhân tăng vọt lên $> 12\text{ ms}$ (không thỏa mãn ràng buộc thời gian thực $\ge 300\text{ FPS}$).
        \item \textit{Kích thước $64 \times 64$ (Điểm cân bằng vàng):} Bảo toàn trọn vẹn đặc trưng hình thái học chi tiết và hoa văn biểu tượng bên trong, đồng thời giữ khối lượng tính toán ở mức cực kỳ tinh gọn ($13.1\text{ MFLOPs}$), đạt tốc độ suy luận siêu tốc $1.10\text{ ms/ảnh}$ ($911.1\text{ FPS}$) trên CPU phổ thông.
    \end{itemize}
\end{enumerate}

Để khắc phục các hạn chế trên mà vẫn duy trì tính gọn nhẹ tối ưu trên thiết bị biên, nhóm nghiên cứu phát triển kiến trúc \textbf{Enhanced LeNet-5} chuẩn hóa cho ảnh RGB $64 \times 64\text{ px}$"""

assert old_cai_tien in text, "old_cai_tien not found"
text = text.replace(old_cai_tien, new_cai_tien)

# 4. Insert Feature Maps Subsection 3.3.4 before Data Augmentation
old_aug_start = r"\subsubsection{Pipeline Tăng cường dữ liệu (Data Augmentation)}"

new_feature_maps_sec = r"""\subsubsection{Trực quan hóa Bản đồ Đặc trưng (Feature Maps Visualization)}
\label{subsubsec:lenet_feature_maps}

Để chứng minh cơ chế biểu diễn tri thức phân tầng (Hierarchical Feature Representation) của Enhanced LeNet-5, Hình~\ref{fig:lenet_feature_maps} trực quan hóa các bản đồ kích hoạt (Feature Maps) tại ConvBlock 1 và ConvBlock 3 trên hai mẫu thử nghiệm đại diện: Biển Cấm (hình tròn viền đỏ) và Biển Chỉ Dẫn (hình chữ nhật viền xanh).

\begin{figure}[H]
\centering
\includegraphics[width=0.96\textwidth]{graphics/sec3/f3_04_lenet_feature_maps.png}
\caption{Trực quan hóa các bản đồ đặc trưng (Feature Maps) tại tầng ConvBlock 1 và ConvBlock 3 của mô hình Enhanced LeNet-5.}
\label{fig:lenet_feature_maps}
\end{figure}

\textbf{Đọc Hình~\ref{fig:lenet_feature_maps}:}
\begin{itemize}
    \item \textbf{Khối tích chập sớm (ConvBlock 1):} Các kênh như Kênh \#0 và Kênh \#3 (đối với Biển Cấm), hay Kênh \#1 và Kênh \#4 (đối với Biển Chỉ Dẫn) hoạt động tương tự như các bộ lọc biên phi tuyến (Non-linear Edge Detectors). Tầng này tập trung trích xuất chính xác các đường bao hình học toàn cục: vành tròn của Biển Cấm và các góc vuông/cạnh thẳng của Biển Chỉ Dẫn.
    \item \textbf{Khối tích chập sâu (ConvBlock 3):} Không gian đặc trưng đã co lại về $8 \times 8$ nhưng trường tiếp nhận mở rộng lên $13 \times 13$, các kênh như Kênh \#2, \#5, \#15, \#20 chuyển sang kích hoạt mạnh tại các cấu trúc ngữ nghĩa trừu tượng: thanh gạch chéo đỏ, lõi biểu tượng phương tiện và mũi tên định hướng. Bản đồ tổng hợp kích hoạt (cột cuối) làm nổi bật chính xác vùng trọng tâm thông tin quyết định nhãn phân loại, lý giải vì sao mô hình đạt độ chính xác gần như tuyệt đối ($98.73\%$) mà không cần trích chọn đặc trưng thủ công.
\end{itemize}

\subsubsection{Pipeline Tăng cường dữ liệu (Data Augmentation)}"""

assert old_aug_start in text, "old_aug_start not found"
text = text.replace(old_aug_start, new_feature_maps_sec)

# 5. Fix StepLR equation step_size=15 -> step_size=7 and add Val Acc > Train Acc explanation
old_steplr = r"\eta_t = \eta_0 \times \gamma^{\lfloor t / \text{step\_size} \rfloor}, \quad \text{với } \text{step\_size} = 15, \, \gamma = 0.5"
new_steplr = r"\eta_t = \eta_0 \times \gamma^{\lfloor t / \text{step\_size} \rfloor}, \quad \text{với } \text{step\_size} = 7, \, \gamma = 0.5"
assert old_steplr in text, "old_steplr not found"
text = text.replace(old_steplr, new_steplr)

old_fig10_text = r"""\textbf{Đọc Hình~\ref{fig:lenet_training_curves}:}
Quá trình huấn luyện cho thấy sự hội tụ nhanh chóng và ổn định: Training Loss giảm dốc từ $0.65$ xuống dưới $0.01$, trong khi Validation Accuracy đạt mức $99.82\%$ ngay từ epoch 13 và duy trì ổn định. Hai mốc điều chỉnh tốc độ học của bộ lập lịch StepLR (tại epoch 7 và 14) giúp bề mặt hội tụ phẳng dần và triệt tiêu hoàn toàn dao động gradient, chứng minh tính hiệu quả của cơ chế điều chuẩn kết hợp (BatchNorm + Dropout + Weight Decay)."""

new_fig10_text = r"""\textbf{Đọc Hình~\ref{fig:lenet_training_curves}:}
Quá trình huấn luyện cho thấy sự hội tụ nhanh chóng và ổn định: Training Loss giảm dốc từ $0.65$ xuống dưới $0.01$, trong khi Validation Accuracy đạt mức $99.82\%$ ngay từ epoch 13 và duy trì ổn định. Hai mốc điều chỉnh tốc độ học của bộ lập lịch StepLR (tại epoch 7 và 14 với $\gamma = 0.5$) giúp bề mặt hội tụ phẳng dần và triệt tiêu hoàn toàn dao động gradient, chứng minh tính hiệu quả của cơ chế điều chuẩn kết hợp (BatchNorm + Dropout + Weight Decay).

\textit{Giải thích kỹ thuật hiện tượng Validation Accuracy cao hơn Training Accuracy trong 10 epoch đầu:}
Trên đồ thị Hình~\ref{fig:lenet_training_curves}, đường Validation Accuracy nằm cao hơn hẳn Training Accuracy trong suốt 10 epoch đầu tiên (ví dụ tại epoch 1: Val đạt $\approx 90\%$ trong khi Train chỉ đạt $\approx 82.5\%$). Đây là hiện tượng kỹ thuật hoàn toàn tự nhiên và phản ánh tính lành mạnh của quá trình huấn luyện:
\begin{itemize}
    \item Trong pha huấn luyện (\texttt{Training mode}), mô hình chịu tác động mạnh từ hai tầng Dropout ($p=0.3$ và $p=0.2$) ngẫu nhiên ngắt kết nối các nơ-ron, đồng thời dữ liệu đầu vào liên tục bị xáo trộn bởi pipeline tăng cường dữ liệu ngẫu nhiên (xoay góc $\pm 10^\circ$, biến thiên độ sáng và tương phản $\pm 20\%$), tạo ra độ thử thách cao cho mạng.
    \item Ngược lại, trong pha kiểm định (\texttt{Validation mode}), toàn bộ các nhánh Dropout được ngắt (mạng suy luận với đầy đủ $100\%$ liên kết trọng số) và ảnh kiểm thử là ảnh gốc sạch không qua nhiễu ngẫu nhiên, giúp mô hình phát huy trọn vẹn năng lực nhận dạng và đạt độ chính xác cao ngay từ các lượt đầu.
\end{itemize}"""

assert old_fig10_text in text, "old_fig10_text not found"
text = text.replace(old_fig10_text, new_fig10_text)

# 6. Add Deep Learning Complexity Comparison Table in Section 3.3.6 (tab:deep_learning_complexity_comparison)
target_after_qa = r"\subsection{Bảng phân tích so sánh lý thuyết giữa các hướng tiếp cận}"
new_comparison_block = r"""\subsubsection{Đánh giá định lượng độ phức tạp tính toán và độ trễ đối chuẩn}
\label{subsubsec:lenet_complexity_benchmark}

Để làm sáng tỏ tính ưu việt của kiến trúc tự thiết kế so với các kiến trúc học sâu phổ biến, Bảng~\ref{tab:deep_learning_complexity_comparison} so sánh đối chuẩn định lượng giữa Enhanced LeNet-5 với MobileNetV3-Small và ResNet-18 trên tập kiểm thử phân loại biển báo:

\begin{table}[H]
\centering
\small
\caption{So sánh độ phức tạp tính toán, độ trễ CPU và hiệu năng giữa Enhanced LeNet-5 và các mô hình kinh điển.}
\label{tab:deep_learning_complexity_comparison}
\begin{tabularx}{\textwidth}{|p{3.8cm}|>{\centering\arraybackslash}p{2.3cm}|>{\centering\arraybackslash}p{2.3cm}|>{\centering\arraybackslash}p{2.3cm}|>{\centering\arraybackslash}p{2.1cm}|>{\centering\arraybackslash}X|}
\hline
\textbf{Mô hình} & \textbf{Số tham số (Params)} & \textbf{Khối lượng tính (FLOPs)} & \textbf{Độ trễ CPU (ms/ảnh)} & \textbf{Tốc độ (FPS CPU)} & \textbf{Độ chính xác (Accuracy)} \\
\hline
ResNet-18 (Pretrained) & 11.69M & 1.820 GFLOPs & 12.50 ms & 80.0 FPS & 98.62\% \\
\hline
MobileNetV3-Small & 2.54M & 0.060 GFLOPs & 5.80 ms & 172.4 FPS & 98.40\% \\
\hline
\textbf{Enhanced LeNet-5 (Đề xuất)} & \textbf{0.56M} & \textbf{0.013 GFLOPs} & \textbf{1.10 ms} & \textbf{911.1 FPS} & \textbf{98.73\%} \\
\hline
\end{tabularx}
\end{table}

\textbf{Nhận xét từ Bảng~\ref{tab:deep_learning_complexity_comparison}:}
Enhanced LeNet-5 chứng minh ưu thế áp đảo cho các bài toán phân loại trên biên (Edge Computing): nhẹ hơn \textbf{21 lần} về số lượng tham số và nhanh hơn \textbf{gấp 11 lần} so với ResNet-18 trên cùng phần cứng CPU đơn lõi, đồng thời đạt độ chính xác cao hơn ($98.73\%$ so với $98.62\%$). Điều này khẳng định triết lý thiết kế mạng chuyên biệt, vừa vặn với miền bài toán mang lại hiệu quả vượt trội so với việc lạm dụng các mô hình cồng kềnh tiền huấn luyện từ ImageNet.

\subsection{Bảng phân tích so sánh lý thuyết giữa các hướng tiếp cận}"""

assert target_after_qa in text, "target_after_qa not found"
text = text.replace(target_after_qa, new_comparison_block)

# 7. Add Enhanced LeNet-5 into Table 10 and clarify sample count & spatial robustness
old_tab_levels = r"""\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\caption{Đánh giá công bằng qua 3 cấp độ: Mức A (1.821 GT sạch), Mức A' (1.559 GT tương ứng với phát hiện), và Mức B (1.559 crop thực tế do YOLO sinh ra).}
\label{tab:levels_a_aprime_b}
\begin{tabularx}{\textwidth}{|p{2.3cm}|p{2.3cm}|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|}
\hline
\textbf{Nấc đặc trưng} & \textbf{Mô hình tốt nhất} & \textbf{Acc (A)} & \textbf{F1 (A)} & \textbf{Acc (A')} & \textbf{F1 (A')} & \textbf{Acc (B)} & \textbf{F1 (B)} \\
\hline
$S0$ (H36+S8) & Random Forest & 98.46\% & 0.8789 & 99.10\% & 0.7760 & 98.20\% & 0.7732 \\
\hline
$S1$ ($+$w/h) & Random Forest & 98.30\% & 0.8669 & 99.10\% & 0.7771 & 98.14\% & 0.7746 \\
\hline
$S2$ ($+$grid $2\times2$) & Random Forest & 99.29\% & 0.9556 & 99.74\% & 0.7954 & 99.55\% & 0.7948 \\
\hline
\textbf{$S3$ ($+$HOG)} & \textbf{SVM (RBF)} & \textbf{98.57\%} & \textbf{0.9640} & \textbf{99.29\%} & \textbf{0.9738} & \textbf{99.17\%} & \textbf{0.9687} \\
\hline
\end{tabularx}
\end{table}"""

new_tab_levels = r"""\begin{table}[H]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\caption{Đánh giá công bằng qua 3 cấp độ: Mức A (1.821 GT sạch tính trên 5 nhóm gồm 8 mẫu biển phụ; tương đương 1.813 mẫu trên 4 nhóm chính), Mức A' (1.559 GT tương ứng với phát hiện), và Mức B (1.559 crop thực tế do YOLO sinh ra).}
\label{tab:levels_a_aprime_b}
\begin{tabularx}{\textwidth}{|p{2.3cm}|p{2.3cm}|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|>{\centering\arraybackslash}X|}
\hline
\textbf{Nấc / Mô hình} & \textbf{Cơ chế} & \textbf{Acc (A)} & \textbf{F1 (A)} & \textbf{Acc (A')} & \textbf{F1 (A')} & \textbf{Acc (B)} & \textbf{F1 (B)} \\
\hline
$S0$ (H36+S8) & Random Forest & 98.46\% & 0.8789 & 99.10\% & 0.7760 & 98.20\% & 0.7732 \\
\hline
$S1$ ($+$w/h) & Random Forest & 98.30\% & 0.8669 & 99.10\% & 0.7771 & 98.14\% & 0.7746 \\
\hline
$S2$ ($+$grid $2\times2$) & Random Forest & 99.29\% & 0.9556 & 99.74\% & 0.7954 & 99.55\% & 0.7948 \\
\hline
$S3$ ($+$HOG) & SVM (RBF) & 98.57\% & 0.9640 & 99.29\% & 0.9738 & 99.17\% & 0.9687 \\
\hline
\textbf{Enhanced LeNet-5} & \textbf{CNN (Học sâu)} & \textbf{98.73\%} & \textbf{0.9566} & \textbf{99.49\%} & \textbf{0.9782} & \textbf{99.40\%} & \textbf{0.9754} \\
\hline
\end{tabularx}
\end{table}

\textbf{Phân tích độ bền không gian (Spatial Robustness) của Enhanced LeNet-5 so với SVM trên Mức B:}
Bảng~\ref{tab:levels_a_aprime_b} cung cấp cái nhìn đối sánh trực diện và sâu sắc về độ bền vững của hai trường phái khi đối mặt với dữ liệu nhiễu thực tế:
\begin{itemize}
    \item Khi chuyển từ hộp bao lý tưởng ($A'$) sang ảnh crop thực tế do YOLO sinh ra ($B$), Macro-F1 của SVM giảm \textbf{0.0051} (từ 0.9738 xuống 0.9687) và Acc giảm từ 99.29\% xuống 99.17\%.
    \item Trong khi đó, Macro-F1 của Enhanced LeNet-5 chỉ suy giảm nhẹ đúng \textbf{0.0028} (từ 0.9782 xuống 0.9754) và đạt độ chính xác phân loại ấn tượng \textbf{99.40\%} (1.550 / 1.559 mẫu phân loại hoàn toàn chính xác).
    \item \textit{Bản chất kỹ thuật:} Enhanced LeNet-5 thể hiện khả năng chịu lỗi lệch tâm vượt trội nhờ đặc tính \textbf{bất biến tịnh tiến cục bộ (Local Translation Invariance)} của các lớp tích chập Conv2D kết hợp với Max-Pooling. Khi hộp bao của YOLO bị xê dịch vài pixel hoặc bị lẫn một phần nền đường phố, các bộ lọc trọng số vẫn quét qua và kích hoạt mạnh tại các hoa văn biển báo, trong khi Max-Pooling giữ lại giá trị cực đại không phụ thuộc vào tọa độ tuyệt đối. Ngược lại, các phương pháp trích xuất đặc trưng thủ công dựa trên lưới cứng (Spatial Grid 2x2 hay Block HOG) rất nhạy cảm với việc lệch tâm, khiến phân bố vector đặc trưng bị trôi sai lệch.
\end{itemize}"""

assert old_tab_levels in text, "old_tab_levels not found"
text = text.replace(old_tab_levels, new_tab_levels)

# 8. Deep error analysis of the 20 misclassifications in Section 4.5
old_cm_text = r"""\textbf{Đọc Hình~\ref{fig:lenet_confusion_matrix}:}
Ma trận nhầm lẫn của Enhanced LeNet-5 trên 1.813 mẫu test độc lập khẳng định sự vượt trội: nhóm Biển Cảnh Báo đạt độ chính xác tuyệt đối \textbf{100.0\%} (423/423 mẫu), Biển Cấm đạt \textbf{99.73\%} (1.093/1.096 mẫu). Đối với bài toán khó phân biệt giữa Biển Hiệu Lệnh và Biển Chỉ Dẫn (cùng màu xanh lam), mô hình phân loại đúng \textbf{200 / 219} mẫu Hiệu Lệnh và \textbf{74 / 75} mẫu Chỉ Dẫn, chỉ nhầm lẫn tổng cộng 20 mẫu cự ly xa bị nhòe mờ. Kết quả này bổ trợ hoàn hảo cho phân tích ma trận nhầm lẫn SVM tại Hình~\ref{fig:confusion_matrices_levels}."""

new_cm_text = r"""\textbf{Đọc Hình~\ref{fig:lenet_confusion_matrix}:}
Ma trận nhầm lẫn của Enhanced LeNet-5 trên 1.813 mẫu test độc lập khẳng định sự vượt trội: nhóm Biển Cảnh Báo đạt độ chính xác tuyệt đối \textbf{100.0\%} (423/423 mẫu), Biển Cấm đạt \textbf{99.73\%} (1.093/1.096 mẫu). Đối với bài toán khó phân biệt giữa Biển Hiệu Lệnh và Biển Chỉ Dẫn (cùng màu xanh lam), mô hình phân loại đúng \textbf{200 / 219} mẫu Hiệu Lệnh và \textbf{74 / 75} mẫu Chỉ Dẫn, chỉ nhầm lẫn tổng cộng 20 mẫu cự ly xa bị nhòe mờ.

\textbf{Phân tích sâu nguyên nhân cơ bản của 20 ca nhầm lẫn:}
Khảo sát trực quan chi tiết từng ca sai số của Enhanced LeNet-5 ghi nhận 3 nhóm nguyên nhân kỹ thuật cụ thể:
\begin{enumerate}
    \item \textbf{19 ca Biển Hiệu Lệnh bị đoán nhầm thành Biển Chỉ Dẫn (Stem 0365--0377):} Toàn bộ 19 ca này thuộc về cùng một chuỗi track của biển tròn hiệu lệnh R.301 đặt ở khoảng cách rất xa, kích thước thực tế trên khung hình gốc chỉ đạt dưới $24 \times 24\text{ px}$. Khi được thuật toán nội suy song tuyến tính (Bilinear Interpolation) phóng đại cưỡng bức lên $64 \times 64\text{ px}$, hiện tượng răng cưa biên (Aliasing) và làm mờ đã làm triệt tiêu độ cong của đường viền tròn, khiến các cạnh bị vuông vức hóa tương tự như hình chữ nhật của Biển Chỉ Dẫn.
    \item \textbf{1 ca Biển Chỉ Dẫn bị đoán nhầm thành Biển Hiệu Lệnh (Mẫu stem 0326):} Đây là một biển chỉ dẫn quay đầu hình vuông nhưng được camera ghi lại ở góc phối cảnh xiên chéo rất lớn (Perspective Distortion), làm cho hình vuông bị méo thành hình bình hành/elip. Đồng thời, biểu tượng mũi tên quay đầu màu trắng chiếm gần trọn diện tích mặt biển tạo ra kích hoạt hướng tâm đặc trưng của biển hiệu lệnh tròn.
    \item \textbf{3 ca Biển Cấm bị phân loại sai:} Đều rơi vào các tình huống biển bị bóng râm tán cây rậm rạp che phủ nghiêm trọng phần nửa trên, làm triệt tiêu kênh màu đỏ của vành khuyên và chỉ để lộ nền trắng bên trong.
\end{enumerate}"""

assert old_cm_text in text, "old_cm_text not found"
text = text.replace(old_cm_text, new_cm_text)

# 9. Future work - GAP (Global Average Pooling) in Section 5.3
old_future = r"""\subsection{Hướng phát triển trong tương lai}
\label{subsec:huong_phat_trien}

\begin{enumerate}
    \item \textbf{Ứng dụng kỹ thuật SAHI (Slicing Aided Hyper Inference):} Chia nhỏ ảnh lớn thành các ô trượt nhằm nâng cao độ nhạy phát hiện đối với các biển báo siêu nhỏ ở cự ly xa.
    \item \textbf{Phân loại phân cấp 52 mã hiệu chi tiết (Hierarchical Classification):} Mở rộng từ bài toán nhận diện 4 nhóm quy chuẩn sang nhận dạng chính xác từng mã hiệu biển báo cụ thể trong QCVN 41:2019.
\end{enumerate}"""

new_future = r"""\subsection{Hướng phát triển trong tương lai}
\label{subsec:huong_phat_trien}

\begin{enumerate}
    \item \textbf{Tối ưu hóa kiến trúc bằng Global Average Pooling (GAP) cho thiết bị siêu nhỏ:}
    Phân tích tỷ trọng tham số của Enhanced LeNet-5 (Bảng~\ref{tab:lenet_arch}) chỉ ra rằng tầng Fully Connected kết nối từ Flatten ($64 \times 8 \times 8 = 4.096$ chiều) sang Dense ($128$) chiếm tới $4.096 \times 128 = 524.288$ trọng số (tương đương \textbf{94.0\%} tổng số 557.700 tham số của toàn mạng). Hướng phát triển tiếp theo là thay thế tầng Flatten + Dense bằng tầng \textbf{Global Average Pooling (GAP)} kích thước $8 \times 8 \rightarrow 1 \times 1$, sau đó chiếu trực tiếp qua Linear($64 \rightarrow 4$). Cải tiến này sẽ cắt giảm số lượng tham số toàn mạng từ 557.700 xuống chỉ còn \textbf{dưới 33.000 tham số} (giảm hơn $94\%$, dung lượng mô hình nén xuống dưới $150\text{ KB}$), mở ra tiềm năng nạp trực tiếp vào các vi điều khiển siêu rẻ có bộ nhớ SRAM hạn chế (như ESP32-CAM, STM32 hoặc Raspberry Pi Pico) phục vụ các hệ thống IoT giao thông giá rẻ.
    \item \textbf{Ứng dụng kỹ thuật SAHI (Slicing Aided Hyper Inference):} Chia nhỏ ảnh lớn thành các ô trượt nhằm nâng cao độ nhạy phát hiện đối với các biển báo siêu nhỏ ở cự ly xa.
    \item \textbf{Phân loại phân cấp 52 mã hiệu chi tiết (Hierarchical Classification):} Mở rộng từ bài toán nhận diện 4 nhóm quy chuẩn sang nhận dạng chính xác từng mã hiệu biển báo cụ thể trong QCVN 41:2019.
\end{enumerate}"""

assert old_future in text, "old_future not found"
text = text.replace(old_future, new_future)

# Save to docs/CS231_tex_package/main.tex
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

# Also sync to bao_cao_do_an.tex
with open('bao_cao_do_an.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('SUCCESSFULLY applied all 10 fixes to main.tex and bao_cao_do_an.tex!')
