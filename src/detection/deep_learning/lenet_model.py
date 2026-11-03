"""
================================================================================
MẠNG NƠ-RON TÍCH CHẬP LENET-5 CẢI TIẾN (ENHANCED LENET-5 FOR TRAFFIC SIGNS)
Phục vụ đề tài Thị giác máy tính - Thành viên 2 (Deep Learning)
================================================================================
So sánh với kiến trúc LeNet-5 cổ điển (LeCun et al., 1998):
1. [Kênh màu]: Nâng cấp từ ảnh xám 1 kênh (Grayscale) lên ảnh màu RGB 3 kênh.
2. [Hàm kích hoạt]: Thay thế hàm Sigmoid/Tanh bằng ReLU (Rectified Linear Unit)
   giúp tránh hiện tượng triệt tiêu gradient (Vanishing Gradient) và hội tụ nhanh hơn.
3. [Chuẩn hóa tầng]: Bổ sung Batch Normalization (BatchNorm2d) sau mỗi tầng tích chập
   để ổn định phân phối dữ liệu qua từng batch (Internal Covariate Shift).
4. [Pooling]: Thay thế Average Pooling bằng Max Pooling 2x2 để giữ lại các đặc trưng
   biên cạnh sắc nét nhất của biển báo.
5. [Tầng tích chập 3]: Thêm 1 khối Conv 64 filters để trích xuất đặc trưng sâu hơn.
6. [Chống học vẹt]: Thêm 2 tầng Dropout (0.3 và 0.2) ở khối Fully Connected
   để chống Overfitting do tập dữ liệu có sự chênh lệch số lượng mẫu giữa các lớp.
================================================================================
"""

import torch
import torch.nn as nn


class EnhancedLeNet(nn.Module):
    """
    Kiến trúc LeNet-5 cải tiến chuẩn hóa cho bài toán phân loại 4 nhóm biển báo.
    Đầu vào: Ảnh RGB kích thước (3, 64, 64)
    Đầu ra: 4 giá trị Logits tương ứng 4 nhóm (Bien_Cam, Bien_Nguy_Hiem, Bien_Hieu_Lenh, Bien_Chi_Dan)
    """

    def __init__(self, num_classes: int = 4):
        super(EnhancedLeNet, self).__init__()

        # --- KHỐI TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTOR) ---
        # Khối 1: (3, 64, 64) -> Conv 5x5 -> (16, 64, 64) -> MaxPool -> (16, 32, 32)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Khối 2: (16, 32, 32) -> Conv 5x5 -> (32, 32, 32) -> MaxPool -> (32, 16, 16)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Khối 3: (32, 16, 16) -> Conv 3x3 -> (64, 16, 16) -> MaxPool -> (64, 8, 8)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- KHỐI PHÂN LOẠI (CLASSIFIER HEAD) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),                       # Chống Overfitting
            nn.Linear(in_features=64 * 8 * 8, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        logits = self.classifier(x)
        return logits
