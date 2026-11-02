"""
================================================================================
SCRIPT HUẤN LUYỆN MÔ HÌNH CNN (LỰA CHỌN A CHO THÀNH VIÊN 2)
Huấn luyện mạng nơ-ron tích chập (Custom CNN / LeNet) trên tập ROI biển báo cắt sẵn.
Dữ liệu sử dụng: data/cropped/train và data/cropped/val (hoặc train_ids.txt, val_ids.txt)
Trọng số lưu tại: checkpoints/deep_learning/best_cnn.pth
================================================================================
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.config import (
    CROPPED_DATA_DIR,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_FILE,
    CHECKPOINTS_DIR,
    TARGET_CLASSES,
    CLASS_TO_IDX,
    IMAGE_SIZE,
    RANDOM_SEED,
)
from src.utils import ensure_dir


# 1. Định nghĩa Dataset tuân thủ Global Data Split
class TrafficSignDataset(Dataset):
    def __init__(self, split_file: Path, transform=None):
        self.samples = []
        self.transform = transform

        if not split_file.exists():
            raise FileNotFoundError(f"[-] Không tìm thấy file split: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rel_path, target_class = parts[0], parts[1]
                    img_path = CROPPED_DATA_DIR / rel_path
                    if img_path.exists():
                        self.samples.append((img_path, CLASS_TO_IDX[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# 2. Định nghĩa kiến trúc mạng CNN mẫu
class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(TrafficSignCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_cnn(epochs=20, batch_size=32, lr=1e-3):
    print("=" * 65)
    print("🚀 [THÀNH VIÊN 2] BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH CNN PHÂN LOẠI BIỂN BÁO")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị huấn luyện: {device}")

    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = TrafficSignDataset(TRAIN_SPLIT_FILE, transform=train_transforms)
    val_dataset = TrafficSignDataset(VAL_SPLIT_FILE, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[*] Số mẫu Train : {len(train_dataset)}")
    print(f"[*] Số mẫu Val   : {len(val_dataset)}")

    model = TrafficSignCNN(num_classes=len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    save_dir = ensure_dir(CHECKPOINTS_DIR / "deep_learning")
    best_val_acc = 0.0
    best_ckpt_path = save_dir / "best_cnn.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Đánh giá trên tập Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, best_ckpt_path)
            print(f"   -> [BEST] Đã lưu checkpoint mới tốt nhất: {best_ckpt_path} (Acc: {val_acc*100:.2f}%)")

    print(f"\n[+] Huấn luyện CNN hoàn tất! Trọng số lưu tại: {best_ckpt_path}")


if __name__ == "__main__":
    train_cnn()
