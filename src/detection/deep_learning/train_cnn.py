"""
================================================================================
SCRIPT HUẤN LUYỆN MÔ HÌNH LENET-5 CẢI TIẾN (THÀNH VIÊN 2)
Huấn luyện mạng nơ-ron tích chập (Enhanced LeNet) trên tập ROI biển báo cắt sẵn.
Dữ liệu sử dụng: data/cropped/train và data/cropped/val (tuân thủ Global Split).
Trọng số tốt nhất được lưu tại: checkpoints/deep_learning/best_cnn.pth
================================================================================
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.detection.deep_learning.lenet_model import EnhancedLeNet
from src.utils import ensure_dir


class TrafficSignDataset(Dataset):
    """Dataset nạp ảnh ROI tuân thủ tập Global Data Split."""

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


def train_cnn(epochs: int = 15, batch_size: int = 64, lr: float = 1e-3):
    print("=" * 65)
    print("🚀 [THÀNH VIÊN 2] BẮT ĐẦU HUẤN LUYỆN MẠNG LENET-5 CẢI TIẾN")
    print("=" * 65)

    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị tính toán (Device) : {device}")
    print(f"[*] Kích thước ảnh chuẩn hóa    : {IMAGE_SIZE}")
    print(f"[*] Số lượt huấn luyện (Epochs) : {epochs}")
    print(f"[*] Kích thước Batch Size       : {batch_size}")
    print(f"[*] Tốc độ học (Learning Rate)  : {lr}")

    # Data Augmentation chống Overfitting và mô phỏng điều kiện xấu
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomRotation(degrees=10),
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

    print(f"[*] Mẫu tập Huấn luyện (Train) : {len(train_dataset)} ảnh")
    print(f"[*] Mẫu tập Đánh giá (Val)     : {len(val_dataset)} ảnh")

    # Khởi tạo mô hình
    model = EnhancedLeNet(num_classes=len(TARGET_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    save_dir = ensure_dir(CHECKPOINTS_DIR / "deep_learning")
    best_val_acc = 0.0
    best_ckpt_path = save_dir / "best_cnn.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}]", leave=False):
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

        scheduler.step()
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
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "arch": "EnhancedLeNet",
            }, best_ckpt_path)
            print(f"   -> ⭐ [BEST MODEL] Đã lưu trọng số tại: {best_ckpt_path} (Val Acc: {val_acc*100:.2f}%)")

    print("\n" + "=" * 65)
    print(f"✅ HUẤN LUYỆN LENET-5 CẢI TIẾN HOÀN TẤT THÀNH CÔNG!")
    print(f"[*] Trọng số tốt nhất đạt: {best_val_acc*100:.2f}% Val Accuracy")
    print(f"[*] Đường dẫn checkpoint  : {best_ckpt_path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced LeNet-5 for Traffic Sign Classification")
    parser.add_argument("--epochs", type=int, default=15, help="Số epochs huấn luyện (mặc định: 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (mặc định: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (mặc định: 0.001)")
    args = parser.parse_args()

    train_cnn(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
